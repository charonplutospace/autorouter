import argparse
import contextlib
import io
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import random
import re
import traceback


def setup(to_route_json, settings_json):
    with open(to_route_json) as file:
    # with open('examples/example_to_route_2.json') as file:
        data = json.load(file)

        """Build sample pads, nets, and board constraints for routing."""
        print("python setup function: entry point when imported as a module")

    all_pads = {
        k: v
        for k, v in data.items()
        if re.fullmatch(r"pad\d+", k)
    }
    all_pads = data["all_pads"]
    connections = data["connections"]
    board = data["board"]

    with open(settings_json) as file:
        settings = json.load(file)

    margin = settings["min_spacing"]
    via = settings["via"]
    via_margin = settings["via_margin"]
    pad_to_via_margin = settings["pad_to_via_margin"]
    board_margin = settings["board_margin"]

    #margin = data["margin"]
    #via = data["via"]
    #via_margin = data["via_margin"]
    #pad_to_via_margin = data["pad_to_via_margin"]
    #board_margin = data["board_margin"]

    seed_env = os.getenv("ROUTER_SEED")
    seed_env = settings["seed_env"]
    seed = int(seed_env) #if seed_env else None


    return (
        all_pads,
        connections,
        margin,
        board,
        via,
        via_margin,
        pad_to_via_margin,
        board_margin,
        seed,
    )


def rect_to_cells(rect):
    """Expand an axis-aligned rectangle dict into a set of grid cells."""
    cells = set()
    for x in range(rect["xmin"], rect["xmax"] + 1):
        for y in range(rect["ymin"], rect["ymax"] + 1):
            cells.add((x, y))
    return cells


def in_board(cell, board):
    """Return True when agrid cell is inside board bondaries."""
    x, y = cell
    return (
        board["board_x_min"] <= x <= board["board_x_max"]
        and board["board_y_min"] <= y <= board["board_y_max"]
    )


def in_board_with_margin(cell, board, board_margin):
    """Return True when a grid cell is inside board boundaries with edge margin."""
    x, y = cell
    return (
        board["board_x_min"] + board_margin <= x <= board["board_x_max"] - board_margin
        and board["board_y_min"] + board_margin
        <= y
        <= board["board_y_max"] - board_margin
    )


def chebyshev_neighbors(cell, radius):
    """Return all cells within Chebyshev radius around a center cell"""
    x, y = cell
    out = set()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            out.add((x + dx, y + dy))
    return out


def normalize_connections(connections):
    """Normalize connection specs and validate pad list and width fields."""
    net_specs = {}
    for net_name, net_data in connections.items():
        if isinstance(net_data, list):
            pads = list(net_data)
            width = 1
        elif isinstance(net_data, dict):
            pads = list(net_data.get("pads", []))
            width = int(net_data.get("width", 1))
        else:
            raise ValueError(f"Invalid connection data for net {net_name}: {net_data}")

        if len(pads) < 2:
            raise ValueError(
                f"Net '{net_name}' must connect at least 2 pads, got {len(pads)}"
            )
        if width < 1:
            raise ValueError(f"Net '{net_name}' must have width >= 1, got {width}")

        net_specs[net_name] = {"pads": pads, "width": width}
    return net_specs


def normalize_pads(all_pads):
    """Normalize pad specs and derive layer-specific occupied cells."""
    pads_spec = {}
    for pad_name, pad_data in all_pads.items():
        if isinstance(pad_data, dict) and "rect" in pad_data:
            rect = pad_data["rect"]
            layers = list(pad_data.get("layers", [1, 2]))
        else:
            # Backwards compatibility: pad is just a rectangle and it exist on both layers.
            rect = pad_data
            layers = [1, 2]

        valid_layers = sorted({int(l) for l in layers})
        if not valid_layers:
            raise ValueError(
                f"Pad '{pad_name}' must specify at least one layer, got {layers}"
            )
        if any(l not in (1, 2) for l in valid_layers):
            raise ValueError(
                f"Pad '{pad_name}' has invalid layers {layers}, only layers 1 and 2 are supported"
            )

        cells = rect_to_cells(rect)
        layer_cells = {
            0: set(cells) if 1 in valid_layers else set(),
            1: set(cells) if 2 in valid_layers else set(),
        }
        pads_spec[pad_name] = {
            "rect": rect,
            "layers": valid_layers,
            "cells_by_layer": layer_cells,
        }
    return pads_spec


def expand_cells(cells, radius, board):
    """Dilate a cell set by Chebyshev radius while clipping to board."""
    if radius <= 0:
        return {c for c in cells if in_board(c, board)}

    out = set()
    for c in cells:
        for n in chebyshev_neighbors(c, radius):
            if in_board(n, board):
                out.add(n)
    return out


def make_via_cells(center, via_size, board, board_margin=0):
    """Create a square via footprint cells centred at a coordinate."""
    cx, cy = center
    half_size = via_size // 2
    cells = set()
    for dx in range(via_size):
        for dy in range(via_size):
            c = (cx - half_size + dx, cy - half_size + dy)
            if not in_board_with_margin(c, board, board_margin):
                return None
            cells.add(c)
    return cells


def octile_distance(a, b):
    """Calculate octile distance for 8-connected grid routing"""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (2**0.5 - 1) * min(dx, dy)


def can_place_center(center, route_radius, board, forbidden_cells, board_margin=0):
    """Check weather a route center and its width footprint can be placed."""
    for c in chebyshev_neighbors(center, route_radius):
        if not in_board_with_margin(c, board, board_margin):
            return False
        if c in forbidden_cells:
            return False
    return True


def route_pair_q_learning_2layer(
    starts,
    targets,
    forbidden_by_layer,
    board,
    route_radius,
    via_size,
    via_keepout_cells,
    pad_to_via_keepout_cells,
    board_margin,
    episodes=1200,
    alpha=0.24,
    gamma=0.92,
    epsilon_start=1.0,
    epsilon_end=0.05,
    max_steps=500,
):
    """Route a one source-target pair on two layers using Q-learning and vias."""

    actions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
    ]  # right, left, down, up, stay
    opposite = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4}
    q = {}  # Q-values for state-action pairs maybe move Stefan Persson

    starts = list(starts)
    targets = list(targets)
    if not starts or not targets:
        return None

    target_xy = {(x, y) for x, y, _ in targets}

    def qv(state_key, action_idx):
        return q.get((state_key, action_idx), 0.0)

    def best_action(state_key):
        vals = [qv(state_key, idx) for idx in range(len(actions) + 1)]
        mv = max(vals)
        idxs = [idx for idx, v in enumerate(vals) if v == mv]
        return random.choice(idxs)

    def is_goal(state):
        return state in targets

    best_path = None
    best_len = float("inf")

    for ep in range(episodes):
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (
            ep / max(1, episodes - 1)
        )
        state = random.choice(starts)
        prev_action = -1
        path = [state]

        for _ in range(max_steps):
            state_key = (state, prev_action)
            if random.random() < epsilon:
                action_idx = random.randrange(len(actions) + 1)
            else:
                action_idx = best_action(state_key)

            x, y, layer = state
            reward = -1.0
            next_state = state
            next_prev_action = prev_action
            done = False

            if action_idx == len(actions):
                via_cells = make_via_cells((x, y), via_size, board, board_margin)
                other_layer = 1 - layer
                if via_cells is None:
                    reward = -10.0
                elif via_cells & forbidden_by_layer[0]:
                    reward = -9.0
                elif via_cells & forbidden_by_layer[1]:
                    reward = -9.0
                elif via_cells & via_keepout_cells:
                    reward = -8.0
                elif via_cells & pad_to_via_keepout_cells:
                    reward = -8.5
                else:
                    next_state = (x, y, other_layer)
                    next_prev_action = len(actions)
                    reward = -1.5
                    if is_goal(next_state):
                        reward = 120.0
                        done = True
            else:
                dx, dy = actions[action_idx]
                nx, ny = x + dx, y + dy
                c = (nx, ny)
                if not can_place_center(
                    c, route_radius, board, forbidden_by_layer[layer], board_margin
                ):
                    reward = -8.0
                else:
                    next_state = (nx, ny, layer)
                    next_prev_action = action_idx
                    prev_dist = min(octile_distance((x, y), t) for t in target_xy)
                    new_dist = min(octile_distance((nx, ny), t) for t in target_xy)
                    reward = -0.8 + 1.0 * (prev_dist - new_dist)

                    if prev_action == action_idx:
                        reward += 0.15
                    elif prev_action != -1:
                        reward -= 0.35
                    if prev_action != -1 and opposite.get(prev_action) == action_idx:
                        reward -= 0.6

                    if is_goal(next_state):
                        reward = 120.0
                        done = True
            next_key = (next_state, next_prev_action)
            next_best = max(qv(next_key, i) for i in range(len(actions) + 1))
            old = qv(state_key, action_idx)
            q[(state_key, action_idx)] = old + alpha * (
                reward + gamma * next_best - old
            )

            state = next_state
            prev_action = next_prev_action
            path.append(state)

            if done:
                if len(path) < best_len:
                    best_len = len(path)
                    best_path = path[:]
                break
    if best_path is None:
        return None

    compact = [best_path[0]]
    for s in best_path[1:]:
        if s != compact[-1]:
            compact.append(s)
    return compact


def route_nets_with_rl_2layers(
    all_pads,
    connections,
    margin,
    board,
    via_size,
    via_margin,
    pad_to_via_margin=0,
    board_margin=0,
    max_restarts=45,
):
    """Route all nets across two layers with retries and keepout constraints."""
    net_specs = normalize_connections(connections)
    pad_specs = normalize_pads(all_pads)
    all_pad_names = set(pad_specs.keys())

    net_pad_names = {net: set(cfg["pads"]) for net, cfg in net_specs.items()}

    all_pad_cells = set()
    for spec in pad_specs.values():
        all_pad_cells.update(spec["cells_by_layer"][0])
        all_pad_cells.update(spec["cells_by_layer"][1])

    pad_to_via_keepout_cells = expand_cells(all_pad_cells, pad_to_via_margin, board)

    board_margin_cells = set()
    for x in range(board["board_x_min"], board["board_x_max"] + 1):
        for y in range(board["board_y_min"], board["board_y_max"] + 1):
            c = (x, y)
            if in_board(c, board) and (
                not in_board_with_margin(c, board, board_margin)
            ):
                board_margin_cells.add(c)

    net_names = list(net_specs.keys())
    last_error = None

    for attempt in range(max_restarts):
        net_order = net_names[:]
        if attempt > 0:
            random.shuffle(net_order)

        copper_by_layer = {
            0: {net: set() for net in net_specs},
            1: {net: set() for net in net_specs},
        }

        for net, cfg in net_specs.items():
            for pad_name in cfg["pads"]:
                copper_by_layer[0][net].update(pad_specs[pad_name]["cells_by_layer"][0])
                copper_by_layer[1][net].update(pad_specs[pad_name]["cells_by_layer"][1])

        via_centers = []
        via_cells_all = set()
        via_keepout_cells = set()
        routed_paths = {net: [] for net in net_specs}

        failed = False

        for net_name in net_order:
            cfg = net_specs[net_name]
            pads = cfg["pads"]
            route_radius = cfg["width"] - 1

            connected_states = set()
            for layer_idx in (0, 1):
                for c in pad_specs[pads[0]]["cells_by_layer"][layer_idx]:
                    connected_states.add((c[0], c[1], layer_idx))

            for target_pad_name in pads[1:]:
                forbidden_by_layer = {0: set(), 1: set()}
                for layer in (0, 1):
                    for other_net, cells in copper_by_layer[layer].items():
                        if other_net == net_name:
                            continue
                        for c in cells:
                            forbidden_by_layer[layer].update(
                                chebyshev_neighbors(c, margin)
                            )

                blocked_pad_names = (
                    all_pad_names - net_pad_names[net_name] - {target_pad_name}
                )

                for p in blocked_pad_names:
                    forbidden_by_layer[0].update(pad_specs[p]["cells_by_layer"][0])
                    forbidden_by_layer[1].update(pad_specs[p]["cells_by_layer"][1])

                forbidden_by_layer[0].update(board_margin_cells)
                forbidden_by_layer[1].update(board_margin_cells)

                target_states = set()
                for layer_idx in (0, 1):
                    for c in pad_specs[target_pad_name]["cells_by_layer"][layer_idx]:
                        target_states.add((c[0], c[1], layer_idx))

                for layer in (0, 1):
                    forbidden_by_layer[layer] -= copper_by_layer[layer][net_name]
                    forbidden_by_layer[layer] -= {
                        (x, y) for x, y, l in target_states if l == layer
                    }
                    forbidden_by_layer[layer] = {
                        c for c in forbidden_by_layer[layer] if in_board(c, board)
                    }

            path3d = route_pair_q_learning_2layer(
                starts=connected_states,
                targets=target_states,
                forbidden_by_layer=forbidden_by_layer,
                board=board,
                route_radius=route_radius,
                via_size=via_size,
                via_keepout_cells=via_keepout_cells,
                pad_to_via_keepout_cells=pad_to_via_keepout_cells,
                board_margin=board_margin,
            )

            if path3d is None:
                failed = True
                last_error = (
                    f"Could not route net '{net_name}' from pad '{pads[0]}' to pad '{target_pad_name}' after {attempt+1} attempts. "
                    f"Consider increasing board size, reducing margins, or adjusting via parameters. Error details: {last_error}"
                )
                break

            path_l0 = []
            path_l1 = []
            path_vias = []
            prev = path3d[0]
            for cur in path3d[1:]:
                x, y, layer = cur
                if layer == 0:
                    path_l0.append((x, y))
                else:
                    path_l1.append((x, y))

                if cur[2] != prev[2]:
                    vcenter = (x, y)
                    path_vias.append(vcenter)
                    vcells = make_via_cells(vcenter, via_size, board, board_margin)
                    via_cells_all.update(vcells)
                    via_centers.append(vcenter)
                    via_keepout_cells.update(expand_cells(vcells, via_margin, board))
                prev = cur

            copper_l0 = expand_cells(path_l0, route_radius, board)
            copper_l1 = expand_cells(path_l1, route_radius, board)

            copper_by_layer[0][net_name].update(copper_l0)
            copper_by_layer[1][net_name].update(copper_l1)

            routed_paths[net_name].append(
                {
                    "layer1": path_l0,
                    "layer2": path_l1,
                    "vias": path_vias,
                    "path3d": path3d,
                }
            )

            connected_states.update(path3d)
            connected_states.update(target_states)

            if failed:
                break

        if not failed:
            return routed_paths, copper_by_layer, via_centers, via_cells_all

    raise RuntimeError(last_error or "Routing failed after retries-")


def validate_clearance_2layer(copper_by_layer, margin):
    """Validate inter-net clearance independently on each layer."""
    for layer in (0, 1):
        nets = list(copper_by_layer[layer].keys())
        for i in range(len(nets)):
            for j in range(i + 1, len(nets)):
                a = nets[i]
                b = nets[j]
                for ca in copper_by_layer[layer][a]:
                    for cb in copper_by_layer[layer][b]:
                        if max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1])) <= margin:
                            return False, (layer + 1, a, b, ca, cb)
    return True, None


def validate_via_clearance(via_centers, via_margin):
    """Validate via-to-via spacing using Chebyshev distance margin."""
    for i in range(len(via_centers)):
        for j in range(i + 1, len(via_centers)):
            a = via_centers[i]
            b = via_centers[j]
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= via_margin:
                return False, (a, b)
    return True, False


def validate_pad_to_via_clearance(via_cells_all, all_pads, board, pad_to_via_margin):
    """Validate that the via copper stays away from pads by configured margin."""
    pad_specs = normalize_pads(all_pads)
    all_pads_cells = set()
    for spec in pad_specs.values():
        all_pads_cells.update(spec["cells_by_layer"][0])
        all_pads_cells.update(spec["cells_by_layer"][1])

    forbidden = expand_cells(all_pads_cells, pad_to_via_margin, board)
    hit = via_cells_all & forbidden
    if hit:
        return False, sorted(hit)[:20]
    return True, None


def validate_board_margin_2layer(copper_by_layer, via_cell_all, board, board_margin):
    """Validate traces and vias stay inside board minus edge margin."""
    bad = []
    for layer_idx in (0, 1):
        for net_name, cells in copper_by_layer[layer_idx].items():
            for c in cells:
                if not in_board_with_margin(c, board, board_margin):
                    bad.append(("trace", layer_idx + 1, net_name, c))

    for c in via_cell_all:
        if not in_board_with_margin(c, board, board_margin):
            bad.append(("via", 0, "via", c))
    if bad:
        return False, bad[:20]
    return True, None


def check_foregin_pad_overlaps_2layer(copper_by_layer, all_pads, connections):
    """Report any net copper touching pads that do not belong to that net."""
    net_specs = normalize_connections(connections)
    pad_specs = normalize_pads(all_pads)
    overlaps = {}

    for net_name in net_specs:
        owned = set(net_specs[net_name]["pads"])
        foregin = set(all_pads.keys()) - owned
        hits = []
        for pad_name in sorted(foregin):
            touched = set()
            touched.update(
                copper_by_layer[0][net_name] & pad_specs[pad_name]["cells_by_layer"][0]
            )
            touched.update(
                copper_by_layer[1][net_name] & pad_specs[pad_name]["cells_by_layer"][1]
            )
            if touched:
                hits.append({"pad": pad_name, "cells": sorted(touched)})
        if hits:
            overlaps[net_name] = hits

    return overlaps


def check_pad_layer_violations(copper_by_layer, all_pads):
    """Report traces that touch on layers where those pads are disallowed."""
    pad_specs = normalize_pads(all_pads)
    violations = []

    for layer_idx in (0, 1):
        layer_no = layer_idx + 1
        disallowed_pad_cells = {}
        for pad_name, spec in pad_specs.items():
            if layer_no not in spec["layers"]:
                disallowed_pad_cells[pad_name] = rect_to_cells(spec["rect"])

        for net_name, cells in copper_by_layer[layer_idx].items():
            for pad_name, pad_cells in disallowed_pad_cells.items():
                hit = cells & pad_cells
                if hit:
                    violations.append(
                        {
                            "layer": layer_no,
                            "net": net_name,
                            "pad": pad_name,
                            "cells": sorted(hit),
                        }
                    )
    return violations


def write_log_report(report_lines, file_path="log.txt"):
    """Write run summary lines to plain text log file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")


def render_ascii_layer(
    board, all_pads, copper_layer, net_symbols, layer_idx, via_cells_all=None
):
    """Render one board layers as an ASCII with pads, traces, and vias."""
    x_min = board["board_x_min"]
    x_max = board["board_x_max"]
    y_min = board["board_y_min"]
    y_max = board["board_y_max"]

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    grid = [["." for _ in range(width)] for _ in range(height)]

    pad_specs = normalize_pads(all_pads)
    pad_cells = set()
    for spec in pad_specs.values():
        for x, y in spec["cells_by_layer"][layer_idx]:
            gx = x - x_min
            gy = y_max - y
            grid[gy][gx] = "P"
            pad_cells.add((x, y))

    for net_name, cells in copper_layer.items():
        s = net_symbols[net_name]
        for x, y in cells:
            if (x, y) in pad_cells:
                continue
            gx = x - x_min
            gy = y_max - y
            grid[gy][gx] = s

    if via_cells_all:
        for x, y in via_cells_all:
            if (x, y) in pad_cells:
                continue
            gx = x - x_min
            gy = y_max - y
            grid[gy][gx] = "V"

    lines = ["    " + "".join(str(x % 10) for x in range(x_min, x_max + 1))]
    for row_i, row in enumerate(grid):
        y = y_max - row_i
        lines.append(f"{y:>3}" + "".join(row))
    return "\n".join(lines)


def plot_board_2layer(board, all_pads, copper_by_layer, via_cells_all):
    """Plot both copper layers and vias using matplotlib when available."""
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            with contextlib.redirect_stdout(io.StringIO()):
                None
    except Exception as exc:
        print(f"\nMatplotlib plot skipped: {exc}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    x_min = board["board_x_min"]
    x_max = board["board_x_max"]
    y_min = board["board_y_min"]
    y_max = board["board_y_max"]

    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf" "#7f7f7f",
    ]

    pad_specs = normalize_pads(all_pads)

    for li, ax in enumerate(axes):
        layer = li
        ax.set_aspect("equal")
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
        ax.set_xticks(range(x_min, x_max + 1))
        ax.set_yticks(range(y_min, y_max + 1))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_title(f"Layer {layer + 1}")

        for pad_name, spec in pad_specs.items():
            if (layer + 1) not in spec["layers"]:
                continue
            rect = spec["rect"]
            w = rect["xmax"] - rect["xmin"] + 1
            h = rect["ymax"] - rect["ymin"] + 1
            patch = Rectangle(
                (rect["xmin"] - 0.5, rect["ymin"] - 0.5),
                w,
                h,
                facecolor="#f59e0b",
                edgecolor="black",
                linewidth=1.0,
                alpha=0.6,
            )
            ax.add_patch(patch)
            ax.text(
                (rect["xmin"] + rect["xmax"]) / 2,
                (rect["ymin"] + rect["ymax"]) / 2,
                pad_name,
                ha="center",
                va="center",
                fontsize=7,
            )

        for i, net_name in enumerate(sorted(copper_by_layer[layer].keys())):
            color = palette[i % len(palette)]
            xs = [c[0] for c in copper_by_layer[layer][net_name]]
            ys = [c[1] for c in copper_by_layer[layer][net_name]]
            ax.scatter(
                xs, ys, s=40, marker="s", color=color, alpha=0.55, label=net_name
            )

        if via_cells_all:
            vx = [c[0] for c in via_cells_all]
            vy = [c[1] for c in via_cells_all]
            ax.scatter(vx, vy, s=48, marker="s", color="black", alpha=0.85, label="via")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=5)
    plt.tight_layout()
    plt.show()


def main_autorouter(to_route_json, routed_results_json, settings_json, log_txt):
    """Execute routing flow, print chekcs, visualize, and write log report."""
    report_lines = []

    try:
        (
            all_pads,
            connections,
            margin,
            board,
            via_size,
            via_margin,
            pad_to_via_margin,
            board_margin,
            seed,
        ) = setup(to_route_json, settings_json)

        if seed is not None:
            random.seed(seed)
            seed_line = f"Deterministic seed: {seed}"
            print(seed_line)
            report_lines.append(seed_line)

            paths, copper_by_layer, via_centers, via_cell_all = (
                route_nets_with_rl_2layers(
                    all_pads,
                    connections,
                    margin,
                    board,
                    via_size,
                    via_margin,
                    pad_to_via_margin,
                    board_margin,
                )
            )

            report_lines.append("Routing result: SUCCESS")
            report_lines.append(f"Nets: {', '.join(sorted(paths.keys()))}")
            report_lines.append(f"Total vias: {len(via_centers)}")

            print("\nRouted traces (RL, 2 layers):")
            for net, net_paths in paths.items():
                print(f"{net}:")
                for i, p in enumerate(net_paths, start=1):
                    path_line = (
                        f"  path_{i}: L1={len(p['layer1'])} cells, "
                        f"L2={len(p['layer2'])} cells, vias={p['vias']}"
                    )
                    print(path_line)
                    report_lines.append(f"{net} {path_line.strip()}")

            ok, details = validate_clearance_2layer(copper_by_layer, margin)
            layer_check = "PASS" if ok else f"FAIL -> {details}"
            print("\nLayer clearance check:", layer_check)
            report_lines.append(f"Layer clearance check: {layer_check}")

            v_ok, v_details = validate_via_clearance(via_centers, via_margin)
            via_check = "PASS" if v_ok else f"FAIL -> {v_details}"
            print("Via clearance check:", via_check)

            report_lines.append(f"Via clearance check: {via_check}")

            pv_ok, pv_details = validate_pad_to_via_clearance(
                via_cell_all, all_pads, board, pad_to_via_margin
            )
            pad_via_check = "PASS" if pv_ok else f"fail -> {pv_details}"
            print("Pad-to-via clearance check:", pad_via_check)
            report_lines.append(f"Pad-to_via clearance check: {pad_via_check}")

            bm_ok, bm_details = validate_board_margin_2layer(
                copper_by_layer, via_cell_all, board, board_margin
            )
            board_margin_check = "PASS" if bm_ok else f"fail -> {bm_details}"
            print("Board margin check:", board_margin_check)
            report_lines.append(f"Board margin check: {board_margin_check}")

            overlaps = check_foregin_pad_overlaps_2layer(
                copper_by_layer, all_pads, connections
            )
            if not overlaps:
                print("Foregin-pad overlpa check: PASS")
                report_lines.append("Foregin-pad overlap check: PASS")
            else:
                print("Foregin-pad overlap check: FAIL")
                report_lines.append("Foregin-pad overlap check: FAIL")
                for net_name, hits in overlaps.items():
                    print(f"  {net_name} overlaps:")
                    for hit in hits:
                        print(f"  {hit['pad']}: {hit['cells']}")
                        report_lines.append(
                            f"   {net_name} overlaps {hit['pad']}: {hit['cells']}"
                        )
            pad_layer_violations = check_pad_layer_violations(copper_by_layer, all_pads)
            if not pad_layer_violations:
                print("Pad layer violation check: PASS")
                report_lines.append("Pad-layer violation check: PASS")
            else:
                print("Pad-layer violation check: PASS")
                report_lines.append("Pad-layer violation check: PASS")
                for v in pad_layer_violations:
                    print(
                        f"   layer {v['layer']} net {v['net']} touches {v['pad']}: {v['cells']}"
                    )
                    report_lines.append(
                        f"   layer {v['layer']} net {v['net']} touches {v['pad']}: {v['cells']}"
                    )
            net_symbols = {
                net: str((i + 1) % 10)
                for i, net in enumerate(sorted(copper_by_layer[0].keys()))
            }

            print("\nASCII Layer 1 (P=pad, digits=net, V=via):")
            print(
                render_ascii_layer(
                    board, all_pads, copper_by_layer[0], net_symbols, 0, via_cell_all
                )
            )

            print("\nASCII Layer 2 (P=pad, digits=net, V=via):")
            print(
                render_ascii_layer(
                    board, all_pads, copper_by_layer[1], net_symbols, 1, via_cell_all
                )
            )

            print("Legend:")
            for net in sorted(net_symbols):
                print(f"   {net_symbols[net]} -> {net}")

            plot_board_2layer(board, all_pads, copper_by_layer, via_cell_all)

            with open(routed_results_json, "w") as file:
                json.dump(paths, file)

    except Exception as exec:
        report_lines.append("Routed result: FAILED")
        report_lines.append(f"Error: {exec}")
        write_log_report(report_lines, log_txt)
        traceback.print_exc()
        raise

    write_log_report(report_lines, log_txt)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example program"
    )
    parser.add_argument("--to_route_json", default="examples/example_to_route_1.json")
    parser.add_argument("--routed_results_json", default="examples/routed_results.json")
    parser.add_argument("--settings_json", default="settings/settings_example.json")
    parser.add_argument("--log_txt", default="temp/example_to_route_1_log.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    to_route_json = args.to_route_json
    routed_results_json = args.routed_results_json
    settings_json = args.settings_json
    log_txt = args.log_txt

    print(f"Input file is: {to_route_json}")
    print(f"Output file is: {routed_results_json}")
    print(f"Settings file is: {settings_json}")
    main_autorouter(to_route_json, routed_results_json, settings_json, log_txt)


if __name__ == "__main__":
    main()
