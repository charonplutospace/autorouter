#!/usr/bin/env python
"""Extract pads from KiCad PCB file and export as JSON."""
import argparse
import json
from kicad_parser.kicad_pcb import KicadPCB

MILLIINCH = 0.0254 # mm

def extract_pads_json(filename, output_file, settings_json):
    """Extract all pads on F.Cu layer and save as JSON."""

    with open(settings_json) as file:
        settings = json.load(file)


    print(f"Loading PCB file: {filename}")
    try:
        pcb = KicadPCB.load(filename)
    except Exception as e:
        print(f"Error loading file: {e}")
        return False

    pads = []
    pad_count = 0

    my_pads = {}
    my_nets = {}

    # Get all footprints
    footprints = pcb["footprint"]

    # Iterate through all footprints
    for fp_idx, footprint in enumerate(footprints):
        # Get footprint reference
        #if footprint["property"][0][1] == '"R1"':
        #    print('found R1')
        try:
            properties = footprint["property"]
            fp_ref = properties[0][3] if len(properties[0]) > 3 else f"FP{fp_idx}"
        except:
            fp_ref = f"FP{fp_idx}"

        # Get footprint position (absolute)
        try:
            fp_at = footprint["at"]
            fp_x, fp_y = float(fp_at[0]), float(fp_at[1])
        except:
            fp_x, fp_y = 0, 0

        # Get pads
        try:
            fp_pads = footprint["pad"]
        except:
            continue

        # Iterate through pads
        for pad in fp_pads:
            try:
                # Check if on F.Cu layer
                layers = pad["layers"]
                has_fcu = any(
                    "F.Cu" in str(l)
                    for l in (layers if isinstance(layers, list) else [layers])
                )

                has_bcu = any(
                    "B.Cu" in str(l)
                    for l in (layers if isinstance(layers, list) else [layers])
                )

                has_allcu = any(
                    "*.Cu" in str(l)
                    for l in (layers if isinstance(layers, list) else [layers])
                )
                layers = []
                if has_allcu:
                    layers = [1,2]
                elif has_fcu:
                    layers = [1]
                elif has_bcu:
                    layers = [2]


                if has_fcu or has_bcu or has_allcu:
                    # Extract data
                    pad_name = str(pad[0]).strip('"')  # Remove quotes
                    at = pad["at"]
                    size = pad["size"]

                    # Calculate absolute position
                    abs_x = fp_x + float(at[0])
                    abs_y = fp_y + float(at[1])

                    # Get net safely
                    try:
                        net = str(pad["net"]).strip('"')  # Remove quotes
                    except:
                        net = "unconnected"
                    insert = "pad" + str(pad_count)
                    c = 1/(MILLIINCH*settings["min_wire_width"])
                    if((pad[1] + " " + pad[2]) == "smd roundrect"):
                        my_pads.update(
                            {
                                insert: {
                                    "rect": {"xmin": int(c*(abs_x-float(size[0])/2)), "xmax": int(c*(abs_x+float(size[0])/2)), "ymin": int(c*(abs_y-float(size[1])/2)), "ymax": int(c*(abs_y+float(size[1])/2))},
                                    "layers": layers,
                                }
                            }
                        )
                    elif((pad[1] + " " + pad[2]) == "thru_hole circle"):
                        my_pads.update(
                            {
                                insert: {
                                    "rect": {"xmin": int(c*(abs_x-float(size[0])/2)), "xmax": int(c*(abs_x+float(size[0])/2)), "ymin": int(c*(abs_y-float(size[1])/2)), "ymax": int(c*(abs_y+float(size[1])/2))},
                                    "layers": layers,
                                }
                            }
                        )

                    else:
                        raise("only smd roundrect and thru_hole circle are implmented.")

                    if "unconnected" in pad["net"]:
                        pass
                    else:
                        my_nets.setdefault(pad["net"], {"pads": [], "width": 1})
                        my_nets[pad["net"]]["pads"].append("pad"+str(pad_count))
                    pad_count=pad_count+1

            except Exception as e:
                print(f"Error processing pad: {e}")
                continue

    print(f"\nFound {pad_count} pads on F.Cu layer")

    # Get size
    try:
        size_raw = pcb["gr_rect"]
        c = 1/(MILLIINCH*settings["min_wire_width"])
        board = {

            "board_x_min": round(c*size_raw["start"][0]),
            "board_x_max": round(c*size_raw["end"][0]),
            "board_y_min": round(c*size_raw["start"][1]),
            "board_y_max": round(c*size_raw["end"][1])
        }
    except Exception as e:
                print(f"Error processing board size: {e}")

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump({"pad_count": pad_count, "all_pads": my_pads, "connections": my_nets, "board": board}, f, indent=2)

    print(f"Saved to {output_file}")
    return True



def parse_args():
    parser = argparse.ArgumentParser(
        description="Example program"
    )
    parser.add_argument("--input_kicad_pcb", default="examples/projtest1.kicad_pcb")
    parser.add_argument("--output_json", default="pads.json")
    parser.add_argument("--settings_json", default="settings/settings_kicad.json")
    return parser.parse_args()


def main():
    args = parse_args()
    input_kicad_pcb =args.input_kicad_pcb
    output_json = args.output_json
    settings_json = args.settings_json
    print(f"Input file is: {input_kicad_pcb}")
    print(f"Output file is: {output_json}")
    print(f"Settings file is: {settings_json}")
    extract_pads_json(input_kicad_pcb, output_json, settings_json)


if __name__ == "__main__":
    main()
