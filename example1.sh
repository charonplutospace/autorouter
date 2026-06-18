#!/bin/bash
echo "Run simple example 1"
python3 src/extract_pads.py --input_kicad_pcb "examples/projtest1.kicad_pcb" --output_json "temp/kicad_test1.json" --settings_json "settings/settings_kicad.json"
python3 src/autorouter.py --to_route_json "temp/kicad_test1.json" --routed_results_json "temp/kicad_routed_1.json" --settings_json "settings/settings_kicad.json"
# routed really bad with errors
