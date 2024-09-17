#!/usr/bin/env bash

END=$1

for i in $(seq -w 1 $END)
do 
    echo "-------------------- Run: $i --------------------"
    python create_scene_json.py "brownian_motion_$i" 0.01 5.0
    python ideal_gas.py "scenes/brownian_motion_$i.json"
done
