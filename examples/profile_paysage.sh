#!/bin/bash

# Run profiling, output to paysage.cprof

set -eu

echo "Running paysage profiling"
echo "========================="

timestamp=$(date +"%Y%m%d_%H%M%S")

pip3 install snakeviz
python3 -m cProfile -o paysage_${timestamp}.cprof ./profile_paysage.py
snakeviz paysage_${timestamp}.cprof
