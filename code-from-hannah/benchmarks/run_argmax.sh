#!/bin/bash

cd progs/Programs/Source
chmod +x load_data.sh
./load_data.sh

/benchmarks/MP-SPDZ/compile.py -X -R 64 -M argmax.mpc

/benchmarks/MP-SPDZ/dealer-ring-party.x $1 argmax -h 172.18.0.2 -N 3 -v
