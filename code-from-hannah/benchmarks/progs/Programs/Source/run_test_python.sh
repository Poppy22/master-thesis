#!/bin/bash

/benchmarks/MP-SPDZ/compile.py -X -R 64 -M test_python.py

python3 $1 test_python -h 172.18.0.2 -N 1 -v