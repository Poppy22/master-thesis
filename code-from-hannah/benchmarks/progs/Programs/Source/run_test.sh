#!/bin/bash

/benchmarks/MP-SPDZ/compile.py -X -R 64 -M test.mpc

/benchmarks/MP-SPDZ/dealer-ring-party.x $1 test -h 172.18.0.2 -N 3 -v -I