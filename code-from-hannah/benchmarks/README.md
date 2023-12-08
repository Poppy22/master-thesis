## Runtime and communication benchmarks

The implementation uses the [MP-SPDZ](https://github.com/data61/MP-SPDZ) framework for benchmarking secure multi-party computation protocols. For simplicity, we only describe steps necessary that build upon the general MP-SPDZ setup, which is automated by the scripts:
```code/benchmarks/setup_mpspdz.sh``` or ```code/benchmarks/setup_mpspdz_ubuntu.sh``` for the corresponding operating system.

This code is provided as an experimental implemenation for testing purposes and should not be used in a productive environment.

As published, running all steps by executing `code/benchmarks/run_argmax.sh` on three machines simultaneously will benchmark an artificial 16-point dataset in the semi-honest security model for honest majority in the dealer-ring setting. It uses a ring size of 64 bits. In line 11, replace 'machine1' with the ip address of your first machine, and adjust the number '0' that follows to '1' in the script on your second machine and to '2' in the script on your third machine. 

To benchmark other dataset sizes, ```code/benchmarks/progs/Programs/Source/prep_data.py``` must be edited to read a different dataset from the ones in the directory ```code/benchmarks/data```, replacing the file name in line 3. The datasets found there were generated using the ```code/benchmarks/progs/Programs/Source/prep_data.py``` script, by randomly generating the number of integer values from some range specified there. Furthermore, the specified dataset size indicated in line 7 of ```code/benchmarks/progs/Programs/Source/argmax.mpc``` must also be adapted to the correct number before re-running `code/benchmarks/run_argmax.sh`.

To benchmark other ring sizes, it is necessary to edit the ```code/benchmarks/MP-SPDZ/CONFIG.mine``` file to include the line ```MOD = -DRING_SIZE=<bit length>``` and then re-run
```console
make clean
make setup
make -j dealer-ring-party.x
```
Then, the file should be compiled with the corresponding ring size in line 9 of `code/benchmarks/run_argmax.sh`, after which this script can be run again.
