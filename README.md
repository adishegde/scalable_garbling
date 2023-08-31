# Scalable Multiparty Garbling

Accompanying codebase for [ePrint:2023/099](https://eprint.iacr.org/2023/099).

**WARNING:** This is an academic proof-of-concept prototype, and is not suitable for use in production systems.
The code is provided solely for reproducibility and comparisons in future works.

## Contents
The repository consists of the following:
- [Scripts to estimate LPN parameters](<#lpn-parameters>)
- [Rust implementation of the semi-honest secure garbling protocol](<#semi-honest-secure-protocol>)
- [Scripts to estimate the costs for the maliciously secure garbling protocol](<#maliciously-secure-protocol>)

## Dependencies
- Rust `>=1.70`
- Python `>=3.9`

To install the required python packages run: `pip install -r requirements.txt`.

## LPN Parameters
LPN parameters can be computed using the [`gen_lpn_params.py`](https://github.com/adishegde/scalable_garbling/blob/benchmarks/scripts/gen_lpn_params.py) script.

Usage:
```sh
# Generate LPN parameters for the MPC protocol with
# - 80 bits of computational security,
# - 40 bits of statistical security,
# - 2^{-2} bias for the Bernoulli error distribution, and
# - computing over GF(2^{18}).
python gen_lpn_params.py 80 40 2 262144

# Use the --help option for more details.
```

## Semi-Honest Secure Protocol
The repository supports computing the following benchmarks for the semi-honest secure garbling protocol:
- Runtime and communication costs for the pre-processing phase
- Runtime and communication costs for the garbling phase
- Size of the pre-processing material

All benchmarks can be run independent of each other.
The only caveat is that benchmarking the pre-processing phase requires generating the appropriate binary super-invertible matrix which can be done using the [`gen_binary_supmat.py`](https://github.com/adishegde/scalable_garbling/blob/benchmarks/scripts/gen_binary_supmat.py) script.
The repository already includes binary super-invertible matrices for parameters used in the evaluation section of the paper.

For ease of usage, the repository also includes [Briston Fashion MPC circuit descriptions](https://homes.esat.kuleuven.be/~nsmart/MPC/) of some common circuits in `data/circ`.

Usage:
```sh
# --- Run unit tests ---
cargo test --release

# --- Build binaries ---
cargo build --release

# --- Setup ---
# The tmp directory will be used to store temporary log files.
mkdir data/tmp
# The out directory will be used to store the outputs of benchmarks.
mkdir data/out

# Going forward, we will be using the following protocol parameters in our
# examples:
# - Number of parties (n): 16
# - Corruption threshold (t): 5
# - Packing parameter (l): 3
# - LPN key length (L): 3
# - LPN ciphertext length (Q): 5
# Moreover, we will use the 64-bit subtraction circuit in our examples. All
# binaries and scripts support the '--help' option for more details on how to
# change parameter values and inputs for the benchmarks.

# --- Generate binary super-invertible matrix ---
# The script takes the ratio of total number of parties to number of corrupt
# parties as an argument. Ensure that the maximum number of corruptions, as
# output by the script, is at most the parameter 't' and the output minimum
# number of honest parties is at least 'n - t'.
#
# NOTE: The repository already contains binary super-invertible matrices for 
# some parameter values.
python scripts/gen_binary_supmat.py 16 3 --output_path ./data/binsup_mat

# --- Benchmark pre-processing phase ---
# The pre-processing phase can be benchmarked by running an instance of the
# 'preproc' binary with appropriate arguments for each party. In addition to
# the protocol parameters, the binary also requires a network file listing the
# address of each party; an example of which can be found at
# './data/localnet.txt'.
#
# The '--save' option can be used to save the benchmark stats for all parties
# in a single JSON file. The file will be created by the party with ID 0.
#
# When running all parties on a single machine, the 'run.sh' script automates
# running the binary for each party. The script only runs party 0 in the
# foreground, all other parties are run as background tasks. The output of all
# parties is available in 'data/tmp'.
./run.sh ./target/release/preproc 16 --net ./data/localnet.txt \
    --threshold 5 --packing-param 3 --binsup-mat ./data/binsup_mat/n16_t5.txt \
    --lpn-key-len 3 --lpn-mssg-len 5 \
    --circ ./data/circ/sub64.txt \
    --reps 3 --threads 2 --save ./data/out/preproc.json

# --- Benchmark garbling phase ---
# The garbling phase can be benchmarked by running an instance of the 'garble'
# binary with appropriate arguments for each party. As in the case of
# pre-processing, the binary also requires a network file listing the address
# of each party.
#
# As in the case of pre-processing, the '--save' option can be used to save the
# benchmark stats for all parties in a single JSON file created by the party 
# with ID 0.
#
# In the example below, we use the 'run.sh' script to run all parties on the 
# same machine.
./run.sh ./target/release/garble 16 --net ./data/localnet.txt \
    --threshold 5 --packing-param 3 \
    --lpn-key-len 3 --lpn-mssg-len 5 \
    --circ ./data/circ/sub64.txt \
    --reps 3 --threads 2 --save ./data/out/garble.json

# --- Size of pre-processing material ---
# The size of the pre-processing material can be obtained using the 'preproc_size'
# binary.
./target/release/preproc_size --circ ./data/circ/sub64.txt \
    --num-parties 16 --threshold 5 --packing-param 3
```

## Maliciously Secure Protocol
The communication cost and the computation cost of the maliciously secure garbling protocol can be estimated using the [`communication_estimates.py`](https://github.com/adishegde/scalable_garbling/blob/benchmarks/scripts/communication_estimates.py) and [`computation_estimates.py`](https://github.com/adishegde/scalable_garbling/blob/benchmarks/scripts/computation_estimates.py) scripts respectively.
The computation estimates rely upon the time required to carry out individual field operations.
These timings can be obtained by running `cargo bench` (which in turn uses [criterion](https://docs.rs/criterion/latest/criterion/)) as shown below, which benchmarks the performance of the [Fast Galois Field Arithmetic Library](http://web.eecs.utk.edu/~jplank/plank/papers/CS-07-593/), also used in the implementation of the semi-honest secure protocol.

Usage:
```sh
# --- Estimate communication costs ---
# Communication costs can be estimated using the 'communication_estimates.py'
# script. The script takes the number of input wires, AND gates and XOR gates
# as argument along with other protocol parameters. For the purpose of
# estimation, any inversion gates can be considered as XOR gates (since the
# latter are more expensive). Use the '--help' option for more details.
#
# In the example below, we estimate the communication cost for garbling AES-128
# with 128 parties, tolerating a corruption threshold of 31 and with the
# packing parameter set to 33.
python scripts/communication_estimates.py -n 128 -t 31 -l 33 256 6400 30263

# --- Estimate computation costs ---
# Computation costs can be estimated using the 'computation_estimates.py'
# script. The script takes the number of input wires, AND gates and XOR gates
# as argument along with other protocol parameters. For the purpose of
# estimation, any inversion gates can be considered as XOR gates (since the
# latter are more expensive). Use the '--help' option for more details.
#
# In the example below, we estimate the computation cost for garbling AES-128
# with 128 parties, tolerating a corruption threshold of 31 and with the
# packing parameter set to 33, where each party uses 2 threads.
python scripts/computation_estimates.py -n 128 -t 31 -l 33 --threads 2 256 6400 30263

# --- Benchmarking field operations ---
# As discussed earlier, the estimation of the computation costs rely on the
# time required to carry out individual field operations. These can be obtained
# using the command below and used with the script using the '--time_add' and
# '--time_mult' options. The default values in the script correspond to
# benchmarks computed using the evaluation environment described in the paper.
cargo bench
```
