# DESCRIPTION

CCubes is a Boolean minimizer designed to efficiently minimize functions with multiple outputs. For the moment, it only accepts a PLA file as an input and produces a PLA file as output.

The minimization process is extremely efficient with incompletely specified functions with many don't cares. It compares the ON set minterms with the OFF set minterms, and for this reasons it needs a ".type fr" PLA file.

It employs a bottom-up search strategy, starting from the simplest combinations of 1 input, 2 inputs etc. and gradually increases the complexity. After each such level, it tries to solve the minterm matrix and stops as soon as there is a high probability that no new decisive prime implicants will be found at the next level.

Solving the minterm matrix is an NP-hard problem beyond the scope of CCubes, which relies on Gurobi, an industry level optimization solver. For the time being, Gurobi's path is hardcoded in the Makefile and it needs to be changed manually. At runtime, the variable GUROBI_HOME needs to be exported because Gurobi will search for a valid license, which Gurobi offers freely for academic research.

In case Gurobi is not available, the minterm matrix is currently solved using a simple greedy method as a fallback option. More solving methods will be added.

The actual combination of Prime Implicants needed to cover the ON set minterms is decided by the solver, if no weights are applied. This is the quickest exact method, roughly equivalent to `espresso -Dso` type of output although it produces a much more efficient circuit.

Other weighting options are available, for instance  `-w1` for weight based on complexity levels (prime implicants with lower number of literals will be given more weight). The option `-w2` adds additional weight if a prime implicant is shared between multiple outputs. The option `-s12` (unimplemented yet) aims to spend additional time collecting a pool of possible solutions from the solver and decide which solution is best by comparing their shared prime implicants with those from the other outputs. This might be useful since it is quite possible that different shared prime implicants are selected by the solver, despite them having an equal (weighted) contribution to optimality.

Unlike other minimizers like Espresso (usually single threaded), CCubes is scalable and can handle larger problem instances more efficiently. Where possible, it will use a parallel search process using available CPU cores. Theoretically, its scalability can be extended to distributed computing environments, allowing it to tackle even larger instances by using multiple machines.

For parallel search it uses OpenMP, which users need to make sure it is installed and linked at compile time.

There is a minimal help system integrated into CCubes. Users can enable debug output by setting the debug level via command line options. This will provide insights into the internal workings of the minimization process and can be useful for troubleshooting and understanding the behavior of the tool.

The debug levels are preliminary, and more detailed logging functionality will be added in the future.

Compile the binary according to your system using `make`, with various options indicated in the Makefile.

The destination .pla file is optional. If not specified, a .pla file prefixed with "ccubes_" will be created in the current directory.

# USAGE

```
ccubes [options] source.pla [dest.pla]
Options:
  -k<number>          : start searching from level k
  -e<number>          : end criterion (default +1 level with the same minima)
  -b<number>          : bits per word, either 8, 16, 32, 64 (default) or 128
  -c<number>          : number of CPU cores / threads to use, if OpenMP available
  -w<number>          : weights applied to the prime implicants:
                          0 no weight
                          1 (default) weight based on complexity levels k
                          2 additional weight if shared between outputs
  -s<number>          : how to solve the covering problem:
                          0 greedy (fast)
                          11 (default) Gurobi exact
                          12 (not implemented), Gurobi solution pool to select the one
                              with the highest number of shared prime implicants
  -d<level>[=<file>] : incremental debug information
                          0 error only
                          1 errors + warnings
                          2 errors + warnings + info
                          3 everything (trace)
  -h, --help          : show this help message
```