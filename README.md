# DESCRIPTION

CCubes is a Boolean minimizer designed to efficiently minimize functions with multiple outputs. For the moment, it only accepts a PLA file as an input and produces a PLA file as output.

The minimization process is extremely efficient with incompletely specified functions with many don't cares. It compares the ON set minterms with the OFF set minterms, and for this reasons it needs a ".type fr" PLA file.

It employs a bottom-up search strategy, starting from the simplest combinations of 1 input, 2 inputs etc. and gradually increases the complexity. After each such level, it tries to solve the minterm matrix and stops as soon as there is a high probability that no new decisive prime implicants will be found at the next level.

Solving the minterm matrix coverage is an NP-hard problem beyond the scope of CCubes. Exact solutions are provided by Gurobi, an industry level power optimization solver. For compilation, Gurobi's path is hardcoded in the Makefile and it needs to be changed manually, according to the user's installation and operating system. At runtime, Gurobi will search for a valid license, freely offered for academic research.

The default option, in case Gurobi is not available, is to solve the minterm matrix coverage using a custom implementation of a parallel processing Lagrangian relaxation method, based on the sources of BOOM (courtesy of Petr Fišer). More solving methods will be added.

If no weights are applied, the combination of prime implicants that cover the ON set minterms is the quickest exact method, roughly equivalent to `espresso -Dso` type of output, although it produces a much more efficient circuit especially with an exact optimization.

Two weighting options are available, for instance the default `-w1` for weight based on complexity levels (prime implicants with lower number of literals will be given more weight). The option `-w2` adds additional weight if a prime implicant is shared between multiple outputs.

The option `-p` aims to spend additional time collecting a pool of possible solutions from the solver and decide which solution is best by comparing their shared prime implicants with those from the other outputs. Activating this parameter to more than 1 solutions will automatically set the weighting to `-w2`. Its rationale is that the solver may select different shared prime implicants, even though all have an equal (weighted) contribution to optimality.

Unlike other minimizers like Espresso (usually single threaded), CCubes is scalable and can handle larger problem instances more efficiently. Where possible, it will use a parallel search process using available CPU cores. Theoretically, its scalability can be extended to distributed computing environments, allowing it to tackle even larger instances by using multiple machines.

For parallel search it uses OpenMP, which users need to make sure it is installed and linked at compile time.

There is a minimal help system integrated into CCubes. Users can enable debug output by setting the debug level via command line options. This will provide insights into the internal workings of the minimization process and can be useful for troubleshooting and understanding the behavior of the tool.

The debug levels are preliminary, and more detailed logging functionality will be added in the future.

For very large problem instances, CCubes can save its state into a binary checkpoint file and exit, when a certain `-t` time limit is reached. The process can be resumed later from the checkpoint file, allowing users to continue the minimization process without re-specifying the input and output files. Even when `-r` resuming from a checkpoint, a further successive time limit can be specified to save another intermediate checkpoint. If the binary checkpoint file is not specified, it will default to `chk_<basename(source)>.bin`.Preliminary tests show that creating intermediate checkpoints more than double the performance in execution time, likely because of memory allocations.

Compile the binary according to your system using `make`, with various options and customization indicated in the Makefile.

The destination .pla file is optional. If not specified, a `ccubes_<basename(source)>.pla` file will be created in the current directory.

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
                          0 (default) Lagrangian relaxation heuristic
                          1 Gurobi exact
  -p<number>          : decide from a pool of up to <number> equally optimal solutions
  -d<level>[=<file>]  : incremental debug information
                          0 (default) errors + warnings
                          1 errors + warnings + info
                          2 everything (trace)
  -t<sec>[=<file>]    : time limit after which a checkpoint will be saved to <file>
  -r=<file>           : resume from checkpoint file
  -i=<file>           : inspect checkpoint (print progress and metadata)
  -h, --help          : show this help message
```