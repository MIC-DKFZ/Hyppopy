Release 0.5.0.0

- settings structure changed, additional settings now can be addded as additional entries in the config dict or using the methods add_setting or set_settings
- sections solver and custom in config dict are removed completely
- use_solver setting in config dict is renamed to solver
- hyperparameter type now a native type, not a string anymore
- automatic consistency check between config and solver conditions, each solver defines now it's interface which is checked when executing the solver throwing exceptions if the project instance and the solvers interface doesn't work together
- bayesOpt solver removed, extremely slow and not very good

Release 0.4.2.0

New feature QuasiRandomSolver added. The QuasiRandomSolver provides a randomized gridsampling. This means that depending
on max_iterations a grid over all numerical parameter is spanned and each cell is populated with a random value within the
the cell bounds for numerical and a random draw for each categorical parameter. This ensures a random sampling of the 
parameter space and a good space coverage without random cluster building. The solver also supports normal and
loguniform sampling.  