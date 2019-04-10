Release 0.4.2.0

New feature QuasiRandomSolver added. The QuasiRandomSolver provides a randomized gridsampling. This means that depending
on max_iterations a grid over all numerical parameter is spanned and each cell is populated with a random value within the
the cell bounds for numerical and a random draw for each categorical parameter. This ensures a random sampling of the 
parameter space and a good space coverage without random cluster building. The solver also supports normal and
loguniform sampling.  