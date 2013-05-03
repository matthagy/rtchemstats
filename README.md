rtchemstats - Real-Time Chemical Statistics
==================================================================
Python library to compute statistical description of chemical systems 
in real-time without the need to write trajectories to disk. This allows
one to collect high resolution distribution functions with roughly 4 orders
of magnitude less disk space. In practice 100GB trajectory files have
been replaced by 50MB state files.

This librariy is based around StatComputers which extract the necesary
information from a sequence of simulation configurations to compute
a specific distribution function. StatComputers are memory efficient
in that they extract the minimimal information from each configuration.
The internal analysis code is implemented in Cython for efficency.
StatComputers are also restartable, in that they can be pickled to disk
as part of a simulation restart file.

StatComputers are implemented for the following distribution functions
 o Isotropic pair correlation function (i.e. h(r) = g(r) - 1)
 o 2-Dimensional pair correlation function 
 o 2-Dimensional orienation correlation function
 o Bond angle about a common atom distribution
 o Mean squared displacment function
 o Velocity autocorrelation function
 o Reversible bond duration distribution

LICENSE
------------------------------------------------------------------
rtchemstats is Licensed under the permissive Apache Licensed.
See included LICENSE file.
