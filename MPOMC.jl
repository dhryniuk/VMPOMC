module MPOMC

using LinearAlgebra
using Distributed
using TensorOperations
using SparseArrays
using ArnoldiMethod
using Random
using MPI



#Other utilities:
include("utils/utils.jl")
include("utils/mpi.jl")


#Exact diagonalisation routines:
include("ED/ED_Ising.jl")
#include("ED/ED_Lindblad.jl")
include("ED/Lindbladians.jl")
include("ED/operators.jl")
include("ED/observables.jl")
include("ED/utils.jl")


#MPS/MPO backend:
include("MPO/MPS.jl")
include("MPO/MPO.jl")
include("MPO/observables.jl")



#Optimizers:
include("Optimisers/optimizer.jl")



#Monte Carlo samplers:
include("Samplers/MPS_Metropolis.jl")
include("Samplers/MPO_Metropolis.jl")



#Optimizer routines:
include("Optimisers/gradient.jl")
include("Optimisers/eigen_operations.jl")

include("Optimisers/MPS/Exact.jl")
include("Optimisers/MPS/SGD.jl")
include("Optimisers/MPS/SR.jl")
#include("Optimisers/MPS/LM.jl")

include("Optimisers/MPO/Exact.jl")
include("Optimisers/MPO/SGD.jl")
include("Optimisers/MPO/SR.jl")
include("Optimisers/MPO/MPI_SR.jl")

include("Optimisers/MPO/Exact_two_body.jl")
include("Optimisers/MPO/SGD_two_body.jl")
include("Optimisers/MPO/SR_two_body.jl")



#params=parameters(0,0,0,0,0,0,0,0,0,0,0)

end