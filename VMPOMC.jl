module VMPOMC

using LinearAlgebra
#using Distributed
using TensorOperations
using SparseArrays
using ArnoldiMethod
using Random
using MPI
#using Statistics



#Basic utilities:
include("utils/projector.jl")
include("utils/parameters.jl")
include("utils/workspace.jl")
include("utils/utils.jl")
include("utils/mpi.jl")

 
#MPS/MPO backend:
include("MPO/MPO.jl")


#Optimizers:
include("Optimisers/optimizer.jl")


#Monte Carlo samplers:
include("Samplers/MPO_Metropolis.jl")


#Optimizer routines:
include("Optimisers/diagonal_operators.jl")
include("Optimisers/MPO/common.jl")
include("Optimisers/MPO/Exact.jl")
include("Optimisers/MPO/SGD.jl")
include("Optimisers/MPO/SR.jl")
include("Optimisers/optimizer_dispatch.jl")


#Observables:
include("MPO/observables.jl")

end