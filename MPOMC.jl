module MPOMC

using LinearAlgebra
using Distributed
using TensorOperations
using SparseArrays
using ArnoldiMethod
#using Distributions
#using Plots
#using NPZ
#using DelimitedFiles



#Other utilities:
include("utils/utils.jl")


#Exact diagonalisation routines:
include("ED/ED_Ising.jl")
include("ED/ED_Lindblad.jl")

#export generate_bit_basis_reversed
#export make_one_body_Lindbladian, id, sx, sy, sz, sp, sm


#Matrix Product functions:
include("MPO/MPS.jl")
include("MPO/MPO.jl")
include("MPO/observables.jl")


#Samplers:
include("Samplers/Metropolis.jl")


#Optimisers:
include("Optimisers/MPS/Exact.jl")
include("Optimisers/MPS/SGD.jl")
include("Optimisers/MPS/SR.jl")
include("Optimisers/MPS/LM.jl")
#include("Optimisers/MPS/distributed_SR.jl")

include("Optimisers/MPO/Exact.jl")
include("Optimisers/MPO/SGD.jl")
include("Optimisers/MPO/SR.jl")


#export calculate_gradient, calculate_MC_gradient_full
#export SR_calculate_gradient, SR_calculate_MC_gradient_full


params=parameters(0,0,0,0,0,0,0,0)

#addprocs(10)


end