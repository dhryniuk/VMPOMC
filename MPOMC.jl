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

#Exact diagonalisation routines:
include("ED/ED_Ising.jl")
include("ED/ED_Lindblad.jl")

#export generate_bit_basis_reversed
#export make_one_body_Lindbladian, id, sx, sy, sz, sp, sm


#Matrix Product functions:
include("MPO/MPO.jl")
include("MPO/MPS.jl")


#Samplers:
include("Samplers/Metropolis.jl")


#Optimisers:
include("Optimisers/SGD.jl")
include("Optimisers/SR.jl")
include("Optimisers/gradient.jl")

#export calculate_gradient, calculate_MC_gradient_full
#export SR_calculate_gradient, SR_calculate_MC_gradient_full


#Other utilities:
include("utils/utils.jl")


params=parameters(0,0,0,0,0,0,0)

#addprocs(10)


end