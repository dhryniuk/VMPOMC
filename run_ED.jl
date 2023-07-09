include("VMPOMC.jl")
using .VMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random


#Define constants:
const Jx= 0.0 #interaction strength
const Jy= 0.0 #interaction strength
const J = 0.5 #interaction strength
const hx= 1.0 #transverse field strength
const hz= 0.0 #transverse field strength
const γ = 1.0 #spin decay rate
const γ_d = 0.0 #spin decay rate
const α=0
#const N=10
#χ=8 #bond dimension
const burn_in = 0

#set values from command line optional parameters:
N = parse(Int64,ARGS[1])
χ = 0

const dim = 2^N

params = Parameters(N,2^N,χ,Jx,Jy,J,hx,hz,γ,γ_d,α, 0)

@time begin
L = sparse_DQIM(params, "periodic")
end

@time begin
vals, vecs = eigen_sparse(L)
end

ρ=reshape(vecs,2^N,2^N)
ρ./=tr(ρ)
#ρ=round.(ρ,digits = 12)

Mx=real( magnetization(sx,ρ,params) )
println("True x-magnetization is: ", Mx)

My=real( magnetization(sy,ρ,params) )
println("True y-magnetization is: ", My)

Mz=real( magnetization(sz,ρ,params) )
println("True z-magnetization is: ", Mz)