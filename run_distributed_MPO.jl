using Distributed
#addprocs(4)
@everywhere include("MPOMC.jl")
@everywhere using .MPOMC
using NPZ
using Plots
using LinearAlgebra
using BenchmarkTools
import Random
Random.seed!(1)

#Define constants:
const J = 0.5 #interaction strength
const hx= 0.25 #transverse field strength
const hz= 0.0 #transverse field strength
const γ = 1.0 #spin decay rate
const α=0
const N=4
const dim = 2^N
χ=3 #bond dimension
const burn_in = 0

MPOMC.set_parameters(N,χ,J,hx,hz,γ,α,burn_in)

"""
const l1 = make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm)
display(l1)
error()


L=sparse_DQIM(MPOMC.params, "periodic")
vals, vecs = eigen_sparse(L)
ρ=reshape(vecs,2^N,2^N)
ρ=round.(ρ,digits = 12)
ρ./=tr(ρ)

Mx=real( magnetization(sx,ρ,MPOMC.params) )
println("True x-magnetization is: ", Mx)

My=real( magnetization(sy,ρ,MPOMC.params) )
println("True y-magnetization is: ", My)

Mz=real( magnetization(sz,ρ,MPOMC.params) )
println("True z-magnetization is: ", Mz)
error()
"""

#Make single-body Lindbladian:
#const l1 = make_one_body_Lindbladian(MPOMC.params,sx,sm)
const l1 = make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm)
#display(l1)


A_init=rand(ComplexF64, χ,χ,2,2)
A=deepcopy(A_init)
A=reshape(A,χ,χ,4)

list_of_L = Array{Float64}(undef, 0)

δ::Float16 = 0.05

#Levy_dist = truncated(Levy(1.0, 0.001),0,10)
N_MC=200
Q=1
F::Float16=0.97
ϵ::Float16=0.1

@time begin
#@profview begin
    for k in 1:100
        L=0
        acc::Float64=0
        for l in 1:10
            new_A=zeros(ComplexF64, χ,χ,4)
            ∇,L,acc=distributed_SR_MPO_gradient(A,l1,10*4*χ^2+k,ϵ, MPOMC.params)
            ∇./=maximum(abs.(∇))
            new_A = A - δ*F^(k)*∇#.*(1+0.5*rand())

            global A = new_A
            global A = normalize_MPO(MPOMC.params, A)
        end

        #println("k=$k: ", real(L), " ; acc_rate=", round(acc*100,sigdigits=2), "%")

        mx = calculate_x_magnetization(MPOMC.params,A)
        my = calculate_y_magnetization(MPOMC.params,A)
        mz = calculate_z_magnetization(MPOMC.params,A)

        println("k=$k: ", real(L)/N, " ; acc_rate=", round(acc*100,sigdigits=2), "%", " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))


        push!(list_of_L,L)
    end
end

#error()

L=sparse_DQIM(MPOMC.params, "periodic")
vals, vecs = eigen_sparse(L)
ρ=reshape(vecs,2^N,2^N)
ρ=round.(ρ,digits = 12)
ρ./=tr(ρ)

Mx=real( magnetization(sx,ρ,MPOMC.params) )
println("True x-magnetization is: ", Mx)

My=real( magnetization(sy,ρ,MPOMC.params) )
println("True y-magnetization is: ", My)

Mz=real( magnetization(sz,ρ,MPOMC.params) )
println("True z-magnetization is: ", Mz)
