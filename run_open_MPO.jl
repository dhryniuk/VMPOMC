#using Distributed
#addprocs(4)
#println(nprocs())
#println(nworkers())
#@everywhere include("MPOMC.jl")
#@everywhere using .MPOMC
include("MPOMC.jl")
using .MPOMC

using NPZ
using Plots
using LinearAlgebra
#using Distributions
#using Revise
#using ProfileView
import Random
Random.seed!(1)
using BenchmarkTools
using DoubleFloats

#Define constants:
const J=0.5 #interaction strength
const h=0.5 #transverse field strength
const γ=0.5 #spin decay rate
const α=0
const N=3
const dim = 2^N
χ=6 #bond dimension
const burn_in = 0

MPOMC.set_parameters(N,χ,J,h,γ,α,burn_in)

const basis=generate_bit_basis_reversed(N)
#const basis=generate_bit_basis(N)
#display(basis)

const l1 = make_one_body_Lindbladian(MPOMC.params,sx,sm)

A_init=rand(ComplexF64, χ,χ,2,2)
A=copy(A_init)
A=reshape(A,χ,χ,4)

display(A)

V_init = rand(ComplexF64, χ,2,2)
V = copy(V_init)
V=reshape(V,χ,4)
display(V)

#∇,E=calculate_MPS_gradient(MPOMC.params,A,basis) 
#display(E)
#error()

list_of_E = Array{Float64}(undef, 0)

δ = 0.01

Q=1
F=0.996
ϵ=0.01

@time begin
    for k in 1:1500
        E=0
        acc::Float64=0
        for i in 1:1
            #display(A)

            new_A=zeros(Float64, χ,χ,2)
            #∇_A,∇_V,E=Exact_fMPO_gradient(MPOMC.params,A,V,l1,basis) 
            ∇_A,∇_V,E=calculate_open_MPO_gradient(MPOMC.params,A,V,l1,basis) 

            #update bulk tensors:
            ∇_A./=maximum(abs.(∇_A))
            new_A = A - 1.0*δ*F^k*∇_A
            global A = new_A
            #A/=maximum(A)
            global A./=normalize_open_MPO(MPOMC.params, A, V)

            #update boundary tensors:
            ∇_V./=maximum(abs.(∇_V))
            new_V = V - 1.0*δ*F^k*∇_V
            global V = new_V
            #V/=maximum(V)
            global V./=normalize_open_MPO(MPOMC.params, A, V)
        end

        #L = calculate_mean_local_Lindbladian(MPOMC.params,l1,A,basis)
        #println("k=$k: ", real(L), " ; ", mz, " ; ", mx)
        #println("k=$k: ", abs(real(E)-GS), " ; acc_rate=", round(acc,sigdigits=2))#/MPOMC.params.N)

        println("k=$k: ", real(E)/N, " ; acc_rate=", round(acc,sigdigits=2))#/MPOMC.params.N)

        push!(list_of_E,real(E))
    end
end

#fMPO()
#@code_warntype  MC_SR_calculate_MPS_gradient(MPOMC.params,A,100,ϵ) 
#@code_warntype distributed_SR_calculate_MC_MPS_gradient(MPOMC.params,A,100,ϵ)

function xcalculate_z_magnetization(params, A::Array{ComplexF64,3}, V::Array{ComplexF64})
    mp=transpose(V[:,dINDEX[(1,1)]] + V[:,dINDEX[(0,0)]])
    for _ in 2:params.N-1
        mp*=-A[:,:,dINDEX[(1,1)]] + A[:,:,dINDEX[(0,0)]]
    end
    mp*=V[:,dINDEX[(1,1)]] + V[:,dINDEX[(0,0)]]
    return mp
end