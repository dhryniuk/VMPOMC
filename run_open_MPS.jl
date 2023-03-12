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
const h=1.0 #transverse field strength
const γ=1.0 #spin decay rate
const α=0
const N=8
const dim = 2^N
χ=6 #bond dimension
const burn_in = 0

MPOMC.set_parameters(N,χ,J,h,γ,α,burn_in)

const basis=generate_bit_basis_reversed(N)
#const basis=generate_bit_basis(N)
#display(basis)

H = make_bit_Hamiltonian(N,J,h,basis)

E_n, Psi_n = eigen(H)
GS = E_n[1]
println("Ground state E/N: ", GS/N)


#A_init=rand(ComplexF64, χ,χ,2)
A_init=rand(Float64, χ,χ,2)
A=copy(A_init)
A=reshape(A,χ,χ,2)
A.-=0.5
#A/=maximum(A) #very important
#A/=N

display(A)

V_init = rand(Float64, χ, 2)
V = copy(V_init)
display(V)

#∇,E=calculate_MPS_gradient(MPOMC.params,A,basis) 
#display(E)
#error()

list_of_E = Array{Float64}(undef, 0)

δ = 0.02

Q=1
F=0.99
ϵ=0.01

@time begin
    for k in 1:300
        E=0
        acc::Float64=0
        for i in 1:20
            #display(A)

            new_A=zeros(Float64, χ,χ,2)
            ∇_A,∇_V,E=calculate_open_MPS_gradient(MPOMC.params,A,V,basis) 
            #∇,E=MC_calculate_MPS_gradient(MPOMC.params,A,10*k) 
            #∇,E,acc=MC_SR_calculate_MPS_gradient(MPOMC.params,A,50*k+300,ϵ) 
            #∇,E,acc=MPS_calculate_gradient(MPOMC.params,A,50*k+300,ϵ) 
            #∇,E=distributed_SR_calculate_MC_MPS_gradient(MPOMC.params,A,10*k,ϵ)
            #display(new_A)
            #display(∇)

            #update bulk tensors:
            ∇_A./=maximum(∇_A)
            new_A = A - 1.0*δ*F^k*∇_A
            global A = new_A
            A/=maximum(A)

            #update boundary tensors:
            ∇_V./=maximum(∇_V)
            new_V = V - 1.0*δ*F^k*∇_V
            global V = new_V
            V/=maximum(V)
        end

        #L = calculate_mean_local_Lindbladian(MPOMC.params,l1,A,basis)
        #println("k=$k: ", real(L), " ; ", mz, " ; ", mx)
        #println("k=$k: ", abs(real(E)-GS), " ; acc_rate=", round(acc,sigdigits=2))#/MPOMC.params.N)

        println("k=$k: ", real(E)/N, " ; acc_rate=", round(acc,sigdigits=2))#/MPOMC.params.N)

        push!(list_of_E,E)
    end
end
println("Ground state E/N: ", GS/N)


#@code_warntype  MC_SR_calculate_MPS_gradient(MPOMC.params,A,100,ϵ) 
#@code_warntype distributed_SR_calculate_MC_MPS_gradient(MPOMC.params,A,100,ϵ)