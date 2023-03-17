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


#Define constants:
const J=0.5 #interaction strength
const h=1.0 #transverse field strength
const γ=1.0 #spin decay rate
const α=0
const N=2
const dim = 2^N
χ=2 #bond dimension
const burn_in = 0

MPOMC.set_parameters(N,χ,J,h,γ,α,burn_in)

#const basis=generate_bit_basis_reversed(N)
const basis=generate_bit_basis(N)
#display(basis)

H = make_bit_Hamiltonian(N,J,h,basis)
display(H)
#error()

h1 = h*sx

E_n, Psi_n = eigen(H)
GS = E_n[1]
println("Ground state E/N: ", GS/N)


#A_init=rand(ComplexF64, χ,χ,2)
A_init=rand(Float64, χ,χ,2)
A=copy(A_init)
A=reshape(A,χ,χ,2)

#∇,E=calculate_MPS_gradient(MPOMC.params,A,basis) 
#display(E)
#display(∇)
#error()
#A.-=0.5
#A./=maximum(abs.(A))
A=normalize_MPS(MPOMC.params, A)



#∇,E=calculate_MPS_gradient(MPOMC.params,A,basis) 
#display(E)
#error()

list_of_E = Array{Float64}(undef, 0)

δ = 0.03

Q=1
F=0.95
ϵ=0.01

@time begin
    for k in 1:100
        E=0
        acc::Float64=0
        for i in 1:1
            #display(A)

            new_A=zeros(Float64, χ,χ,2)
            ∇,E=Exact_MPS_gradient(MPOMC.params,A,basis,h1) 
            #∇,E=SGD_MPS_gradient(MPOMC.params,A,10*k,h1) 
            #∇,E,acc=SR_MPS_gradient(MPOMC.params,A,10*k,ϵ,h1) 
            #∇,E,acc=MPS_calculate_gradient(MPOMC.params,A,50*k+300,ϵ) 
            #∇,E=distributed_SR_calculate_MC_MPS_gradient(MPOMC.params,A,10*k,ϵ)
            #display(new_A)
            #display(∇)
            ∇./=maximum(abs.(∇))
            new_A = A - 1.0*δ*F^k*∇
            global A = new_A
            #display(∇)
            global A=normalize_MPS(MPOMC.params, A)
            #global A./=tensor_normalize_MPS(MPOMC.params, A)
            #A/=maximum(abs.(A))
        end

        #L = calculate_mean_local_Lindbladian(MPOMC.params,l1,A,basis)
        #println("k=$k: ", real(L), " ; ", mz, " ; ", mx)
        #println("k=$k: ", abs(real(E)-GS), " ; acc_rate=", round(acc,sigdigits=2))#/MPOMC.params.N)

        println("k=$k: ", E/N, " ; acc_rate=", round(acc,sigdigits=2))#/MPOMC.params.N)

        push!(list_of_E,real(E))
    end
end

println("Ground state E/N: ", GS/N)

#@code_warntype  MC_SR_calculate_MPS_gradient(MPOMC.params,A,100,ϵ) 
#@code_warntype distributed_SR_calculate_MC_MPS_gradient(MPOMC.params,A,100,ϵ)