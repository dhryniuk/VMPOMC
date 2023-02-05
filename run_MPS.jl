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
const N=10
const dim = 2^N
χ=6 #bond dimension

MPOMC.set_parameters(N,χ,J,h,γ,α)

#const basis=generate_bit_basis_reversed(N)
const basis=generate_bit_basis(N)
#display(basis)

H = make_bit_Hamiltonian(N,J,h,basis)

E_n, Psi_n = eigen(H)
GS = E_n[1]
println("Ground state: ", GS)

#const l1=h*sx

#A_init=rand(ComplexF64, χ,χ,2)
A_init=rand(Float64, χ,χ,2)
A=copy(A_init)
A=reshape(A,χ,χ,2)

display(A)

#∇,E=calculate_MPS_gradient(MPOMC.params,A,basis) 
#display(E)
#error()

list_of_E = Array{Float64}(undef, 0)

δ = 0.01

Q=1
F=0.995
ϵ=0.01

@time begin
    for k in 1:500
        E=0
        for i in 1:5
            #display(A)

            new_A=zeros(Float64, χ,χ,2)
            #∇,E=calculate_MPS_gradient(MPOMC.params,A,basis) 
            #∇,E=MC_calculate_MPS_gradient(MPOMC.params,A,10*k) 
            ∇,E=MC_SR_calculate_MPS_gradient(MPOMC.params,A,10*k,ϵ) 
            #display(new_A)
            #display(∇)
            ∇./=maximum(∇)
            new_A = A - 1.0*δ*F^k*∇
            global A = new_A
            #global A./=normalize_MPS(MPOMC.params, A)
            A/=maximum(A)
        end

        #L = calculate_mean_local_Lindbladian(MPOMC.params,l1,A,basis)
        #println("k=$k: ", real(L), " ; ", mz, " ; ", mx)
        println("k=$k: ", abs(real(E)-GS))#/MPOMC.params.N)

        push!(list_of_E,E)
    end
end

#@code_warntype  MC_SR_calculate_MPS_gradient(MPOMC.params,A,100,ϵ) 