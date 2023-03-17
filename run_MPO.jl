include("MPOMC.jl")
using .MPOMC
using NPZ
using Plots
using LinearAlgebra
using BenchmarkTools
#using Cthulhu
import Random
Random.seed!(1)

#Define constants:
const J=0.5 #interaction strength
const h=1.0 #transverse field strength
const γ=1.0 #spin decay rate
const α=0
const N=20
const dim = 2^N
χ=8 #bond dimension
const burn_in = 2

MPOMC.set_parameters(N,χ,J,h,γ,α, burn_in)

#Make single-body Lindbladian:
const l1 = make_one_body_Lindbladian(h*sx,γ*sm)
#display(l1)

const basis=generate_bit_basis_reversed(N)


A_init=rand(ComplexF64, χ,χ,2,2)
A=copy(A_init)
A=reshape(A,χ,χ,4)
B=deepcopy(A)

list_of_L = Array{Float64}(undef, 0)
list_of_LB= Array{Float64}(undef, 0)

old_L=1
old_LB=1

δ = 0.02

#Levy_dist = truncated(Levy(1.0, 0.001),0,10)
N_MC=200
Q=1
QB=1
F=0.99
ϵ=0.1

#@time begin
@profview begin
    for k in 1:500
        L=0
        for l in 1:10
            new_A=zeros(ComplexF64, χ,χ,4)
            #∇,L=Exact_MPO_gradient(MPOMC.params,A,l1,basis)
            #∇,L=SGD_MPO_gradient(MPOMC.params,A,l1,500)#0+50*k)
            ∇,L=SR_MPO_gradient(MPOMC.params,A,l1,500,ϵ)#0+50*k)
            ####∇,L=SR_calculate_MC_gradient_full(MPOMC.params,A,l1,100,0,ϵ)#0+50*k)
            ∇./=maximum(abs.(∇))
            new_A = A - δ*F^(k)*∇#.*(1+0.5*rand())

            global A = new_A
            global A = normalize_MPO(MPOMC.params, A)
        end

        println("k=$k: ", real(L))
        #global old_L = L

        push!(list_of_L,L)
    end
end

@code_warntype SR_MPO_gradient(MPOMC.params,A,l1,100,ϵ)

#@code_warntype  SR_calculate_MC_gradient_full(MPOMC.params,A,l1,100,0,ϵ) 