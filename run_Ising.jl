using Distributed

#addprocs(4)
println(nprocs())
println(nworkers())

@everywhere include("MPOMC.jl")
@everywhere using .MPOMC
using NPZ
using Plots
using LinearAlgebra
#using TensorOperations
#using Distributions
#using Revise
import Random
using JLD


#Vincentini parameters: γ=1.0, J=0.5, h to be varied.

#Define constants:
const Jx= 0.0 #interaction strength
const Jy= 0.0 #interaction strength
const J = 0.5 #interaction strength
const hx= 0.2 #transverse field strength
const hz= 0.0 #transverse field strength
const γ = 1.0 #spin decay rate
const α=0
const N=8
const dim = 2^N
χ=4 #bond dimension
const burn_in = 0

MPOMC.set_parameters(N,χ,Jx,Jy,J,hx,hz,γ,α, burn_in)

#Make single-body Lindbladian:
const l1 = conj( make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm) )
#const l1 = ( make_one_body_Lindbladian(-hx*sx-hz*sz,sqrt(γ)*sm) )
#display(l1)

#const basis=generate_bit_basis_reversed(N)


Random.seed!(1)
A_init=rand(ComplexF64, χ,χ,2,2)
A=deepcopy(A_init)
A=reshape(A,χ,χ,4)

list_of_L = open("list_of_L.out", "w"); close(list_of_L)
list_of_mag = open("list_of_mag.out", "w"); close(list_of_mag)

function set_beta(it, β_inf, decay_rate)
    return β_inf +(1-β_inf)/(it*decay_rate+1)
end

δ::Float16 = 0.03

N_MC=2
Q=1
F::Float16=0.98
ϵ::Float64=0.1
β::Float64=0.6

display(A_init)

#@profview begin
@time begin
    for k in 1:100
        L=0;LB=0
        acc::Float64=0
        for i in 1:10

            new_A=zeros(ComplexF64, χ,χ,4)

            ∇,L,acc=gradient("SR",A,l1,MPOMC.params, N_MC=10*4*χ^2+k,ϵ=ϵ,parallel=true)

            ∇./=maximum(abs.(∇))
            new_A = A - δ*F^(k)*∇

            global A = new_A
            global A = normalize_MPO(MPOMC.params, A)

        end

        Af = reshape(A,χ,χ,2,2) 
        Af_dagger = conj.(permutedims(Af,[1,2,4,3]))

        mx = real( 0.5*( tensor_calculate_magnetization(MPOMC.params,Af,sx) + tensor_calculate_magnetization(MPOMC.params,Af_dagger,sx) ) )
        my = real( 0.5*( tensor_calculate_magnetization(MPOMC.params,Af,sy) + tensor_calculate_magnetization(MPOMC.params,Af_dagger,sy) ) )
        mz = real( 0.5*( tensor_calculate_magnetization(MPOMC.params,Af,sz) + tensor_calculate_magnetization(MPOMC.params,Af_dagger,sz) ) )

        println("k=$k: ", real(L)/N, " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))

        list_of_L = open("list_of_L.out", "a")
        println(list_of_L, L)
        close(list_of_L)
        list_of_mag = open("list_of_mag.out", "a")
        println(list_of_mag, mx, ",", my, ",", mz)
        close(list_of_mag)

        save("MPO_density_matrix.jld", "MPO_density_matrix", Af)
    end
end