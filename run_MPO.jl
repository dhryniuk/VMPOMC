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
const J = 0.0 #interaction strength
const hx= 0.0 #transverse field strength
const hz= 0.0 #transverse field strength
const γ = 0.5 #spin decay rate
const α=0
const N=2
const dim = 2^N
χ=2 #bond dimension
const burn_in = 0

MPOMC.set_parameters(N,χ,J,hx,hz,γ,α,burn_in)

#Make single-body Lindbladian:
const l1 = make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm)
#display(l1)

const basis=generate_bit_basis_reversed(N)


A_init=rand(ComplexF64, χ,χ,2,2)
A=deepcopy(A_init)
A=reshape(A,χ,χ,4)

list_of_L = Array{Float64}(undef, 0)

δ::Float16 = 0.03

#Levy_dist = truncated(Levy(1.0, 0.001),0,10)
N_MC=200
Q=1
F::Float16=0.9
ϵ::Float16=0.1

#@time begin
@profview begin
    for k in 1:100
        L=0
        acc::Float64=0
        for l in 1:10
            new_A=zeros(ComplexF64, χ,χ,4)
            #∇,L=Exact_MPO_gradient(A,l1,basis,MPOMC.params)
            #∇,L,acc=SGD_MPO_gradient(A,l1,5*4*χ^2+k,MPOMC.params)
            #∇,L,acc=partial_SGD_MPO_gradient(MPOMC.params,A,l1,10*4*χ^2+k)
            #∇,L,acc=umbrella_SGD_MPO_gradient(0.5,MPOMC.params,A,l1,5*4*χ^2+k)
            ∇,L,acc=SR_MPO_gradient(A,l1,10*4*χ^2+k,ϵ, MPOMC.params)#0+50*k)
            ####∇,L=SR_calculate_MC_gradient_full(MPOMC.params,A,l1,100,0,ϵ)#0+50*k)
            ∇./=maximum(abs.(∇))
            new_A = A - δ*F^(k)*∇#.*(1+0.5*rand())

            global A = new_A
            global A = normalize_MPO(MPOMC.params, A)
        end

        #L = calculate_mean_local_Lindbladian(MPOMC.params, l1, A, basis)
        println("k=$k: ", real(L), " ; acc_rate=", round(acc*100,sigdigits=2), "%")

        #ss = calculate_spin_spin_correlation(MPOMC.params,A,sz,1)
        #mz=tensor_calculate_z_magnetization(MPOMC.params,A)
        #println(mz)
        #ss = zeros(ComplexF64,N-1)
        #for j in 2:N
        #    ss[j-1] = calculate_spin_spin_correlation(MPOMC.params,A,sz,j-1)
        #end
        #display(ss)

        push!(list_of_L,L)
    end
end
