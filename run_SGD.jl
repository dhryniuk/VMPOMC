include("MPOMC.jl")
using .MPOMC
using NPZ
using Plots
using LinearAlgebra


#Define constants:
const J=0.0 #interaction strength
const h=1.1 #transverse field strength
const γ=0.3 #spin decay rate
const N=4
const dim = 2^N
χ=2 #bond dimension

MPOMC.set_parameters(N,χ,J,h,γ)

#Make single-body Lindbladian:
const l1 = make_one_body_Lindbladian(h*sx,γ*sm)
#display(l1)

const basis=generate_bit_basis_reversed(N)

mutable struct parameters
    N::Int
    dim::Int
    χ::Int
    J::Float64
    h::Float64
    γ::Float64
end

#params=parameters(0,0,0,0,0,0)

function set_parameters(N,χ,J,h,γ)
	params.N = N
    params.dim = 2^N;
    params.χ = χ;
    params.J = J;
    params.h = h;
    params.γ = γ;
end

#set_parameters(N,χ,J,h,γ)




A_init=rand(ComplexF64, χ,χ,2,2)
A=copy(A_init)
A=reshape(A,χ,χ,4)
B=deepcopy(A)

list_of_L = Array{Float64}(undef, 0)
list_of_LB= Array{Float64}(undef, 0)

old_L=1
old_LB=1

δ = 0.05
δB = 0.05

#Levy_dist = truncated(Levy(1.0, 0.001),0,10)
N_MC=100
Q=1
QB=1
F=0.99
ϵ=0.01
@time begin
    for k in 1:1000

        new_A=zeros(ComplexF64, χ,χ,4)
        #∇,L=SR_calculate_gradient(MPOMC.params,A,l1,ϵ,basis)
        ∇,L=SR_LdagL_gradient(MPOMC.params,A,l1,ϵ,basis)
        ∇./=maximum(abs.(∇))
        new_A = A - 1.0*δ*F^(k)*∇#.*(1+0.5*rand())
        #new_A = A - 2.0*δ*F^(k)*sign.(∇)*(1+rand())
        #global δ = adaptive_step_size(δ,L,old_L)*rand(Levy_dist)
        #global δ = δ*min(1,sqrt(L))*F^k
        #new_A = A - δ*∇#.*(1+0.5*rand())#.*(1+0.1*rand())
        #new_A = A - δ*∇#.*(1+0.05*rand())

        global A = new_A
        global A./=normalize_MPO(MPOMC.params, A)
        #Lex=calculate_mean_local_Lindbladian(J,A)
        #global Q=sqrt(calculate_mean_local_Lindbladian(J,A))
        #global Q=sqrt(L)

        new_B=zeros(ComplexF64, χ,χ,4)
        #∇B,LB=calculate_gradient(MPOMC.params,B,l1,basis)
        ∇B,LB=LdagL_gradient(MPOMC.params,B,l1,basis)
        ∇B./=maximum(abs.(∇B))
        #global δB = adaptive_step_size(δB,LB,old_LB)*rand(Levy_dist)
        #global δB = δB*min(1,sqrt(LB))*F^k
        #new_B = B - δB*∇B#.*(1+0.5*rand())#*(1+rand())#.*(1+0.1*rand())
        #global δB=adaptive_step_size(δB,LB,old_LB)#+0.01*F^k
        #new_B = B - 1.0*δ*F^k*sign.(∇B)*(1+rand())
        new_B = B - 0.5*δ*F^k*∇B#.*(1+0.5*rand())
        global B = new_B
        global B./=normalize_MPO(MPOMC.params, B)

        #global QB=sqrt(calculate_mean_local_Lindbladian(J,B))

        println("k=$k: ", real(L), " ; ", real(LB))
        global old_L = L
        global old_LB = LB

        #display(∇)
        #display(∇B)

        push!(list_of_L,L)
        push!(list_of_LB,LB)
    end
end
error()

ρ = make_density_matrix(MPOMC.params, A, basis)
display(ρ)

#npzwrite("data/MPOMC_rho_real_χ=$χ.npy", real.(ρ))
#npzwrite("data/MPOMC_rho_imag_χ=$χ.npy", imag.(ρ))

yticks_array = [10.0^(-i) for i in -1:4]
p=plot(list_of_L, xaxis=:log, yaxis=:log, yticks=(yticks_array))
plot!(list_of_LB)
display(p)


L=DQIM(N,J,h,γ)
vals, vecs = eigen(L)
#vals, vecs = eigen_sparse(L)
display(vals)
display(vecs[:,2^(2N)])
ρ=reshape(vecs[:,2^(2N)],2^N,2^N)
ρ=round.(ρ,digits = 12)
ρ./=tr(ρ)
display(ρ)
#l = make_Liouvillian(H, Γ, id)

#npzwrite("data/rho_real.npy", real.(ρ))
#npzwrite("data/rho_imag.npy", imag.(ρ))
