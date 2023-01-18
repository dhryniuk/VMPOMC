include("MPOMC.jl")
using .MPOMC
using NPZ
using Plots
using LinearAlgebra


#Define constants:
const J=0.5 #interaction strength
const h=0.5 #transverse field strength
const γ=0.5 #spin decay rate
const N=4
const dim = 2^N
χ=4 #bond dimension

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
δB = 0.03

#Levy_dist = truncated(Levy(1.0, 0.001),0,10)
N_MC=200
Q=1
QB=1
F=0.99
ϵ=0.1
@time begin
    for k in 1:500

        new_A=zeros(ComplexF64, χ,χ,4)
        #∇,L=SR_calculate_gradient(MPOMC.params,A,l1,ϵ,basis)
        ∇,L=SR_calculate_MC_gradient_full(MPOMC.params,A,l1,N_MC+10*k,2,k)
        ∇./=maximum(abs.(∇))
        new_A = A - δ*F^(k)*∇#.*(1+0.5*rand())
        #new_A = A - 1.0*δ*F^(k)*sign.(∇)*(1+rand())
        #global δ = adaptive_step_size(δ,L,old_L)*rand(Levy_dist)
        #global δ = δ*min(1,sqrt(L))*F^k
        #new_A = A - δ*∇#.*(1+0.5*rand())#.*(1+0.1*rand())
        #new_A = A - δ*∇#.*(1+0.05*rand())

        global A = new_A
        global A./=normalize_MPO(MPOMC.params, A)

        println("k=$k: ", real(L))
        global old_L = L

        push!(list_of_L,L)
    end
end


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
