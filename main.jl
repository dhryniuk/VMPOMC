module LVMC

using LinearAlgebra
using Distributions
using Plots
using NPZ
using DelimitedFiles

#Exact diagonalisation routines:
include("ED/ED_Ising.jl")
include("ED/ED_Lindblad.jl")

#MPO:
include("MPO/MPO.jl")

#Samplers:
include("Samplers/Metropolis.jl")

#Optimisers:
include("Optimisers/SGD.jl")
include("Optimisers/SR.jl")

#Other utilities:
include("utils/utils.jl")


#Define constants:
const J=0.5 #interaction strength
const h=0.5 #transverse field strength
const γ=0.5 #spin decay rate
const N=3
const dim = 2^N
χ=4 #bond dimension

#Make single-body Lindbladian:
const l1 = make_Liouvillian(h*sx,γ*sm)
#display(l1)

const basis=generate_bit_basis_reversed(N)

#=
#Generate complete basis (not necessary when sampling via MCMC):
#const basis=generate_bit_basis(N)
const basis=generate_bit_basis_reversed(N)
#display(basis)

#display(construct_vec_density_matrix_basis(N))


#error()

l_canonical = DQIM(N,J,h,γ)
#display(l_canonical)
const rev_basis=construct_vec_density_matrix_basis(N)
display(rev_basis)
l_own = own_version_DQIM(N,J,h,γ,rev_basis)
#display(l_own)

cevals, cevecs = eigen(l_canonical)
oevals, oevecs = eigen(l_own)
display(cevals)
display(oevals)


NESS_canonical = reshape(cevecs[:,dim^2],dim,dim)
NESS_canonical./=tr(NESS_canonical)
NESS_own = reshape(oevecs[:,dim^2],dim,dim)
NESS_own = normalize_own_density_matrix(NESS_own,rev_basis)
#NESS_own./=tr(NESS_canonical)

#display(cevecs)
display(NESS_canonical)
#display(reshape(NESS_canonical,dim,dim))
display(NESS_own)
display(calculate_magnetization(NESS_canonical,N)/N)

display(own_z_magnetization(NESS_own,N,rev_basis)/N)

=#

#=
ρ_real=npzread("MPOMC_rho_real_χ=$χ.npy")
ρ_imag=npzread("MPOMC_rho_imag_χ=$χ.npy")
ρ=ρ_real+1im*ρ_imag

display(ρ)
display(calculate_magnetization(ρ,N)/N)
=#


#display(diag(l_canonical))
#display(diag(l_own))

#println(diag(l_canonical)==diag(l_own))


#error()
A_init=rand(ComplexF64, χ,χ,2,2)
A=copy(A_init)
A=reshape(A,χ,χ,4)
B=deepcopy(A)

list_of_L = Array{Float64}(undef, 0)
list_of_LB= Array{Float64}(undef, 0)

old_L=1
old_LB=1

δ = 0.03
δB = 0.03

Levy_dist = truncated(Levy(1.0, 0.001),0,10)
A=B
N_MC=100
Q=1
QB=1
F=0.99
@time begin
    for k in 1:500

        new_A=zeros(ComplexF64, χ,χ,4)
        ∇,L=SR_calculate_gradient(J,A)
        #∇,L=calculate_MC_gradient_full(J,A,N_MC+5*k)
        #∇,L=SRMC_gradient_full(J,A,N_MC+10*k,2,k)
        ∇./=maximum(abs.(∇))
        new_A = A - 2.0*δ*F^(k)*∇#.*(1+0.5*rand())
        #new_A = A - 1.0*δ*F^(k)*sign.(∇)*(1+rand())
        #global δ = adaptive_step_size(δ,L,old_L)*rand(Levy_dist)
        #global δ = δ*min(1,sqrt(L))*F^k
        #new_A = A - δ*∇#.*(1+0.5*rand())#.*(1+0.1*rand())
        #new_A = A - δ*∇#.*(1+0.05*rand())

        global A = new_A
        global A./=normalize_MPO(A)
        #Lex=calculate_mean_local_Lindbladian(J,A)
        #global Q=sqrt(calculate_mean_local_Lindbladian(J,A))
        #global Q=sqrt(L)
        #println("k=$k: ", real(L), " ; ", real(Lex))
        #println("k=$k: ", real(L))

        new_B=zeros(ComplexF64, χ,χ,4)
        #∇B,LB=calculate_MC_gradient_full(J,B,N_MC+10*k,2)
        ∇B,LB=calculate_gradient(J,B)
        ∇B./=maximum(abs.(∇B))
        #global δB = adaptive_step_size(δB,LB,old_LB)*rand(Levy_dist)
        #global δB = δB*min(1,sqrt(LB))*F^k
        #new_B = B - δB*∇B#.*(1+0.5*rand())#*(1+rand())#.*(1+0.1*rand())
        #global δB=adaptive_step_size(δB,LB,old_LB)#+0.01*F^k
        #new_B = B - 1.0*δ*F^k*sign.(∇B)*(1+rand())
        new_B = B - 1.0*δ*F^k*∇B#.*(1+0.5*rand())
        global B = new_B
        global B./=normalize_MPO(B)

        #global QB=sqrt(calculate_mean_local_Lindbladian(J,B))

        println("k=$k: ", real(L), " ; ", real(LB))
        global old_L = L
        global old_LB = LB

        push!(list_of_L,L)
        push!(list_of_LB,LB)
    end
end
#display(list_of_L)
#open("MPOMC_L_SR.txt", "w") do file
#    write(file, list_of_L)
#end
npzwrite("MPOMC_L_SR_χ=$χ.npy", list_of_L)
npzwrite("MPOMC_L_SGD_χ=$χ.npy", list_of_LB)



ρ = make_density_matrix(A,basis)
display(ρ)

npzwrite("MPOMC_rho_real_χ=$χ.npy", real.(ρ))
npzwrite("MPOMC_rho_imag_χ=$χ.npy", imag.(ρ))

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

npzwrite("rho_real.npy", real.(ρ))
npzwrite("rho_imag.npy", imag.(ρ))

end

























#OLD:
error()
B=deepcopy(A)

list_of_L = Array{Float64}(undef, 0)
list_of_LB= Array{Float64}(undef, 0)

old_L=1
old_LB=1

δ = 0.15
δB = 0.05

Levy_dist = truncated(Levy(1.0, 0.001),0,10)
A=B
N_MC=500
Q=1
QB=1
F=0.99
@time begin
    for k in 1:500

        new_A=zeros(ComplexF64, χ,χ,4)
        #∇,L=SR_calculate_gradient(J,A)
        #∇,L=calculate_MC_gradient_full(J,A,N_MC+5*k)
        ∇,L=SRMC_gradient_full(J,A,N_MC+1*k,1,k)
        ∇./=maximum(abs.(∇))
        new_A = A - 1.0*δ*F^(k)*∇#.*(1+0.5*rand())
        #new_A = A - 1.0*δ*F^(k)*sign.(∇)*(1+rand())
        #global δ = adaptive_step_size(δ,L,old_L)*rand(Levy_dist)
        #global δ = δ*min(1,sqrt(L))*F^k
        #new_A = A - δ*∇#.*(1+0.5*rand())#.*(1+0.1*rand())
        #new_A = A - δ*∇#.*(1+0.05*rand())

        global A = new_A
        global A./=normalize_MPO(A)
        #Lex=calculate_mean_local_Lindbladian(J,A)
        #global Q=sqrt(calculate_mean_local_Lindbladian(J,A))
        #global Q=sqrt(L)
        #println("k=$k: ", real(L), " ; ", real(Lex))
        #println("k=$k: ", real(L))

        new_B=zeros(ComplexF64, χ,χ,4)
        ∇B,LB=calculate_MC_gradient_full(J,B,N_MC+1*k,1)
        #∇B,LB=calculate_gradient(J,B)
        ∇B./=maximum(abs.(∇B))
        #global δB = adaptive_step_size(δB,LB,old_LB)*rand(Levy_dist)
        #global δB = δB*min(1,sqrt(LB))*F^k
        #new_B = B - δB*∇B#.*(1+0.5*rand())#*(1+rand())#.*(1+0.1*rand())
        #global δB=adaptive_step_size(δB,LB,old_LB)#+0.01*F^k
        #new_B = B - 1.0*δ*F^k*sign.(∇B)*(1+rand())
        new_B = B - 1.0*δ*F^k*∇B#.*(1+0.5*rand())
        global B = new_B
        global B./=normalize_MPO(B)

        #global QB=sqrt(calculate_mean_local_Lindbladian(J,B))

        println("k=$k: ", real(L), " ; ", real(LB))
        global old_L = L
        global old_LB = LB

        push!(list_of_L,L)
        push!(list_of_LB,LB)
    end
end
display(A)
#display(list_of_L)
#open("MPOMC_L_SR.txt", "w") do file
#    write(file, list_of_L)
#end
npzwrite("MPOMC_L_SR_χ=$χ.npy", list_of_L)
npzwrite("MPOMC_L_SGD_χ=$χ.npy", list_of_LB)


ρ = make_density_matrix(B,basis)

npzwrite("MPOMC_rho_real_χ=$χ.npy", real.(ρ))
npzwrite("MPOMC_rho_imag_χ=$χ.npy", imag.(ρ))

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

npzwrite("rho_real.npy", real.(ρ))
npzwrite("rho_imag.npy", imag.(ρ))






#l=l1⊗id⊗id+id⊗id⊗l1
#display(l)

L=DQIM(N,-J,-h,γ)
#display(L)

vals, vecs = eigen(L)
#display(vals)
#display(vecs[:,2^(2N)])
ρ=reshape(vecs[:,2^(2N)],2^N,2^N)
ρ=round.(ρ,digits = 12)
ρ./=tr(ρ)
display(ρ)
#println(l==L)

display(calculate_magnetization(ρ,N))

MPOρ_real=npzread("MPOMC_rho_real_χ=4_step100.npy", real.(ρ))
MPOρ_imag=npzread("MPOMC_rho_imag_χ=4_step100.npy", imag.(ρ))
MPOρ=MPOρ_real+1im*MPOρ_imag

display(calculate_magnetization(MPOρ,N))

MPOρ_real=npzread("MPOMC_rho_real_χ=4_step500.npy", real.(ρ))
MPOρ_imag=npzread("MPOMC_rho_imag_χ=4_step500.npy", imag.(ρ))
MPOρ=MPOρ_real+1im*MPOρ_imag

display(calculate_magnetization(MPOρ,N))

error()



#OLD:

A_init=rand(ComplexF64, χ,χ,2,2)
A=copy(A_init)
A=reshape(A,χ,χ,4)

B=deepcopy(A)

list_of_L = Array{Float64}(undef, 0)
list_of_LB= Array{Float64}(undef, 0)

old_L=1
old_LB=1

δ = 0.15
δB = 0.05

Levy_dist = truncated(Levy(1.0, 0.001),0,10)
A=B
N_MC=100
Q=1
QB=1
F=0.98
@time begin
    for k in 1:100

        new_A=zeros(ComplexF64, χ,χ,4)
        #∇,L=calculate_gradient(J,A)
        #∇,L=SR_calculate_gradient(J,A)
        ∇,L=calculate_MC_gradient_full(J,A,N_MC+5*k,2)
        #∇,L=SRMC_gradient_full(J,A,N_MC+1*k,1,k)
        ∇./=maximum(abs.(∇))
        new_A = A - 1.0*δ*F^(k)*∇#.*(1+0.5*rand())
        #new_A = A - 1.0*δ*F^(k)*sign.(∇)*(1+rand())
        #global δ = adaptive_step_size(δ,L,old_L)*rand(Levy_dist)
        #global δ = δ*min(1,sqrt(L))*F^k
        #new_A = A - δ*∇#.*(1+0.5*rand())#.*(1+0.1*rand())
        #new_A = A - δ*∇#.*(1+0.05*rand())

        global A = new_A
        global A./=normalize_MPO(A)
        println("k=$k: ", real(L))
        global old_L = L

        push!(list_of_L,L)
    end
end
#display(list_of_L)
#open("MPOMC_L_SR.txt", "w") do file
#    write(file, list_of_L)
#end
npzwrite("MPOMC_L_SGD_χ=$χ.npy", list_of_L)
