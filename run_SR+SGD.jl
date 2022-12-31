include("MPOMC.jl")
using .MPOMC
using NPZ
using Plots
using LinearAlgebra
using Distributions
using Revise

#Define constants:
const J=0.25 #interaction strength
const h=0.5 #transverse field strength
const γ=0.5 #spin decay rate
const N=4
const dim = 2^N
χ=4 #bond dimension

MPOMC.set_parameters(N,χ,J,h,γ)

#Make single-body Lindbladian:
const l1 = make_one_body_Lindbladian(h*MPOMC.sx,γ*MPOMC.sm)
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
δB = 0.02

Levy_dist = truncated(Levy(1.0, 0.001),0,10)
N_MC=2
Q=1
QB=1
F=0.99
ϵ=0.3

@time begin
    for k in 1:300
        L=0;LB=0
        for i in 1:10

            new_A=zeros(ComplexF64, χ,χ,4)
            #∇,L=SR_calculate_gradient(MPOMC.params,A,l1,ϵ,basis)
            #∇,L=SR_calculate_MC_gradient_full(MPOMC.params,A,l1,100,2,ϵ) 
            #∇,L=calculate_MC_gradient_full(MPOMC.params,A,l1,10,2) 
            ∇,L=MT_SGD_MC_grad(MPOMC.params,A,l1,20,2) 
            #∇,L=calculate_MC_gradient("SR",MPOMC.params,A,l1,N_MC+5*k,2,ϵ) 
            ∇./=maximum(abs.(∇))
            #global δ = adaptive_step_size(δ,L,old_L)
            new_A = A - 1.0*δ*F^k*∇#.*(1+0.5*rand())
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
            #∇B,LB=calculate_MC_gradient("SGD",MPOMC.params,B,l1,N_MC+5*k,2,ϵ) 
            #∇B,LB=calculate_MC_gradient_full(MPOMC.params,B,l1,100,2) 
            #∇B,LB=SR_calculate_MC_gradient_full(MPOMC.params,B,l1,1,50,ϵ) 
            ∇B,LB=multi_threaded_SR_calculate_MC_gradient_full(MPOMC.params,B,l1,20,2,ϵ) 
            ∇B./=maximum(abs.(∇B))
            #global δB = adaptive_step_size(δB,LB,old_LB)*rand(Levy_dist)
            #global δB = δB*min(1,sqrt(LB))*F^k
            #new_B = B - δB*∇B#.*(1+0.5*rand())#*(1+rand())#.*(1+0.1*rand())
            #global δB=adaptive_step_size(δB,LB,old_LB)#+0.01*F^k
            #new_B = B - 1.0*δ*F^k*sign.(∇B)*(1+rand())
            #global δB = adaptive_step_size(δB,LB,old_LB)
            new_B = B - 1.0*δB*F^k*∇B#.*(1+0.5*rand())
            global B = new_B
            global B./=normalize_MPO(MPOMC.params, B)

            #global QB=sqrt(calculate_mean_local_Lindbladian(J,B))
        end

        L = calculate_mean_local_Lindbladian(MPOMC.params,l1,A,basis)
        LB= calculate_mean_local_Lindbladian(MPOMC.params,l1,B,basis)
        println("k=$k: ", real(L), " ; ", real(LB))
        global old_L = L
        global old_LB = LB

        push!(list_of_L,L)
        push!(list_of_LB,LB)
    end
end
#error()

#display(list_of_L)
#open("MPOMC_L_SR.txt", "w") do file
#    write(file, list_of_L)
#end
#npzwrite("data/MPOMC_L_SR_χ=$χ.npy", list_of_L)
#npzwrite("data/MPOMC_L_SGD_χ=$χ.npy", list_of_LB)



ρ = make_density_matrix(MPOMC.params, A, basis)
display(ρ)

npzwrite("data/MPOMC_rho_real_χ=$χ.npy", real.(ρ))
npzwrite("data/MPOMC_rho_imag_χ=$χ.npy", imag.(ρ))

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

npzwrite("data/rho_real.npy", real.(ρ))
npzwrite("data/rho_imag.npy", imag.(ρ))
