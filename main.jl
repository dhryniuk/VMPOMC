module LVMC

using LinearAlgebra
using Plots

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
const J=1.0 #interaction strength
const h=1.0 #transverse field strength
const γ=1.0 #spin decay rate
const N=2
const dim = 2^N
χ=2 #bond dimension

#Make single-body Lindbladian:
const l1 = make_Liouvillian(h*sx,γ*sm)
display(l1)

#Generate complete basis (not necessary when sampling via MCMC):
const basis=generate_bit_basis(N)
display(basis)

A_init=rand(ComplexF64, χ,χ,2,2)
A=copy(A_init)
A=reshape(A,χ,χ,4)

B=deepcopy(A)

list_of_L =[]
list_of_LB=[]

old_L=1
old_LB=1

δ = 0.05
δB = 0.05

N_MC=100
Q=1
QB=1
F=0.95
@time begin
    #display(A)
    #display(B)
    #error()
    for k in 1:500

        new_A=zeros(ComplexF64, χ,χ,4)
        #∇,L=SR_calculate_gradient(J,A)
        #∇,L=calculate_MC_gradient_full(J,A,N_MC+5*k)
        ∇,L=SRMC_gradient_full(J,A,N_MC+5*k,k)
        ∇./=maximum(abs.(∇))
        #new_A = A - δ*Q*(sign.(∇).+0.01*rand())
        new_A = A - 1.0*δ*F^(k)*∇*(1+rand())#.*(1+0.1*rand())
        #global δ=adaptive_step_size(δ,L,old_L)#+0.01*F^k
        #new_A = A - δ*∇#.*(1+0.05*rand())

        global A = new_A
        global A./=normalize_MPO(A)
        #Lex=calculate_mean_local_Lindbladian(J,A)
        #global Q=sqrt(calculate_mean_local_Lindbladian(J,A))
        #global Q=sqrt(L)
        #println("k=$k: ", real(L), " ; ", real(Lex))
        #println("k=$k: ", real(L))

        new_B=zeros(ComplexF64, χ,χ,4)
        ∇B,LB=calculate_MC_gradient_full(J,B,N_MC+5*k)
        #∇B,LB=calculate_gradient(J,B)
        ∇B./=maximum(abs.(∇B))
        new_B = B - 1.0*δ*F^k*∇B*(1+rand())#.*(1+0.1*rand())
        #global δB=adaptive_step_size(δB,LB,old_LB)#+0.01*F^k
        #new_B = B - δB*∇B#.*(1+0.05*rand())
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

p=plot(list_of_L, xaxis=:log, yaxis=:log)
plot!(list_of_LB)
display(p)

end