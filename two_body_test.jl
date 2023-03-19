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
const h=0.5 #transverse field strength
const γ=0.5 #spin decay rate
const α=0
const N=3
const dim = 2^N
χ=3 #bond dimension
const burn_in = 0

MPOMC.set_parameters(N,χ,J,h,γ,α, burn_in)

#Make single-body Lindbladian:
#const l1 = make_one_body_Lindbladian(MPOMC.params,sx,sm)
const l1 = conj( make_one_body_Lindbladian(MPOMC.params,sx,sm) )
#display(l1)

⊗(x,y) = kron(x,y)

#display(l1⊗id⊗id+id⊗id⊗l1)
#error()

const basis=generate_bit_basis_reversed(N)

Jx = 1
Jy = 1

const l2 = J*make_two_body_Lindblad_Hamiltonian(sx,sx)# + Jy*make_two_body_Lindblad_Hamiltonian(sy,sy)
#const l2 = J*alt_make_two_body_Lindblad_Hamiltonian(sx,sx)


A_init=rand(ComplexF64, χ,χ,2,2)
A=copy(A_init)
A=reshape(A,χ,χ,4)
B=deepcopy(A)

list_of_L = Array{Float64}(undef, 0)
list_of_LB= Array{Float64}(undef, 0)

old_L=1
old_LB=1

δ = 0.01

#Levy_dist = truncated(Levy(1.0, 0.001),0,10)
N_MC=200
Q=1
QB=1
F=0.99
ϵ=0.1

@time begin
#@profview begin
    for k in 1:1000
        L=0
        for l in 1:10
            new_A=zeros(ComplexF64, χ,χ,4)
            ∇,L=Two_body_Exact_MPO_gradient(MPOMC.params,A,l1,l2,basis)
            #∇,L=SGD_MPO_gradient(MPOMC.params,A,l1,500)#0+50*k)
            #∇,L=SR_MPO_gradient(MPOMC.params,A,l1,500,ϵ)#0+50*k)
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
mx = calculate_x_magnetization(MPOMC.params,A)
my = calculate_y_magnetization(MPOMC.params,A)
mz = calculate_z_magnetization(MPOMC.params,A) 

println("Mx=",mx)
println("My=",my)
println("Mz=",mz)

println(tensor_purity(MPOMC.params,A))
println(calculate_purity(MPOMC.params,A))

#@code_warntype SR_MPO_gradient(MPOMC.params,A,l1,100,ϵ)