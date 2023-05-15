include("MPOMC.jl")
using .MPOMC
using NPZ
using Plots
using LinearAlgebra
using BenchmarkTools
#using Cthulhu
import Random

function make_two_body_Lindblad_Hamiltonian(A, B)
    L_H = -1im*( (A⊗id)⊗(B⊗id) - (id⊗transpose(A))⊗(id⊗transpose(B)) )
    return L_H
end

#Define constants:
const Jx= 0.6 #interaction strength
const Jy= 0.2 #interaction strength
const J = 0.4 #interaction strength
const hx= 0.3 #transverse field strength
const hz= 0.0 #transverse field strength
const γ = 1.0 #spin decay rate
const α=0
const N=4
const dim = 2^N
χ=4 #bond dimension
const burn_in = 0


MPOMC.set_parameters(N,χ,Jx,Jy,J,hx,hz,γ,α, burn_in)

#Make single-body Lindbladian:
const l1 = conj( make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm) )
#display(l1)

⊗(x,y) = kron(x,y)

#display(l1⊗id⊗id+id⊗id⊗l1)
#error()

const basis=generate_bit_basis_reversed(N)

const l2 = Jx*make_two_body_Lindblad_Hamiltonian(sx,sx) + Jy*make_two_body_Lindblad_Hamiltonian(sy,sy)
#const l2 = J*alt_make_two_body_Lindblad_Hamiltonian(sx,sx)


Random.seed!(1)
A_init=rand(ComplexF64, χ,χ,2,2)
A=deepcopy(A_init)
A=reshape(A,χ,χ,4)

list_of_L = Array{Float64}(undef, 0)
list_of_LB= Array{Float64}(undef, 0)

old_L=1
old_LB=1

δ = 0.03

#Levy_dist = truncated(Levy(1.0, 0.001),0,10)
N_MC=200
Q=1
QB=1
F=0.97
ϵ=0.1

@time begin
#@profview begin
    for k in 1:200
        L=0
        acc::Float64=0
        for l in 1:10
            new_A=zeros(ComplexF64, χ,χ,4)

            ∇,L,acc=gradient("SR",A,l1,MPOMC.params,N_MC=10*4*χ^2+k,ϵ=ϵ,parallel=true,l2=l2)

            #∇,L=Exact_MPO_gradient_two_body(A,l1,l2,basis,MPOMC.params)
            #∇,L,acc=SGD_MPO_gradient_two_body(A,l1,l2,50*4*χ^2+k,MPOMC.params)
            #∇,L,acc=SR_MPO_gradient_two_body(A,l1,l2,(10+N)*4*χ^2+k,ϵ,MPOMC.params)
            ∇./=maximum(abs.(∇))
            new_A = A - δ*F^(k)*∇#.*(1+0.5*rand())

            global A = new_A
            global A = normalize_MPO(MPOMC.params, A)
        end

        println("k=$k: ", real(L)/N, " ; acc_rate=", round(acc*100,sigdigits=2), "%")#, " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))
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

#println(tensor_purity(MPOMC.params,A))
#println(calculate_purity(MPOMC.params,A))

#@code_warntype SR_MPO_gradient(MPOMC.params,A,l1,100,ϵ)

L = XYZ_Lindbald(MPOMC.params,"periodic")
#L=sparse_DQIM(MPOMC.params, "periodic")
vals, vecs = eigen_sparse(L)
ρ=reshape(vecs,2^N,2^N)
ρ=round.(ρ,digits = 12)
ρ./=tr(ρ)

Mx=real( magnetization(sx,ρ,MPOMC.params) )
println("True x-magnetization is: ", Mx)

My=real( magnetization(sy,ρ,MPOMC.params) )
println("True y-magnetization is: ", My)

Mz=real( magnetization(sz,ρ,MPOMC.params) )
println("True z-magnetization is: ", Mz)