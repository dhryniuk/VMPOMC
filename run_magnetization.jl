#include("MPOMC.jl")
using .MPOMC
using NPZ
using Plots
using LinearAlgebra
using Distributions
using Revise

#Define constants:
const J=0.5 #interaction strength
const h=1.0 #transverse field strength
const γ=1.0 #spin decay rate
const N=2
const dim = 2^N
χ=3 #bond dimension

MPOMC.set_parameters(N,χ,J,h,γ)

#Make single-body Lindbladian:
const l1 = make_one_body_Lindbladian(h*MPOMC.sx,γ*MPOMC.sm)
#display(l1)

const basis=generate_bit_basis_reversed(N)


A_init=rand(ComplexF64, χ,χ,2,2)
A=copy(A_init)
A=reshape(A,χ,χ,4)

list_of_L = Array{Float64}(undef, 0)
list_of_Mz= Array{ComplexF64}(undef, 0)
list_of_Mx= Array{ComplexF64}(undef, 0)

old_L=1

δ = 0.03

Levy_dist = truncated(Levy(1.0, 0.001),0,10)
N_MC=2
Q=1
QB=1
F=0.99
ϵ=0.3

@time begin
    for k in 1:200
        L=0;LB=0
        for i in 1:20

            new_A=zeros(ComplexF64, χ,χ,4)
            ∇,L=multi_threaded_SR_calculate_MC_gradient_full(MPOMC.params,A,l1,40,2,ϵ) 
            #∇,L=SR_calculate_gradient(MPOMC.params,A,l1,ϵ,basis)
            ∇./=maximum(abs.(∇))
            new_A = A - 1.0*δ*F^k*∇

            global A = new_A
            global A./=normalize_MPO(MPOMC.params, A)

        end
        mz = calculate_z_magnetization(MPOMC.params,A)
        mx = calculate_x_magnetization(MPOMC.params,A)

        #L = calculate_mean_local_Lindbladian(MPOMC.params,l1,A,basis)
        #println("k=$k: ", real(L), " ; ", mz, " ; ", mx)
        println("k=$k: ", real(L), " ; ", round(mz,sigdigits=4), " ; ", round(mx,sigdigits=4))
        global old_L = L

        push!(list_of_L,L)
        push!(list_of_Mz,mz)
        push!(list_of_Mx,mx)
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

#npzwrite("data/MPOMC_rho_real_χ=$χ.npy", real.(ρ))
#npzwrite("data/MPOMC_rho_imag_χ=$χ.npy", imag.(ρ))

yticks_array = [10.0^(-i) for i in -1:4]
p=plot(list_of_L, xaxis=:log, yaxis=:log, yticks=(yticks_array))
display(p)

#p=plot(real(list_of_m), xaxis=:log)
#plot!(imag(list_of_m))
#display(p)


#L=own_version_DQIM(MPOMC.params,basis)
L=DQIM(MPOMC.params)
vals, vecs = eigen(L)
#vals, vecs = eigen_sparse(L)
display(vals)
display(vecs[:,2^(2N)])
ρ=reshape(vecs[:,2^(2N)],2^N,2^N)
ρ=round.(ρ,digits = 12)
ρ./=tr(ρ)
display(ρ)

vec_basis=construct_vec_density_matrix_basis(MPOMC.params.N)

Mz=real( own_z_magnetization(ρ,MPOMC.params,vec_basis) )
println("True z-magnetization is: ", Mz)

Mx=real( own_x_magnetization(ρ,MPOMC.params,vec_basis) )
println("True x-magnetization is: ", Mx)

p=plot(real(list_of_Mz), xaxis=:log)
#plot!(imag(list_of_Mz))
hline!([Mz,Mz])

plot!(real(list_of_Mx))
#plot!(imag(list_of_Mx))
hline!([Mx,Mx])

display(p)