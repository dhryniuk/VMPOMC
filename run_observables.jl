using Distributed

#addprocs(4)
#println(nprocs())
#println(nworkers())

@everywhere include("MPOMC.jl")
@everywhere using .MPOMC
using NPZ
using Plots
using LinearAlgebra
#using TensorOperations
#using Distributions
#using Revise
import Random


#Vincentini parameters: γ=1.0, J=0.5, h to be varied.

#Define constants:
const Jx= 0.0 #interaction strength
const Jy= 0.0 #interaction strength
const J = 0.5 #interaction strength
const hx= 0.3 #transverse field strength
const hz= 0.0 #transverse field strength
const γ = 1.0 #spin decay rate
const α=0
const N=4
const dim = 2^N
χ=3 #bond dimension
const burn_in = 0

MPOMC.set_parameters(N,χ,Jx,Jy,J,hx,hz,γ,α, burn_in)

#Make single-body Lindbladian:
const l1 = conj( make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm) )
#const l1 = ( make_one_body_Lindbladian(-hx*sx-hz*sz,sqrt(γ)*sm) )
#display(l1)

const basis=generate_bit_basis_reversed(N)


Random.seed!(1)
A_init=rand(ComplexF64, χ,χ,2,2)
A=deepcopy(A_init)
A=reshape(A,χ,χ,4)

list_of_L = Array{Float64}(undef, 0)
list_of_Mx= Array{ComplexF64}(undef, 0)
list_of_My= Array{ComplexF64}(undef, 0)
list_of_Mz= Array{ComplexF64}(undef, 0)

list_of_purities= Array{ComplexF64}(undef, 0)

list_of_density_matrices= Array{Matrix{ComplexF64}}(undef, 0)

function set_beta(it, β_inf, decay_rate)
    return β_inf +(1-β_inf)/(it*decay_rate+1)
end

δ::Float16 = 0.03

N_MC=2
Q=1
F::Float16=0.98
ϵ::Float16=0.1
β::Float64=0.6

display(A_init)

#@profview begin
@time begin
    for k in 1:200
        L=0;LB=0
        acc::Float64=0
        for i in 1:10

            new_A=zeros(ComplexF64, χ,χ,4)

            ∇,L=gradient("exact",A,l1,MPOMC.params,basis=basis)

            #∇,L=Exact_MPO_gradient(A,l1,basis,MPOMC.params)
            #∇,L,acc=SGD_MPO_gradient(A,l1,10*4*χ^2+k,MPOMC.params)
            #∇,L,acc=reweighted_SGD_MPO_gradient(set_beta(k,0.4,0.02),A,l1,10*4*χ^2+k,MPOMC.params)#0+50*k)
            #∇,L,acc=SR_MPO_gradient(A,l1,10*4*χ^2+k,ϵ, MPOMC.params)
            #∇,L,acc=reweighted_SR_MPO_gradient(set_beta(k,0.4,0.02),A,l1,10*4*χ^2+k,ϵ, MPOMC.params)#0+50*k)
            #∇,L=distributed_SR_calculate_MC_gradient_full(MPOMC.params,A,l1,300,0, ϵ)
            #∇,L=SGD_MC_grad_distributed(MPOMC.params,A,l1,25,0)
            #∇,L=MT_SGD_MC_grad(MPOMC.params,A,l1,5,2)
            #∇,L=multi_threaded_SR_calculate_MC_gradient_full(MPOMC.params,A,l1,1,0,ϵ) 
            #display(∇)
            #error()
            ∇./=maximum(abs.(∇))
            #display(∇)
            #error()
            new_A = A - δ*F^(k)*∇#.*(1+0.5*rand())

            global A = new_A
            global A = normalize_MPO(MPOMC.params, A)

        end
        mx = calculate_x_magnetization(MPOMC.params,A)
        my = calculate_y_magnetization(MPOMC.params,A)
        mz = calculate_z_magnetization(MPOMC.params,A)  ### THERE IS AN INCONSISTENCY WITH MC RESULTS
          ### CHECK X MAGNETIZATION FOR H NON ZERO ONLY

        #L = calculate_mean_local_Lindbladian(MPOMC.params,l1,A,basis)
        #println("k=$k: ", real(L), " ; ", mz, " ; ", mx)
        println("k=$k: ", real(L)/N, " ; acc_rate=", round(acc*100,sigdigits=2), "%", " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))

        push!(list_of_L,L)
        push!(list_of_Mx,mx)
        push!(list_of_My,my)
        push!(list_of_Mz,mz)

        #push!(list_of_purities, tensor_purity(MPOMC.params, A))

        #push!(list_of_density_matrices, make_density_matrix(MPOMC.params, A, basis))
    end
end

"""
error()
#npzwrite("data/observables/MPOMC_list_rho_real_χ=$χ.npy", list_of_density_matrices[:])
#npzwrite("data/observables/MPOMC_list_rho_imag_χ=$χ.npy", imag.(list_of_density_matrices))

#display(list_of_L)
#open("MPOMC_L_SR.txt", "w") do file
#    write(file, list_of_L)
#end
#npzwrite("data/MPOMC_L_SR_χ=$χ.npy", list_of_L)
#npzwrite("data/MPOMC_L_SGD_χ=$χ.npy", list_of_LB)



ρ = make_density_matrix(MPOMC.params, A, basis)
display(ρ)

npzwrite("data/observables/MPOMC_rho_real_χ=$χ.npy", real.(ρ))
npzwrite("data/observables/MPOMC_rho_imag_χ=$χ.npy", imag.(ρ))

yticks_array = [10.0^(-i) for i in -1:4]
p=plot(list_of_L, xaxis=:log, yaxis=:log, yticks=(yticks_array))
display(p)

#p=plot(real(list_of_m), xaxis=:log)
#plot!(imag(list_of_m))
#display(p)
"""

#L=own_version_DQIM(MPOMC.params,basis)
L=sparse_DQIM(MPOMC.params, "periodic")
#vals, vecs = eigen(L)
vals, vecs = eigen_sparse(L)
#display(vals)
#display(vecs[:,2^(2N)])
#ρ=reshape(vecs[:,2^(2N)],2^N,2^N)
ρ=reshape(vecs,2^N,2^N)
ρ=round.(ρ,digits = 12)
ρ./=tr(ρ)
#display(ρ)

#npzwrite("data/observables/rho_real.npy", real.(ρ))
#npzwrite("data/observables/rho_imag.npy", imag.(ρ))

#vec_basis=construct_vec_density_matrix_basis(MPOMC.params.N)

#Mx=real( own_x_magnetization(ρ,MPOMC.params,vec_basis) )
Mx=real( magnetization(sx,ρ,MPOMC.params) )
println("True x-magnetization is: ", Mx)

My=real( magnetization(sy,ρ,MPOMC.params) )
println("True y-magnetization is: ", My)

#Mz=real( own_z_magnetization(ρ,MPOMC.params,vec_basis) )
Mz=real( magnetization(sz,ρ,MPOMC.params) )
println("True z-magnetization is: ", Mz)

error()

p=plot(real(list_of_Mx), xaxis=:log)#, ylims=(-0.3,0.3))
#plot!(imag(list_of_Mz))
hline!([Mx,Mx])

plot!(real(list_of_My))
#plot!(imag(list_of_Mx))
hline!([My,My])

plot!(real(list_of_Mz))
#plot!(imag(list_of_Mx))
hline!([Mz,Mz])

display(p)

insert!(list_of_Mx, 1, Mx)
insert!(list_of_My, 1, My)
insert!(list_of_Mz, 1, Mz)

npzwrite("data/observables/mag_x_N16_2.npy", list_of_Mx)
npzwrite("data/observables/mag_y_N16_2.npy", list_of_My)
npzwrite("data/observables/mag_z_N16_2.npy", list_of_Mz)

ED_purity = calculate_purity(ρ)
insert!(list_of_purities, 1, ED_purity)
ED_entropy = -log2(ED_purity)
list_of_entropies = -log2.(list_of_purities)
insert!(list_of_entropies, 1, ED_entropy)

npzwrite("data/observables/purity.npy", list_of_purities)
npzwrite("data/observables/entropy.npy", list_of_entropies)