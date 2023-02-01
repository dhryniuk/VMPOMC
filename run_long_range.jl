using Distributed
import Random
Random.seed!(1)

addprocs(8)
println(nprocs())
println(nworkers())

@everywhere include("MPOMC.jl")
@everywhere using .MPOMC
using NPZ
using Plots
using LinearAlgebra
#using TensorOperations
#using Distributions
#using Revise

#Vincentini parameters: γ=1.0, J=0.5, h to be varied.

#Define constants:
const J=0.5 #interaction strength
const h=1.0 #transverse field strength
const γ=1.0 #spin decay rate
const α=0
const N=10
const dim = 2^N
χ=4 #bond dimension

MPOMC.set_parameters(N,χ,J,h,γ,α)

#Make single-body Lindbladian:
const l1 = make_one_body_Lindbladian(h*MPOMC.sx,γ*MPOMC.sm)
#display(l1)

const basis=generate_bit_basis_reversed(N)


A_init=rand(ComplexF64, χ,χ,2,2)
A_init[:,:,1,1]=rand(Float64,χ,χ)
A_init[:,:,2,2]=rand(Float64,χ,χ)
A=copy(A_init)
A=reshape(A,χ,χ,4)

list_of_L = Array{Float64}(undef, 0)
list_of_Mx= Array{ComplexF64}(undef, 0)
list_of_My= Array{ComplexF64}(undef, 0)
list_of_Mz= Array{ComplexF64}(undef, 0)

list_of_purities= Array{ComplexF64}(undef, 0)

list_of_density_matrices= Array{Matrix{ComplexF64}}(undef, 0)

old_L=1

δ = 0.005

N_MC=2
Q=1
QB=1
F=0.99
#ϵ=100.1

@time begin
    for k in 1:200
        L=0;LB=0
        ϵ=0.2*0.99^k
        for i in 1:10

            new_A=zeros(ComplexF64, χ,χ,4)
            #∇,L=calculate_gradient(MPOMC.params,A,l1,basis)
            #∇,L=SR_calculate_gradient(MPOMC.params,A,l1,ϵ,basis)
            #∇,L=calculate_MC_gradient_full(MPOMC.params,A,l1,50,0)
            #∇,L=SR_calculate_MC_gradient_full(MPOMC.params, A, l1, 50, 0, ϵ)
            ∇,L=distributed_SR_calculate_MC_gradient_full(MPOMC.params,A,l1,200,0, ϵ)
            #∇,L=SGD_MC_grad_distributed(MPOMC.params,A,l1,25,0)
            #∇,L=MT_SGD_MC_grad(MPOMC.params,A,l1,50,0)
            #∇,L=multi_threaded_SR_calculate_MC_gradient_full(MPOMC.params,A,l1,1,0,ϵ) 
            #∇,L=SR_calculate_gradient(MPOMC.params,A,l1,ϵ,basis)
            ∇./=maximum(abs.(∇))
            new_A = A - 1.0*δ*F^k*∇

            global A = new_A
            global A./=normalize_MPO(MPOMC.params, A)
            #global A = hermetize_MPO(MPOMC.params, A)

        end
        mx = calculate_x_magnetization(MPOMC.params,A)
        my = calculate_y_magnetization(MPOMC.params,A)
        mz = calculate_z_magnetization(MPOMC.params,A)  ### THERE IS AN INCONSISTENCY WITH MC RESULTS
          ### CHECK X MAGNETIZATION FOR H NON ZERO ONLY

        #L = calculate_mean_local_Lindbladian(MPOMC.params,l1,A,basis)
        #println("k=$k: ", real(L), " ; ", mz, " ; ", mx)
        println("k=$k: ", real(L), " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))
        global old_L = L

        push!(list_of_L,L)
        push!(list_of_Mx,mx)
        push!(list_of_My,my)
        push!(list_of_Mz,mz)

        #push!(list_of_purities, tensor_purity(MPOMC.params, A))

        #push!(list_of_density_matrices, make_density_matrix(MPOMC.params, A, basis))
    end
end

#error()

#npzwrite("data/LR_Ising/MPOMC_list_rho_real_χ=$χ.npy", list_of_density_matrices[:])
#npzwrite("data/LR_Ising/MPOMC_list_rho_imag_χ=$χ.npy", imag.(list_of_density_matrices))

#display(list_of_L)
#open("MPOMC_L_SR.txt", "w") do file
#    write(file, list_of_L)
#end
#npzwrite("data/MPOMC_L_SR_χ=$χ.npy", list_of_L)
#npzwrite("data/MPOMC_L_SGD_χ=$χ.npy", list_of_LB)



ρ = make_density_matrix(MPOMC.params, A, basis)
display(ρ)

npzwrite("data/LR_Ising/MPOMC_rho_real_N=$(MPOMC.params.N)_χ=$(MPOMC.params.χ)_J=$(MPOMC.params.J)_h=$(MPOMC.params.h)_γ=$(MPOMC.params.γ)_α=$α.npy", real.(ρ))
npzwrite("data/LR_Ising/MPOMC_rho_imag_N=$(MPOMC.params.N)_χ=$(MPOMC.params.χ)_J=$(MPOMC.params.J)_h=$(MPOMC.params.h)_γ=$(MPOMC.params.γ)_α=$α.npy", imag.(ρ))

yticks_array = [10.0^(-i) for i in -1:4]
p=plot(list_of_L, xaxis=:log, yaxis=:log, yticks=(yticks_array))
display(p)

#p=plot(real(list_of_m), xaxis=:log)
#plot!(imag(list_of_m))
#display(p)


#L=DQIM(MPOMC.params)
#L=sparse_long_range_DQIM(MPOMC.params)
L=sparse_DQIM(MPOMC.params)
#vals, vecs = eigen(L)
#NESS = vecs[:,2^(2N)]
vals, NESS = eigen_sparse(L)
#display(vals)
#display(vecs[:,2^(2N)])
ρED=reshape(NESS,2^N,2^N)
#ρ=round.(ρ,digits = 12)
ρED./=tr(ρED)
display(ρED)

npzwrite("data/LR_Ising/ED_rho_real_N=$(MPOMC.params.N)_χ=$(MPOMC.params.χ)_J=$(MPOMC.params.J)_h=$(MPOMC.params.h)_γ=$(MPOMC.params.γ)_α=$α.npy", real.(ρED))
npzwrite("data/LR_Ising/ED_rho_imag_N=$(MPOMC.params.N)_χ=$(MPOMC.params.χ)_J=$(MPOMC.params.J)_h=$(MPOMC.params.h)_γ=$(MPOMC.params.γ)_α=$α.npy", imag.(ρED))


sqrt_a=sqrt(ρED)
f=(tr(sqrt(sqrt_a*ρ*sqrt_a)))^2
print(f)




#vec_basis=construct_vec_density_matrix_basis(MPOMC.params.N)

#Mx=real( own_x_magnetization(ρ,MPOMC.params,vec_basis) )
Mx=real( ED_magnetization(sx,ρED,MPOMC.params.N) )
println("True x-magnetization is: ", Mx)

My=real( ED_magnetization(sy,ρED,MPOMC.params.N) )
println("True y-magnetization is: ", My)

#Mz=real( own_z_magnetization(ρ,MPOMC.params,vec_basis) )
Mz=real( ED_magnetization(sz,ρED,MPOMC.params.N) )
println("True z-magnetization is: ", Mz)

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

npzwrite("data/LR_Ising/Mx_N=$(MPOMC.params.N)_χ=$(MPOMC.params.χ)_J=$(MPOMC.params.J)_h=$(MPOMC.params.h)_γ=$(MPOMC.params.γ)_α=$α.npy", list_of_Mx)
npzwrite("data/LR_Ising/My_N=$(MPOMC.params.N)_χ=$(MPOMC.params.χ)_J=$(MPOMC.params.J)_h=$(MPOMC.params.h)_γ=$(MPOMC.params.γ)_α=$α.npy", list_of_My)
npzwrite("data/LR_Ising/Mz_N=$(MPOMC.params.N)_χ=$(MPOMC.params.χ)_J=$(MPOMC.params.J)_h=$(MPOMC.params.h)_γ=$(MPOMC.params.γ)_α=$α.npy", list_of_Mz)


error()

ED_purity = calculate_purity(ρ)
insert!(list_of_purities, 1, ED_purity)
ED_entropy = -log2(ED_purity)
list_of_entropies = -log2.(list_of_purities)
insert!(list_of_entropies, 1, ED_entropy)

npzwrite("data/LR_Ising/purity N=$N χ=$χ J=$J h=$h γ=$γ α=$α.npy", list_of_purities)
npzwrite("data/LR_Ising/entropy N=$N χ=$χ J=$J h=$h γ=$γ α=$α.npy", list_of_entropies)