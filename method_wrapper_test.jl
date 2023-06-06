include("MPOMC.jl")
using .MPOMC
using NPZ
using Plots
using LinearAlgebra
#using TensorOperations
#using Distributions
#using Revise
import Random


#Vincentini parameters: γ=1.0, J=0.5, h to be varied.

#Define constants:
const Jx= 0.5 #interaction strength
const Jy= 0.0 #interaction strength
const J = 0.5 #interaction strength
const hx= 0.5 #transverse field strength
const hz= 0.0 #transverse field strength
const γ = 1.0 #spin decay rate
const α=1
const N=4
const dim = 2^N
χ=6 #bond dimension
const burn_in = 0

params = parameters(N,dim,χ,Jx,Jy,J,hx,hz,γ,α, burn_in)

#Make single-body Lindbladian:
const l1 = conj( make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm) )
#const l1 = ( make_one_body_Lindbladian(-hx*sx-hz*sz,sqrt(γ)*sm) )
#display(l1)

const basis=generate_bit_basis_reversed(N)



function make_two_body_Lindblad_Hamiltonian(A, B)
    L_H = -1im*( (A⊗id)⊗(B⊗id) - (id⊗transpose(A))⊗(id⊗transpose(B)) )
    return L_H
end
const l2 = Jx*make_two_body_Lindblad_Hamiltonian(sx,sx) + Jy*make_two_body_Lindblad_Hamiltonian(sy,sy)



list_of_L = Array{Float64}(undef, 0)
list_of_Mx= Array{ComplexF64}(undef, 0)
list_of_My= Array{ComplexF64}(undef, 0)
list_of_Mz= Array{ComplexF64}(undef, 0)

δ::Float64 = 0.02
F::Float64=0.99
ϵ::Float64=0.1

#display(A_init)

Random.seed!(1)

sampler = MetropolisSampler(20*χ^2, 0)
#optimizer_cache = Exact(A,params)

optimizer = Exact(sampler, l1, params, "Ising")
#optimizer = Exact(sampler, l1, l2, params, "LRIsing")
#optimizer = SGD(sampler, l1, params, "LRIsing")
#optimizer = SGD(sampler, l1, l2, params, "LRIsing")
#optimizer = SR(sampler, l1, ϵ, params, "LRIsing")
#optimizer = SR(sampler, l1, l2, ϵ, params, "LRIsing")

#display(optimizer.sampler)

#display(optimizer.A)

#error()

#display(typeof(basis)); error()

#@code_warntype UpdateSR!(optimizer); error()
#error()
#@profview begin
@time begin
    for k in 1:300
        L=0;LB=0
        acc::Float64=0
        for i in 1:10

            Optimize!(optimizer,basis,δ*F^(k))
            #Optimize!(optimizer,δ*F^(k))

            #display(optimizer.A); error()

            #new_A=zeros(ComplexF64, χ,χ,4)
            #∇,L=Exact_MPO_gradient(optimizer,basis)
            #∇./=maximum(abs.(∇))
            #new_A = optimizer.A - δ*F^(k)*∇#.*(1+0.5*rand())
            #global optimizer.A = new_A
            #global optimizer.A = normalize_MPO!(params, optimizer.A)

        end
        Af = reshape(optimizer.A,χ,χ,2,2) 
        Af_dagger = conj.(permutedims(Af,[1,2,4,3]))

        mx = real( 0.5*( tensor_calculate_magnetization(params,Af,sx) + tensor_calculate_magnetization(params,Af_dagger,sx) ) )
        my = real( 0.5*( tensor_calculate_magnetization(params,Af,sy) + tensor_calculate_magnetization(params,Af_dagger,sy) ) )
        mz = real( 0.5*( tensor_calculate_magnetization(params,Af,sz) + tensor_calculate_magnetization(params,Af_dagger,sz) ) )

        #L = calculate_mean_local_Lindbladian(MPOMC.params,l1,A,basis)
        #println("k=$k: ", real(L), " ; ", mz, " ; ", mx)
        println("k=$k: ", real(optimizer.optimizer_cache.mlL)/N, " ; acc_rate=", round(acc*100,sigdigits=2), "%", " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))
        #println("k=$k: ", real(L)/N, " ; acc_rate=", round(acc*100,sigdigits=2), "%", " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))

        push!(list_of_L,L)
        push!(list_of_Mx,mx)
        push!(list_of_My,my)
        push!(list_of_Mz,mz)

        #push!(list_of_purities, tensor_purity(MPOMC.params, A))

        #push!(list_of_density_matrices, make_density_matrix(MPOMC.params, A, basis))
    end
end
error()

for dist in 1:7
    println(calculate_spin_spin_correlation(MPOMC.params,A,sz,dist))
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

#L = XYZ_Lindbald(params,"periodic")
L=sparse_DQIM(params, "periodic")
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
Mx=real( magnetization(sx,ρ,params) )
println("True x-magnetization is: ", Mx)

My=real( magnetization(sy,ρ,params) )
println("True y-magnetization is: ", My)

#Mz=real( own_z_magnetization(ρ,MPOMC.params,vec_basis) )
Mz=real( magnetization(sz,ρ,params) )
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