
include("VMPOMC.jl")
using .VMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random
using MPI

mpi_cache = set_mpi()

#Vincentini parameters: γ=1.0, J=0.5, h to be varied.

#Define constants:
const Jx= 0.0 #interaction strength
const Jy= 0.0 #interaction strength
const J = 0.5 #interaction strength
const hx= 0.5 #transverse field strength
const hz= 0.0 #transverse field strength
const γ = 1.0 #spin decay rate
const γ_d = 0.5 #spin decay rate
const α=1
N=16
const dim = 2^N
χ=8 #bond dimension
const burn_in = 0

params = Parameters(N,dim,χ,Jx,Jy,J,hx,hz,γ,γ_d,α, burn_in)

#Make single-body Lindbladian:
const l1 = conj( make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm) )
#const l1 = conj( one_body_Hamiltonian_term(hx*sx+hz*sz) + γ*one_body_Lindblad_term(sm) + γ_d*one_body_Lindblad_term(sz) )

#const basis=generate_bit_basis_reversed(N)

function make_two_body_Lindblad_Hamiltonian(A, B)
    L_H = -1im*( (A⊗id)⊗(B⊗id) - (id⊗transpose(A))⊗(id⊗transpose(B)) )
    return L_H
end
#const l2 = conj( Jx*make_two_body_Lindblad_Hamiltonian(sx,sx) + Jy*make_two_body_Lindblad_Hamiltonian(sy,sy) )




if mpi_cache.rank == 0
    Random.seed!(0)
    A_init=rand(ComplexF64, χ,χ,2,2)
    A=deepcopy(A_init)
    A=reshape(A,χ,χ,4)
    #MPI.Bcast!(A, root, comm)

    #display(A_init)

    list_of_L = Array{Float64}(undef, 0)
    list_of_Mx= Array{ComplexF64}(undef, 0)
    list_of_My= Array{ComplexF64}(undef, 0)
    list_of_Mz= Array{ComplexF64}(undef, 0)

    L=0
    acc=0
else
    #println(MPI.Comm_rank(comm))
    Random.seed!(mpi_cache.rank)
    A = Array{ComplexF64}(undef, χ,χ,4)
end
MPI.Bcast!(A, 0, mpi_cache.comm)


δ::Float64 = 0.01
F::Float64=0.995
ϵ::Float64=0.1#1


sampler = MetropolisSampler(20*χ^2, 0)
#optimizer = SR(sampler, A, l1, l2, ϵ, params, "Ising", "Local")
optimizer = SR(sampler, A, l1, ϵ, params, "Ising", "Local")
#optimizer = SR(sampler, A, l1, ϵ, params, "Ising", "Collective")

#@profview begin
@time begin
    for k in 1:300
        #N_MC = 1*χ^2
        for i in 1:5

            ComputeGradient!(optimizer)
            MPI_mean!(optimizer,mpi_cache.comm)

            if mpi_cache.rank == 0
                MPI_normalize!(optimizer,mpi_cache.nworkers)
                Optimize!(optimizer,δ*F^(k))
            end
            MPI.Bcast!(optimizer.A, 0, mpi_cache.comm)

        end

        #Record observables:
        if mpi_cache.rank == 0
            Af = reshape(optimizer.A,χ,χ,2,2) 
            Af_dagger = conj.(permutedims(Af,[1,2,4,3]))

            mx = real( 0.5*( tensor_calculate_magnetization(params,Af,sx) + tensor_calculate_magnetization(params,Af_dagger,sx) ) )
            my = real( 0.5*( tensor_calculate_magnetization(params,Af,sy) + tensor_calculate_magnetization(params,Af_dagger,sy) ) )
            mz = real( 0.5*( tensor_calculate_magnetization(params,Af,sz) + tensor_calculate_magnetization(params,Af_dagger,sz) ) )
    
            #L = calculate_mean_local_Lindbladian(MPOMC.params,l1,A,basis)
            #println("k=$k: ", real(L), " ; ", mz, " ; ", mx)
            println("k=$k: ", real(optimizer.optimizer_cache.mlL)/N, " ; acc_rate=", round(acc*100,sigdigits=2), "%", " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))

            ρ_1 = one_body_reduced_density_matrix(params, optimizer.A)
            ρ_1 += adjoint(ρ_1)
            ρ_1./=2
            println(tr(ρ_1*adjoint(ρ_1)))

            #println("sssf = ", calculate_steady_state_structure_factor(params,Af))

            push!(list_of_L,L)
            push!(list_of_Mx,mx)
            push!(list_of_My,my)
            push!(list_of_Mz,mz)
        end
    end
end

N = 16
params = Parameters(N,dim,χ,Jx,Jy,J,hx,hz,γ,γ_d,α, burn_in)
#optimizer = SR(sampler, optimizer.A, l1, ϵ, params, "Ising", "Local")
optimizer = SR(sampler, A, l1, ϵ, params, "Ising", "Collective")
optimizer.A = normalize_MPO!(optimizer.params, optimizer.A)

@time begin
    for k in 1:200
        #N_MC = 1*χ^2
        for i in 1:10

            ComputeGradient!(optimizer)
            MPI_mean!(optimizer,mpi_cache.comm)

            if mpi_cache.rank == 0
                MPI_normalize!(optimizer,mpi_cache.nworkers)
                Optimize!(optimizer,δ*F^(k))
            end
            MPI.Bcast!(optimizer.A, 0, mpi_cache.comm)

        end

        #Record observables:
        if mpi_cache.rank == 0
            Af = reshape(optimizer.A,χ,χ,2,2) 
            Af_dagger = conj.(permutedims(Af,[1,2,4,3]))

            mx = real( 0.5*( tensor_calculate_magnetization(params,Af,sx) + tensor_calculate_magnetization(params,Af_dagger,sx) ) )
            my = real( 0.5*( tensor_calculate_magnetization(params,Af,sy) + tensor_calculate_magnetization(params,Af_dagger,sy) ) )
            mz = real( 0.5*( tensor_calculate_magnetization(params,Af,sz) + tensor_calculate_magnetization(params,Af_dagger,sz) ) )
    
            #L = calculate_mean_local_Lindbladian(MPOMC.params,l1,A,basis)
            #println("k=$k: ", real(L), " ; ", mz, " ; ", mx)
            println("k=$k: ", real(optimizer.optimizer_cache.mlL)/N, " ; acc_rate=", round(acc*100,sigdigits=2), "%", " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))

            ρ_1 = one_body_reduced_density_matrix(params, optimizer.A)
            ρ_1 += adjoint(ρ_1)
            ρ_1./=2
            println(tr(ρ_1*adjoint(ρ_1)))

            #println("sssf = ", calculate_steady_state_structure_factor(params,Af))

            push!(list_of_L,L)
            push!(list_of_Mx,mx)
            push!(list_of_My,my)
            push!(list_of_Mz,mz)
        end
    end
end


if mpi_cache.rank == 0
    ρ_1 = one_body_reduced_density_matrix(params, optimizer.A)
    ρ_1 += adjoint(ρ_1)
    ρ_1./=2
    println(tr(ρ_1*adjoint(ρ_1)))

    error()

    #L = XYZ_Lindbald(params,"periodic")
    #L = sparse_DQIM(params, "periodic")
    L = sparse_DQIM_local_dephasing(params, "periodic")
    #L = sparse_DQIM_collective_dephasing(params, "periodic")

    #display(L); error()
    #display(imag.(L)); error()
    #display(real.(L)); error()
    vals, vecs = eigen_sparse(L)
    ρ=reshape(vecs,2^N,2^N)
    ρ./=tr(ρ)
    ρ=round.(ρ,digits = 12)

    Mx=real( magnetization(sx,ρ,params) )
    println("True x-magnetization is: ", Mx)

    My=real( magnetization(sy,ρ,params) )
    println("True y-magnetization is: ", My)

    Mz=real( magnetization(sz,ρ,params) )
    println("True z-magnetization is: ", Mz)

    #sssf = steady_state_structure_factor(ρ,params)
    #println("SSSF = ", sssf)
end