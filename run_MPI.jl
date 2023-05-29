#using MPI
#MPI.Init()

#const comm = MPI.COMM_WORLD
#const root = 0
#const nworkers = MPI.Comm_size(comm) - 1

#using Distributed

#addprocs(8)
#println(nprocs())
#println(nworkers())

#@everywhere include("MPOMC.jl")
#@everywhere using .MPOMC
include("MPOMC.jl")
using .MPOMC
using NPZ
using Plots
using LinearAlgebra
#using TensorOperations
#using Distributions
#using Revise
import Random
using MPI

MPI.Init()
comm = MPI.COMM_WORLD
root = 0
nworkers = max(1,MPI.Comm_size(comm) - 1)

#Vincentini parameters: γ=1.0, J=0.5, h to be varied.

#Define constants:
const Jx= 0.0 #interaction strength
const Jy= 0.0 #interaction strength
const J = 0.5 #interaction strength
const hx= 0.5 #transverse field strength
const hz= 0.0 #transverse field strength
const γ = 1.0 #spin decay rate
const α=3
const N=8
const dim = 2^N
χ=4 #bond dimension
const burn_in = 0

MPOMC.set_parameters(N,χ,Jx,Jy,J,hx,hz,γ,α, burn_in)

#Make single-body Lindbladian:
const l1 = conj( make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm) )
#const l1 = ( make_one_body_Lindbladian(-hx*sx-hz*sz,sqrt(γ)*sm) )
#display(l1)

#const basis=generate_bit_basis_reversed(N)


if MPI.Comm_rank(comm) == root
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
    Random.seed!(MPI.Comm_rank(comm))
    A = Array{ComplexF64}(undef, χ,χ,4)
end
MPI.Bcast!(A, root, comm)

δ::Float16 = 0.03
F::Float16=0.99
ϵ::Float64=0.1



#@profview begin
@time begin
    for k in 1:100
        N_MC = 3*χ^2
        for i in 1:10

            par_cache = set_SR_cache(A,MPOMC.params)
            reduced_one_worker_MPI_SR_MPO_gradient(A,l1,N_MC,ϵ,MPOMC.params,comm,par_cache)

            if MPI.Comm_rank(comm) == root
                global A, L, acc = MPI_SR_MPO_optimize!(par_cache, δ*F^(k), A, N_MC, ϵ, MPOMC.params, nworkers)
            end
            MPI.Bcast!(A, root, comm)

        end

        #Record observables:
        if MPI.Comm_rank(comm) == root
            Af = reshape(A,χ,χ,2,2) 
            Af_dagger = conj.(permutedims(Af,[1,2,4,3]))

            mx = real( 0.5*( tensor_calculate_magnetization(MPOMC.params,Af,sx) + tensor_calculate_magnetization(MPOMC.params,Af_dagger,sx) ) )
            my = real( 0.5*( tensor_calculate_magnetization(MPOMC.params,Af,sy) + tensor_calculate_magnetization(MPOMC.params,Af_dagger,sy) ) )
            mz = real( 0.5*( tensor_calculate_magnetization(MPOMC.params,Af,sz) + tensor_calculate_magnetization(MPOMC.params,Af_dagger,sz) ) )

            #L = calculate_mean_local_Lindbladian(MPOMC.params,l1,A,basis)
            #println("k=$k: ", real(L), " ; ", mz, " ; ", mx)
            println("k=$k: ", real(L)/N, " ; acc_rate=", round(acc*100,sigdigits=2), "%", " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))

            push!(list_of_L,L)
            push!(list_of_Mx,mx)
            push!(list_of_My,my)
            push!(list_of_Mz,mz)
        end
    end
end
#end