include("VMPOMC.jl")
using .VMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random
using MPI
using Dates
using JLD


mpi_cache = set_mpi()

#Vincentini parameters: γ=1.0, J=0.5, h to be varied.

#Define constants:
const Jx= 0.0 #interaction strength
const Jy= 0.0 #interaction strength
const J = 0.5 #interaction strength
#const hx= 1.0 #transverse field strength
const hz= 0.0 #transverse field strength
const γ = 1.0 #spin decay rate
const γ_d = 0.0 #spin decay rate
const α=0
#const N=10
#χ=8 #bond dimension
#const burn_in = 0

#set values from command line optional parameters:
N = parse(Int64,ARGS[1])
χ = parse(Int64,ARGS[2])
hx= parse(Float64,ARGS[3])

params = Parameters(N,χ,Jx,Jy,J,hx,hz,γ,γ_d,α)

#Make single-body Lindbladian:
#const l1 = conj( make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm) )

const l1 = conj( make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm) )

#const l1 = ( make_one_body_Lindbladian(-hx*sx-hz*sz,sqrt(γ)*sm) )
#display(l1)

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

    #display(A_init)

    #list_of_L = Array{Float64}(undef, 0)
    #list_of_Mx= Array{ComplexF64}(undef, 0)
    #list_of_My= Array{ComplexF64}(undef, 0)
    #list_of_Mz= Array{ComplexF64}(undef, 0)

    #list_of_L = open("list_of_C_chi$(χ)_N$(N).out", "w"); close(list_of_L)
    #list_of_Mx = open("list_of_Mx_chi$(χ)_N$(N).out", "w"); close(list_of_Mx)
    #list_of_My = open("list_of_Mx_chi$(χ)_N$(N).out", "w"); close(list_of_Mx)
    #list_of_Mz = open("list_of_Mx_chi$(χ)_N$(N).out", "w"); close(list_of_Mx)
    #list_of_P = open("list_of_P_chi$(χ)_N$(N).out", "w"); close(list_of_P)

    L=0
    acc=0
else
    #println(MPI.Comm_rank(comm))
    Random.seed!(mpi_cache.rank)
    A = Array{ComplexF64}(undef, χ,χ,4)
end
MPI.Bcast!(A, 0, mpi_cache.comm)


δ::Float64 = 0.01
F::Float64=0.9998
ϵ::Float64=0.1


sampler = MetropolisSampler(10*χ^2, 0)
optimizer = SR(sampler, A, l1, ϵ, params, "Ising")
N_iterations = 50

#Save parameters to file:
if mpi_cache.rank == 0

    #println(Base.summarysize(sampler))
    #println(Base.summarysize(optimizer))
    #sleep(10)
    #error()
    
    start = now()
    
    #Set directory path:
    dir = "Ising_decay_chi$(χ)_N$(N)_hx$(hx)_$(start)"
    isdir(dir) || mkdir(dir)
    cd(dir)

    list_of_parameters = open("Ising_decay_chi$(χ)_N$(N)_hx$(hx)_$(start).params", "w")#; close(list_of_L)
    redirect_stdout(list_of_parameters)
    #println(run(`git rev-parse --short HEAD`))
    display(params)
    display(mpi_cache)
    display(sampler)
    display(optimizer)
    println("\nN_iter\t", N_iterations)
    println("δ\t\t", δ)
    println("F\t\t", F)
    close(list_of_parameters)

    #o = open("Ising_decay_chi$(χ)_N$(N)_$(start).out", "w")
    #redirect_stdout(o)
    #close(o)
end



#@profview begin
@time begin
    for k in 1:N_iterations
        for i in 1:1#10

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
            if mod(k,10)==1

                o = open("Ising_decay_chi$(χ)_N$(N)_hx$(hx).out", "a")
                #redirect_stdout(o)
                println(o,"k=$k: ", real(optimizer.optimizer_cache.mlL)/N, " ; acc_rate=", round(acc*100,sigdigits=2), "%", " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))
                close(o)
            end

            #push!(list_of_L,L)
            #push!(list_of_Mx,mx)
            #push!(list_of_My,my)
            #push!(list_of_Mz,mz)

            list_of_C = open("list_of_C_χ$(χ)_N$(N)_hx$(hx).data", "a")
            println(list_of_C, real(optimizer.optimizer_cache.mlL)/N)
            close(list_of_C)
            list_of_mag = open("list_of_mag_χ$(χ)_N$(N)_hx$(hx).data", "a")
            println(list_of_mag, mx, ",", my, ",", mz)
            close(list_of_mag)
            list_of_P = open("list_of_P_χ$(χ)_N$(N)_hx$(hx).data", "a")
            P = real(tensor_purity(params, optimizer.A))
            println(list_of_P, P)
            close(list_of_P)

            save("MPO_density_matrix_χ$(χ)_N$(N)_hx$(hx).jld", "MPO_density_matrix", Af)
        end
    end
end
exit()


δ::Float64 = 0.03
F::Float64=0.99
ϵ::Float64=0.1

N=10
params = Parameters(N,2^N,χ,Jx,Jy,J,hx,hz,γ,γ_d,α, 0)


optimizer.A = normalize_MPO!(optimizer.params, optimizer.A)

sampler = MetropolisSampler(5*χ^2, 0)
optimizer = SR(sampler, optimizer.A, l1, ϵ, params, "Ising")



if mpi_cache.rank == 0
    L = sparse_DQIM(params, "periodic")

    vals, vecs = eigen_sparse(L)
    ρ=reshape(vecs,2^N,2^N)
    ρ./=tr(ρ)
    #ρ=round.(ρ,digits = 12)

    Mx=real( magnetization(sx,ρ,params) )
    println("True x-magnetization is: ", Mx)

    My=real( magnetization(sy,ρ,params) )
    println("True y-magnetization is: ", My)

    Mz=real( magnetization(sz,ρ,params) )
    println("True z-magnetization is: ", Mz)
end



#@profview begin
@time begin
    for k in 1:500
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

            #push!(list_of_L,L)
            #push!(list_of_Mx,mx)
            #push!(list_of_My,my)
            #push!(list_of_Mz,mz)
        end
    end
end

if mpi_cache.rank == 0
    L = sparse_DQIM(params, "periodic")

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
end
    