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
#const γ_d = 0.0 #spin decay rate
const α=1#0000
#const N=10
#χ=12 #bond dimension
#const burn_in = 0


#set values from command line optional parameters:
N = parse(Int64,ARGS[1])
const γ_d = 0
hx = parse(Float64,ARGS[2])
χ = parse(Int64,ARGS[3])


params = Parameters(N,χ,Jx,Jy,J,hx,hz,γ,γ_d,α)

const l1 = make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm)

N_MC::Int64 = 10*4*χ^2
δ::Float64 = 0.01
F::Float64 = 0.9999
ϵ::Float64 = parse(Float64,ARGS[4])
N_iterations::Int64 = 10000
last_iteration_step::Int64 = 1

#Save parameters to file:
if mpi_cache.rank == 0
    
    start = now()
    
    dir = "Ising_decay_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d)"

    if isdir(dir)==false
        mkdir(dir)
        cd(dir)

        path = open("pwd.txt", "w")
        current_directory = dirname(pwd())
        println(path, current_directory)
        close(path)

        Random.seed!(0)
        A_init=rand(ComplexF64, χ,χ,2,2)
        A=deepcopy(A_init)
        A=reshape(A,χ,χ,4)

        sampler = MetropolisSampler(N_MC, 0)
        optimizer = Optimizer("SR", sampler, A, l1, ϵ, params, "LRIsing", "Local")

        list_of_parameters = open("Ising_decay_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).params", "w")
        redirect_stdout(list_of_parameters)
        display(params)
        display(mpi_cache)
        display(sampler)
        display(optimizer)
        println("\nN_iter\t", N_iterations)
        println("δ\t\t", δ)
        println("F\t\t", F)
        close(list_of_parameters)
    else
        cd(dir)
        list_of_C = open("list_of_C_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).data", "r")
        last_iteration_step=countlines(list_of_C)+1

        ### NEED TO ALSO CHECK IF OTHER PARAMETERS ARE THE SAME BY EXPLICITLY COMPARING THE list_of_parameters FILES
    
        A_init = load("MPO_density_matrix_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).jld")["MPO_density_matrix"]
        A = reshape(A_init,χ,χ,4)
        A = normalize_MPO!(params, A)
    end
    L=0
    acc=0
    t0::Float64=0
    a::Float64=0
    b::Float64=0
    c::Float64=0
    d::Float64=0
else
    Random.seed!(mpi_cache.rank)
    A = Array{ComplexF64}(undef, χ,χ,4)
end
MPI.bcast(last_iteration_step, mpi_cache.comm)
MPI.Bcast!(A, 0, mpi_cache.comm)

sampler = MetropolisSampler(N_MC, 0)
optimizer = Optimizer("SR", sampler, A, l1, ϵ, params, "LRIsing", "Local")


if mpi_cache.rank == 0
    global t0 = time()
end
for k in last_iteration_step:N_iterations
    for i in 1:1
        if mpi_cache.rank == 0
            global a = time()
        end
        ComputeGradient!(optimizer)
        #ComputeGradient!(optimizer, basis)
        if mpi_cache.rank == 0
            global b = time()
        end
        MPI_mean!(optimizer,mpi_cache)
        if mpi_cache.rank == 0
            global c = time()
            Optimize!(optimizer,δ*F^(k))
            #Optimize!(optimizer,basis,δ*F^(k))
            global d = time()
        end
        MPI.Bcast!(optimizer.A, 0, mpi_cache.comm)
        if mpi_cache.rank == 0
            global e = time()
        end
    end

    #Record observables:
    if mpi_cache.rank == 0
        Af = reshape(optimizer.A,χ,χ,2,2) 
        Af_dagger = conj.(permutedims(Af,[1,2,4,3]))

        mx = real(tensor_calculate_magnetization(params,Af,sx))
        my = real(tensor_calculate_magnetization(params,Af,sy))
        mz = real(tensor_calculate_magnetization(params,Af,sz))

        sxx = real( tensor_calculate_correlation(params,Af,sx))
        syy = real( tensor_calculate_correlation(params,Af,sy))
        szz = real( tensor_calculate_correlation(params,Af,sz))

        o = open("mem.out", "a")
        println(o, "k=$k: ", Base.Sys.free_memory())
        close(o)

        if mod(k,10)==1

            o = open("Ising_decay_chi$(χ)_N$(N)_hx$(hx).out", "a")
            #redirect_stdout(o)
            println(o,"k=$k: ", real(optimizer.optimizer_cache.mlL)/N, " ; acc_rate=", round(acc*100,sigdigits=2), "%", " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))
            println(o,Base.Sys.free_memory())
            close(o)
        end

        list_of_C = open("list_of_C_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).data", "a")
        println(list_of_C, real(optimizer.optimizer_cache.mlL)/N)
        close(list_of_C)
        #list_of_acceptance = open("list_of_acceptance_rates_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).data", "a")
        #println(list_of_acceptance, round(real(optimizer.optimizer_cache.acceptance),digits=6))
        #close(list_of_acceptance)
        list_of_mag = open("list_of_mag_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).data", "a")
        println(list_of_mag, mx, ",", my, ",", mz)
        close(list_of_mag)
        list_of_P = open("list_of_P_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).data", "a")
        P = real(tensor_purity(params, optimizer.A))
        println(list_of_P, P)
        close(list_of_P)
        list_of_cor = open("list_of_cor_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).data", "a")
        println(list_of_cor, sxx, ",", syy, ",", szz)
        close(list_of_cor)

        save("MPO_density_matrix_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).jld", "MPO_density_matrix", Af)
        #sleep(1)
        #save("MPO_density_matrix_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d)_backup.jld", "MPO_density_matrix", Af)
        global f = time()
        list_of_times = open("list_of_times.data", "a")
        println(list_of_times, "Total for step ", k, ": ", round(f-a; sigdigits = 3))
        println(list_of_times, "In parts: ", round(b-a; sigdigits = 3), " ; ", round(c-b; sigdigits = 3), " ; ", round(d-c; sigdigits = 3), " ; ", round(e-d; sigdigits = 3), " ; ", round(f-e; sigdigits = 3))
        close(list_of_times)
    end

    if mod(k,10)==0
        GC.gc()
    end
end

if mpi_cache.rank == 0
    list_of_times = open("list_of_times.data", "a")
    println(list_of_times, "Sum total: ", f-t0)
    close(list_of_times)
end