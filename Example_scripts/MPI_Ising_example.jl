include("../VMPOMC.jl")
using .VMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random
using MPI
using Dates
using JLD


mpi_cache = set_mpi()

#Set parameters:
J = 0.5 #interaction strength
hx= 0.5 #transverse field strength
hz= 0.0 #longitudinal field strength
γ = 1.0 #spin decay rate
N = 20 #number of spins

#Set hyperparameters:
χ = 4 #MPO bond dimension
N_MC = 1000 #number of Monte Carlo samples
burn_in = 2 #Monte Carlo burn-in
δ = 0.15 #step size
F = 0.999
ϵ = 0.001
N_iterations = 1000

params = Parameters(N,χ,0.0,0.0,J,hx,hz,γ,0.0,0.0)

#Define one-body Lindbladian operator:
const l1 = make_one_body_Lindbladian(hx*sx+hz*sz, sqrt(γ)*sm)

#Save parameters to file:
dir = "Ising_decay_chi$(χ)_N$(N)_J$(J)_hx$(hx)_hz$(hz)_γ$(γ)"

if mpi_cache.rank == 0
    if isdir(dir)==true
        error("Directory already exists")
    end
    mkdir(dir)
    cd(dir)

    #Initialize random MPO:
    A_init = rand(ComplexF64,χ,χ,2,2)
    A = deepcopy(A_init)
    A = reshape(A,χ,χ,4)
end
MPI.Bcast!(A, 0, mpi_cache.comm)

#Define sampler and optimizer:
sampler = MetropolisSampler(N_MC, burn_in)
optimizer = Optimizer("SR", sampler, A, l1, ϵ, params, "Ising", "Local")

if mpi_cache.rank == 0
    #Save parameters to parameter file:
    list_of_parameters = open("Ising_decay.params", "w")
    redirect_stdout(list_of_parameters)
    display(params)
    display(sampler)
    display(optimizer)
    println("\nN_iter\t", N_iterations)
    println("δ\t\t", δ)
    println("F\t\t", F)
    close(list_of_parameters)
end

for k in 1:N_iterations

    #Optimize MPO:
    compute_gradient!(optimizer)
    MPI_mean!(optimizer, mpi_cache)
    if mpi_cache.rank == 0
        optimize!(optimizer, δ*F^(k))
    end
    MPI.Bcast!(optimizer.A, 0, mpi_cache.comm)

    if mpi_cache.rank == 0
        #Calculate steady-state magnetizations:
        mx = tensor_calculate_magnetization(optimizer, sx)
        my = tensor_calculate_magnetization(optimizer, sy)
        mz = tensor_calculate_magnetization(optimizer, sz)

        #Record iteration step:
        o = open("Ising_decay.out", "a")
        println(o,"k=$k: ", real(optimizer.optimizer_cache.mlL)/N, " ; acc_rate=", round(optimizer.optimizer_cache.acceptance*100,sigdigits=3), "%", " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))
        close(o)
    end
end
