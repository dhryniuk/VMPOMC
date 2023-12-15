export SGD, Optimize!, ComputeGradient!


mutable struct SGDCache{T} <: StochasticCache
    #Ensemble averages:
    L∂L::Array{T,3}
    ΔLL::Array{T,3}

    #Sums:
    mlL::T
    acceptance::Float64

    #Gradient:
    ∇::Array{T,3}
end

function SGDCache(A::Array{T,3}, params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    cache=SGDCache(
        zeros(T,params.χ,params.χ,4),
        zeros(T,params.χ,params.χ,4),
        convert(T,0),
        0.0,
        zeros(T,params.χ,params.χ,4)
    )  
    return cache
end

abstract type SGD{T} <: Stochastic{T} end

mutable struct SGDl1{T<:Complex{<:AbstractFloat}} <: SGD{T}

    #MPO:
    A::Array{T,3}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::SGDCache{T}

    #1-local Lindbladian:
    l1::Matrix{T}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters

    #Workspace:
    workspace::Workspace{T}

end

Base.display(optimizer::SGDl1) = begin
    println("\nOptimizer:")
    println("method\t\tSGD-l1")
    println("ising_op\t", optimizer.ising_op)
    println("dephasing_op\t", optimizer.dephasing_op)
    println("l1\t\t",optimizer.l1)
end

#Constructor:
function SGD(sampler::MetropolisSampler, A::Array{T,3}, l1::Matrix{T}, params::Parameters, ising_op::String="Ising", dephasing_op::String="Local") where {T<:Complex{<:AbstractFloat}} 
    #A = rand(ComplexF64,params.χ,params.χ,4)
    if ising_op=="Ising"
        if dephasing_op=="Local"
            optimizer = SGDl1(A, sampler, SGDCache(A, params), l1, Ising(), LocalDephasing(), params, set_workspace(A, params))
        elseif dephasing_op=="Collective"
            optimizer = SGDl1(A, sampler, SGDCache(A, params), l1, Ising(), CollectiveDephasing(), params, set_workspace(A, params))        
        else
            error("Unrecognized eigen-operation")
        end
    elseif ising_op=="LongRangeIsing" || ising_op=="LRIsing" || ising_op=="Long Range Ising"
        @assert params.α>0
        optimizer = SGDl1(A, sampler, SGDCache(A, params), l1, LongRangeIsing(params), LocalDephasing(), params, set_workspace(A, params))
    else
        error("Unrecognized eigen-operation")
    end
    return optimizer
end

mutable struct SGDl2{T<:Complex{<:AbstractFloat}} <: SGD{T}

    #MPO:
    A::Array{T,3}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::SGDCache{T}

    #1-local Lindbladian:
    l1::Matrix{T}

    #2-local Lindbladian:
    l2::Matrix{T}

    #Diagonal operators:
    ising_op::IsingInteraction

    #Parameters:
    params::Parameters

    #Workspace:
    workspace::Workspace{T}

end

#Constructor:
function SGD(sampler::MetropolisSampler, A::Array{T,3}, l1::Matrix{T}, l2::Matrix{T}, params::Parameters, ising_op::String="Ising") where {T<:Complex{<:AbstractFloat}} 
    #A = rand(ComplexF64,params.χ,params.χ,4)
    if ising_op=="Ising"
        optimizer = SGDl2(A, sampler, SGDCache(A, params), l1, l2, Ising(), params, set_workspace(A, params))
    elseif ising_op=="LongRangeIsing" || ising_op=="LRIsing" || ising_op=="Long Range Ising"
        @assert params.α>0
        optimizer = SGDl2(A, sampler, SGDCache(A, params), l1, l2, LongRangeIsing(params), params, set_workspace(A, params))
    else
        error("Unrecognized eigen-operation")
    end
    return optimizer
end

function Initialize!(optimizer::SGD{T}) where {T<:Complex{<:AbstractFloat}}
    optimizer.optimizer_cache = SGDCache(optimizer.A, optimizer.params)
    optimizer.workspace = set_workspace(optimizer.A, optimizer.params)
end

function SweepLindblad!(sample::Projector, ρ_sample::T, optimizer::SGDl1{T}) where {T<:Complex{<:AbstractFloat}} 

    params = optimizer.params
    micro_sample = optimizer.workspace.micro_sample
    micro_sample = Projector(sample)

    temp_local_L::T = 0
    temp_local_∇L::Array{T,3} = zeros(T,params.χ,params.χ,4)
    #temp_local_L = optimizer.cache.temp_local_L
    #temp_local_L = 0
    #temp_local_∇L = optimizer.cache.temp_local_∇L
    #temp_local_∇L = zeros(T,params.χ,params.χ,4)

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        temp_local_L, temp_local_∇L = one_body_Lindblad_term!(temp_local_L, temp_local_∇L, sample, micro_sample, j, optimizer)
    end

    temp_local_L  /= ρ_sample
    temp_local_∇L./= ρ_sample

    #optimizer.local_L += temp_local_L
    #optimizer.local_∇L+= temp_local_∇L

    return temp_local_L, temp_local_∇L
end

function SweepLindblad!(sample::Projector, ρ_sample::T, optimizer::SGDl2{T}) where {T<:Complex{<:AbstractFloat}} 

    params=optimizer.params
    micro_sample = optimizer.workspace.micro_sample
    micro_sample = Projector(sample)

    local_L::T = 0
    local_∇L::Array{T,3} = zeros(T,params.χ,params.χ,4)

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        local_L, local_∇L = one_body_Lindblad_term!(local_L, local_∇L, sample, micro_sample, j, optimizer)
    end
    for j::UInt8 in 1:params.N-1
        local_L, local_∇L = two_body_Lindblad_term!(local_L, local_∇L, sample, micro_sample, j, optimizer)
    end
    if params.N>2
        local_L, local_∇L = boundary_two_body_Lindblad_term!(local_L, local_∇L, sample, micro_sample, optimizer)
    end

    local_L  /= ρ_sample
    local_∇L./= ρ_sample

    return local_L, local_∇L
end

function Update!(optimizer::Stochastic{T}, sample::Projector) where {T<:Complex{<:AbstractFloat}} #... the ensemble averages etc.

    params=optimizer.params
    A=optimizer.A
    data=optimizer.optimizer_cache
    cache = optimizer.workspace

    #Initialize auxiliary arrays:
    #local_L = cache.local_L
    #local_∇L = cache.local_∇L
    #l_int = cache.l_int
    local_L = 0
    local_∇L = zeros(T,params.χ,params.χ,4)
    l_int = 0
    cache.local_∇L_diagonal_coeff = 0

    ρ_sample::T = tr(cache.R_set[params.N+1])
    cache.L_set = L_MPO_strings!(cache.L_set, sample,A,params,cache)
    cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache)./ρ_sample

    #Sweep lattice:
    local_L, local_∇L = SweepLindblad!(sample, ρ_sample, optimizer)
    #SweepLindblad!(sample, ρ_sample, optimizer)

    #Add in diagonal part of the local derivative:
    local_∇L.+=cache.local_∇L_diagonal_coeff.*cache.Δ

    #Add in Ising interaction terms:
    l_int = Ising_interaction_energy(optimizer.ising_op, sample, optimizer)
    l_int += Dephasing_term(optimizer.dephasing_op, sample, optimizer)
    local_L  +=l_int
    local_∇L.+=l_int*cache.Δ

    #Update L∂L* ensemble average:
    data.L∂L.+=local_L*conj(local_∇L)

    #Update ΔLL ensemble average:
    data.ΔLL.+=cache.Δ

    #Mean local Lindbladian:
    data.mlL += local_L*conj(local_L)
end

function Finalize!(optimizer::SGD{T}) where {T<:Complex{<:AbstractFloat}}
    N_MC = optimizer.sampler.N_MC
    data = optimizer.optimizer_cache

    data.mlL /= N_MC
    data.ΔLL .= conj.(data.ΔLL) #remember to take the complex conjugate
    data.ΔLL .*= data.mlL
    data.∇ = (data.L∂L-data.ΔLL)/N_MC
end

function ComputeGradient!(optimizer::SGD{T}) where {T<:Complex{<:AbstractFloat}}

    Initialize!(optimizer)
    sample = optimizer.workspace.sample

    # Initialize sample and L_set for that sample:
    sample = MPO_Metropolis_burn_in(optimizer)

    mags = []

    for _ in 1:optimizer.sampler.N_MC

        #Generate sample:
        sample, acc = Mono_Metropolis_sweep_left(sample, optimizer)
        optimizer.optimizer_cache.acceptance += acc/(optimizer.params.N*optimizer.sampler.N_MC)

        #Compute local estimators:
        Update!(optimizer, sample) 

        push!(mags, compute_z_magnetization(sample,optimizer))
    end
    #Finalize!(optimizer)

    println("AVERAGE MAG = ", mean(mags))
end

function Optimize!(optimizer::SGD{T}, δ::Float64) where {T<:Complex{<:AbstractFloat}}

    #ComputeGradient!(optimizer)

    Finalize!(optimizer)

    ∇ = optimizer.optimizer_cache.∇
    ∇./=maximum(abs.(∇))

    new_A = similar(optimizer.A)
    new_A = optimizer.A - δ*∇
    optimizer.A = new_A
    optimizer.A = normalize_MPO!(optimizer.params, optimizer.A)
end

function MPI_mean!(optimizer::SGD{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    par_cache = optimizer.optimizer_cache

    MPI.Allreduce!(par_cache.L∂L, +, mpi_cache.comm)
    MPI.Allreduce!(par_cache.ΔLL, +, mpi_cache.comm)

    mlL = [par_cache.mlL]
    MPI.Reduce!(mlL, +, mpi_cache.comm, root=0)
    if mpi_cache.rank == 0
        par_cache.mlL = mlL[1]/mpi_cache.nworkers
        par_cache.L∂L./=mpi_cache.nworkers
        par_cache.ΔLL./=mpi_cache.nworkers
    end
end

function compute_z_magnetization(sample, optimizer)
    mag = 0
    N = optimizer.params.N
    if sample.ket==sample.bra
        for j in 1:N
            mag += (1-2*sample.ket[j])
        end
    end
    #return mag*MPO(optimizer.params,sample,optimizer.A)/N
    return mag/(N*conj(MPO(optimizer.params,sample,optimizer.A)))
end

function compute_x_magnetization(sample, optimizer)
    mag = 0
    N = optimizer.params.N
    for j in 1:N
        sample_flipped = deepcopy(sample)
        sample_flipped.ket[j]=!sample_flipped.ket[j]
        if sample_flipped.ket==sample.bra
            mag += 1
        end
    end
    return mag/(N*conj(MPO(optimizer.params,sample,optimizer.A)))
end

function compute_z_magnetization(sample, optimizer)
    mag = 0
    N = optimizer.params.N
    if sample.ket==sample.bra
        for j in 1:N
            mag += (1-2*sample.ket[j])
        end
    end
    #return mag*MPO(optimizer.params,sample,optimizer.A)/N
    return mag/(N*conj(MPO(optimizer.params,sample,optimizer.A)))
end

function compute_magnetization(op,sample,optimizer)
    if op==sx
        return compute_x_magnetization(sample, optimizer)
    elseif op==sz
        return compute_z_magnetization(sample, optimizer)
    else
        error()
    end
end