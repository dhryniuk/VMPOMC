export SGD, optimize!, compute_gradient!


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

    if ising_op=="Ising"
        if dephasing_op=="Local"
            optimizer = SGDl1(A, sampler, SGDCache(A, params), l1, Ising(), LocalDephasing(), params, set_workspace(A, params))
        elseif dephasing_op=="Collective"
            optimizer = SGDl1(A, sampler, SGDCache(A, params), l1, Ising(), CollectiveDephasing(), params, set_workspace(A, params))        
        else
            error("Unrecognized dephasing operator")
        end
    elseif ising_op=="LongRangeIsing" || ising_op=="LRIsing" || ising_op=="Long Range Ising"
        @assert params.α>0
        optimizer = SGDl1(A, sampler, SGDCache(A, params), l1, LongRangeIsing(params), LocalDephasing(), params, set_workspace(A, params))
    else
        error("Unrecognized Ising interaction")
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
        error("Unrecognized Ising interaction")
    end
    return optimizer
end

function initialize!(optimizer::SGD{T}) where {T<:Complex{<:AbstractFloat}}
    optimizer.optimizer_cache = SGDCache(optimizer.A, optimizer.params)
    optimizer.workspace = set_workspace(optimizer.A, optimizer.params)
end

function sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::SGDl1{T}) where {T<:Complex{<:AbstractFloat}} 

    params = optimizer.params
    sub_sample = optimizer.workspace.sub_sample
    sub_sample = Projector(sample)

    temp_local_L::T = 0
    temp_local_∇L::Array{T,3} = zeros(T, params.χ, params.χ, 4)

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        temp_local_L, temp_local_∇L = one_body_Lindblad_term!(temp_local_L, temp_local_∇L, sample, sub_sample, j, optimizer)
    end

    temp_local_L /= ρ_sample
    temp_local_∇L ./= ρ_sample

    return temp_local_L, temp_local_∇L
end

function sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::SGDl2{T}) where {T<:Complex{<:AbstractFloat}} 

    params = optimizer.params
    sub_sample = optimizer.workspace.sub_sample
    sub_sample = Projector(sample)

    local_L::T = 0
    local_∇L::Array{T,3} = zeros(T, params.χ, params.χ, 4)

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        local_L, local_∇L = one_body_Lindblad_term!(local_L, local_∇L, sample, sub_sample, j, optimizer)
    end
    for j::UInt8 in 1:params.N-1
        local_L, local_∇L = two_body_Lindblad_term!(local_L, local_∇L, sample, sub_sample, j, optimizer)
    end
    if params.N>2
        local_L, local_∇L = boundary_two_body_Lindblad_term!(local_L, local_∇L, sample, sub_sample, optimizer)
    end

    local_L /= ρ_sample
    local_∇L ./= ρ_sample

    return local_L, local_∇L
end

function update!(optimizer::Stochastic{T}, sample::Projector) where {T<:Complex{<:AbstractFloat}} #... the ensemble averages etc.

    params = optimizer.params
    A = optimizer.A
    data = optimizer.optimizer_cache
    ws = optimizer.workspace

    #Initialize auxiliary arrays:
    #local_L = ws.local_L
    #local_∇L = ws.local_∇L
    #l_int = ws.l_int
    local_L = 0
    local_∇L = zeros(T,params.χ,params.χ,4)
    l_int = 0
    ws.local_∇L_diagonal_coeff = 0

    ρ_sample::T = tr(ws.R_set[params.N+1])
    ws.L_set = L_MPO_products!(ws.L_set, sample, A, params, ws)
    ws.Δ = ∂MPO(sample, ws.L_set, ws.R_set, params, ws)./ρ_sample

    #Sweep lattice:
    local_L, local_∇L = sweep_Lindblad!(sample, ρ_sample, optimizer)
    #sweep_Lindblad!(sample, ρ_sample, optimizer)

    #Add in diagonal part of the local derivative:
    local_∇L.+=ws.local_∇L_diagonal_coeff.*ws.Δ

    #Add in Ising interaction terms:
    l_int = Ising_interaction_energy(optimizer.ising_op, sample, optimizer)
    l_int += Dephasing_term(optimizer.dephasing_op, sample, optimizer)
    local_L  +=l_int
    local_∇L.+=l_int*ws.Δ

    #Update L∂L* ensemble average:
    data.L∂L.+=local_L*conj(local_∇L)

    #Update ΔLL ensemble average:
    data.ΔLL.+=ws.Δ

    #Mean local Lindbladian:
    data.mlL += local_L*conj(local_L)
end

function finalize!(optimizer::SGD{T}) where {T<:Complex{<:AbstractFloat}}

    N_MC = optimizer.sampler.N_MC
    data = optimizer.optimizer_cache

    data.mlL /= N_MC
    data.ΔLL .= conj.(data.ΔLL)
    data.ΔLL .*= data.mlL
    data.∇ = (data.L∂L-data.ΔLL)/N_MC
end

function compute_gradient!(optimizer::SGD{T}) where {T<:Complex{<:AbstractFloat}}

    initialize!(optimizer)
    sample = optimizer.workspace.sample

    # Initialize sample and L_set for that sample:
    sample = Metropolis_burn_in!(optimizer)

    #mags = []

    for _ in 1:optimizer.sampler.N_MC
        #Generate sample:
        sample, acc = Metropolis_sweep_left!(sample, optimizer)
        optimizer.optimizer_cache.acceptance += acc/(optimizer.params.N*optimizer.sampler.N_MC)

        #Compute local estimators:
        update!(optimizer, sample) 

        #push!(mags, compute_z_magnetization(sample,optimizer))
    end
end

function optimize!(optimizer::SGD{T}, δ::Float64) where {T<:Complex{<:AbstractFloat}}

    finalize!(optimizer)

    ∇ = optimizer.optimizer_cache.∇
    ∇ ./= maximum(abs.(∇))

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
        par_cache.L∂L ./= mpi_cache.nworkers
        par_cache.ΔLL ./= mpi_cache.nworkers
    end
end