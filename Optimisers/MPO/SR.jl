export SR, optimize!, compute_gradient!, MPI_mean!, MPI_normalize!


mutable struct SRCache{T} <: StochasticCache
    
    #Ensemble averages:
    L∂L::Array{T,3}
    ΔLL::Array{T,3}

    #Sums:
    mlL::T
    acceptance::Float64#UInt64

    #Gradient:
    ∇::Array{T,3}

    # Metric tensor:
    S::Array{T,2}
    avg_G::Array{T}
end

function SRCache(A::Array{T,3},params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    cache=SRCache(
        zeros(T,params.χ,params.χ,4),
        zeros(T,params.χ,params.χ,4),
        convert(T,0),
        0.0,
        zeros(T,params.χ,params.χ,4),
        zeros(T,4*params.χ^2,4*params.χ^2),
        zeros(T,4*params.χ^2)
    )  
    return cache
end

abstract type SR{T} <:  Stochastic{T} end

mutable struct SRl1{T<:Complex{<:AbstractFloat}} <: SR{T}

    #MPO:
    A::Array{T,3}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::SRCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters
    ϵ::Float64

    #Workspace:
    workspace::Workspace{T}#Union{workspace,Nothing}

end

Base.display(optimizer::SRl1) = begin
    println("\nOptimizer:")
    println("method\t\tSR-l1")
    println("ϵ\t\t", optimizer.ϵ)
    println("ising_op\t", optimizer.ising_op)
    println("dephasing_op\t", optimizer.dephasing_op)
    println("l1\t\t",optimizer.l1)
end


#Constructor:
function SR(sampler::MetropolisSampler, A::Array{T,3}, l1::Matrix{T}, ϵ::Float64, params::Parameters, ising_op::String="Ising", dephasing_op::String="Local") where {T<:Complex{<:AbstractFloat}} 

    if ising_op=="Ising"
        if dephasing_op=="Local"
            optimizer = SRl1(A, sampler, SRCache(A, params), l1, Ising(), LocalDephasing(), params, ϵ, set_workspace(A, params))
        elseif dephasing_op=="Collective"
            optimizer = SRl1(A, sampler, SRCache(A, params), l1, Ising(), CollectiveDephasing(), params, ϵ, set_workspace(A, params))
        else
            error("Unrecognized dephasing operator")
        end
    elseif ising_op=="LongRangeIsing" || ising_op=="LRIsing" || ising_op=="Long Range Ising"
        @assert params.α>=0
        optimizer = SRl1(A, sampler, SRCache(A, params), l1, LongRangeIsing(params), LocalDephasing(), params, ϵ, set_workspace(A, params))
    else
        error("Unrecognized Ising interaction")
    end
    return optimizer
end

mutable struct SRl2{T<:Complex{<:AbstractFloat}} <: SR{T}

    #MPO:
    A::Array{T,3}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::SRCache{T}

    #1-local Lindbladian:
    l1::Matrix{T}

    #2-local Lindbladian:
    l2::Matrix{T}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters
    ϵ::Float64

    #Workspace:
    workspace::Workspace{T}

end

#Constructor:
function SR(sampler::MetropolisSampler, A::Array{T,3}, l1::Matrix{T}, l2::Matrix{T}, ϵ::Float64, params::Parameters, ising_op::String="Ising", dephasing_op::String="Local") where {T<:Complex{<:AbstractFloat}} 

    if ising_op=="Ising"
        if dephasing_op=="Local"
            optimizer = SRl2(A, sampler, SRCache(A, params), l1, l2, Ising(), LocalDephasing(), params, ϵ, set_workspace(A, params))
        elseif dephasing_op=="Collective"
            optimizer = SRl2(A, sampler, SRCache(A, params), l1, l2, Ising(), CollectiveDephasing(), params, ϵ, set_workspace(A, params))
        else
            error("Unrecognized dephasing operator")
        end
    elseif ising_op=="LongRangeIsing" || ising_op=="LRIsing" || ising_op=="Long Range Ising"
        @assert params.α>0
        optimizer = SRl2(A, sampler, SRCache(A, params), l1, l2, LongRangeIsing(params), LocalDephasing(), params, ϵ, set_workspace(A, params))
    else
        error("Unrecognized Ising interaction")
    end
    return optimizer
end

function initialize!(optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}}

    optimizer.optimizer_cache = SRCache(optimizer.A, optimizer.params)
    optimizer.workspace = set_workspace(optimizer.A, optimizer.params)
end


#### REPLACE WITH HOLY TRAITS ---

function sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::SRl1{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    sub_sample = optimizer.workspace.sub_sample
    sub_sample = Projector(sample)

    temp_local_L::T = 0
    temp_local_∇L::Array{T,3} = zeros(T, params.χ, params.χ, 4)
    #temp_local_L = optimizer.workspace.temp_local_L
    #temp_local_L = 0.0+0.0im
    #temp_local_∇L = optimizer.workspace.temp_local_∇L
    #temp_local_∇L = zeros(T,params.χ,params.χ,4)

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        temp_local_L, temp_local_∇L = one_body_Lindblad_term!(temp_local_L, temp_local_∇L, sample, sub_sample, j, optimizer)
    end

    temp_local_L /= ρ_sample
    temp_local_∇L ./= ρ_sample

    return temp_local_L, temp_local_∇L
end

function sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::SRl2{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    sub_sample = optimizer.workspace.sub_sample
    sub_sample = Projector(sample)

    local_L::T = 0
    local_∇L::Array{T,3} = zeros(T,params.χ,params.χ,4)

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

#### ---.


function update_SR!(optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}}
    S::Array{T,2} = optimizer.optimizer_cache.S
    avg_G::Vector{T} = optimizer.optimizer_cache.avg_G
    params::Parameters = optimizer.params
    ws = optimizer.workspace
    
    G::Vector{T} = reshape(ws.Δ,4*params.χ^2)
    conj_G = conj(G)
    avg_G .+= G
    mul!(ws.plus_S,conj_G,transpose(G))
    S .+= ws.plus_S 
end

function reconfigure!(optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor
    data = optimizer.optimizer_cache
    N_MC = optimizer.sampler.N_MC
    ϵ = optimizer.ϵ
    params = optimizer.params

    #Compute metric tensor:
    rmul!(data.S, 1/N_MC)
    rmul!(data.avg_G, 1/N_MC)
    conj_avg_G = conj(data.avg_G)
    data.S -= data.avg_G*transpose(conj_avg_G)

    # Regularize the metric tensor
    @inbounds for i in 1:size(data.S, 1)
        data.S[i,i] += ϵ
    end
    
    # Reconfigure gradient
    grad_size = (params.χ, params.χ, 4)
    flat_grad_size = 4*params.χ^2
    
    grad = reshape(view(data.L∂L, :), grad_size) .- reshape(view(data.ΔLL, :), grad_size)
    rmul!(grad, 1/N_MC)
    flat_grad = reshape(grad, flat_grad_size)
    
    flat_grad = inv(data.S)*flat_grad
    data.∇ = reshape(flat_grad, params.χ, params.χ, 4)
end

function finalize!(optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}}
    N_MC = optimizer.sampler.N_MC
    data = optimizer.optimizer_cache

    data.mlL /= N_MC
    data.ΔLL .= conj.(data.ΔLL)
    data.ΔLL .*= data.mlL
end

function compute_gradient!(optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}}

    initialize!(optimizer)
    sample = optimizer.workspace.sample

    sample = Metropolis_burn_in!(optimizer)

    for _ in 1:optimizer.sampler.N_MC
        #Generate sample:
        sample, acc = Metropolis_sweep_left!(sample, optimizer)
        optimizer.optimizer_cache.acceptance += acc/(optimizer.params.N*optimizer.sampler.N_MC)

        #Compute local estimators:
        update!(optimizer, sample) 

        #Update metric tensor:
        update_SR!(optimizer)
    end
end

function optimize!(optimizer::SR{T}, δ::Float64) where {T<:Complex{<:AbstractFloat}}

    finalize!(optimizer)

    reconfigure!(optimizer)

    ∇ = optimizer.optimizer_cache.∇
    ∇ ./= maximum(abs.(∇))

    new_A = similar(optimizer.A)
    new_A = optimizer.A - δ*∇
    optimizer.A = new_A
    optimizer.A = normalize_MPO!(optimizer.params, optimizer.A)
end

function MPI_mean!(optimizer::SR{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}

    par_data = optimizer.optimizer_cache

    MPI.Allreduce!(par_data.L∂L, +, mpi_cache.comm)
    MPI.Allreduce!(par_data.ΔLL, +, mpi_cache.comm)
    MPI.Allreduce!(par_data.S, +, mpi_cache.comm)
    MPI.Allreduce!(par_data.avg_G, +, mpi_cache.comm)
    #MPI.Allreduce!(par_data.acceptance, +, mpi_cache.comm)

    mlL = [par_data.mlL]
    MPI.Reduce!(mlL, +, mpi_cache.comm, root=0)

    acceptance = [par_data.acceptance]
    MPI.Reduce!(acceptance, +, mpi_cache.comm, root=0)

    if mpi_cache.rank == 0
        par_data.mlL = mlL[1]/mpi_cache.nworkers
        par_data.L∂L ./= mpi_cache.nworkers
        par_data.ΔLL ./= mpi_cache.nworkers
        par_data.S ./= mpi_cache.nworkers
        par_data.avg_G ./= mpi_cache.nworkers
        par_data.acceptance = acceptance[1]/mpi_cache.nworkers
    end
end