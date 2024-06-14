export Exact, optimize!, compute_gradient!


mutable struct ExactCache{T} <: OptimizerCache

    #Ensemble averages:
    L∂L::Array{T,3}
    ΔLL::Array{T,3}

    #Sums:
    Z::T
    mlL::T   #mean local Lindbladian

    #Gradient:
    ∇::Array{T,3}
end

function ExactCache(A::Array{T,3}, params::Parameters) where {T<:Complex{<:AbstractFloat}} 

    exact=ExactCache(
        zeros(T,params.χ,params.χ,4),
        zeros(T,params.χ,params.χ,4),
        convert(T,0),
        convert(T,0),
        zeros(T,params.χ,params.χ,4)
    )  
    return exact
end

mutable struct Exactl1{T<:Complex{<:AbstractFloat}} <: Exact{T}

    #Basis:
    basis::Basis

    #MPO:
    A::Array{T,3}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::ExactCache{T}

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

Base.display(optimizer::Exactl1) = begin
    println("\nOptimizer:")
    println("method\t\tExact-l1")
    println("ising_op\t", optimizer.ising_op)
    println("dephasing_op\t", optimizer.dephasing_op)
    println("l1\t\t",optimizer.l1)
end

#Constructor:
function Exact(sampler::MetropolisSampler, A::Array{T,3}, l1::Matrix{T}, params::Parameters, ising_op::String="Ising", dephasing_op::String="Local") where {T<:Complex{<:AbstractFloat}} 

    basis::Basis=generate_bit_basis(params.N)
    if ising_op=="Ising"
        #optimizer = Exactl1(A, sampler, ExactCache(A, params), l1, Ising(), params, set_workspace(A, params))
        if dephasing_op=="Local"
            optimizer = Exactl1(basis, A, sampler, ExactCache(A, params), l1, Ising(), LocalDephasing(), params, set_workspace(A, params))
        elseif dephasing_op=="Collective"
            optimizer = Exactl1(basis, A, sampler, ExactCache(A, params), l1, Ising(), CollectiveDephasing(), params, set_workspace(A, params))
        else
            error("Unrecognized dephasing operator")
        end
    elseif ising_op=="LongRangeIsing" || ising_op=="LRIsing" || ising_op=="Long Range Ising"
        @assert params.α>=0
        optimizer = Exactl1(basis, A, sampler, ExactCache(A, params), l1, LongRangeIsing(params), LocalDephasing(), params, set_workspace(A, params))
    else
        error("Unrecognized Ising interaction")
    end
    return optimizer
end

mutable struct Exactl2{T<:Complex{<:AbstractFloat}} <: Exact{T}

    #Basis:
    basis::Basis

    #MPO:
    A::Array{T,3}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::ExactCache{T}

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

Base.display(optimizer::Exactl2) = begin
    println("\nOptimizer:")
    println("method\t\tExact-l2")
    println("ising_op\t", optimizer.ising_op)
    println("dephasing_op\t", optimizer.dephasing_op)
    println("l1\t\t",optimizer.l1)
    println("l2\t\t",optimizer.l2)
end

#Constructor:
function Exact(sampler::MetropolisSampler, A::Array{T,3}, l1::Matrix{T}, l2::Matrix{T}, params::Parameters, ising_op::String="Ising") where {T<:Complex{<:AbstractFloat}} 

    basis::Basis=generate_bit_basis(params.N)
    if ising_op=="Ising"
        optimizer = Exactl2(basis, A, sampler, ExactCache(A, params), l1, l2, Ising(), params, set_workspace(A, params))
    elseif ising_op=="LongRangeIsing" || ising_op=="LRIsing" || ising_op=="Long Range Ising"
        @assert params.α>=0
        optimizer = Exactl2(basis, A, sampler, ExactCache(A, params), l1, l2, LongRangeIsing(params), params, set_workspace(A, params))
    else
        error("Unrecognized Ising interaction")
    end    
    return optimizer
end

function initialize!(optimizer::Exact{T}) where {T<:Complex{<:AbstractFloat}}

    optimizer.optimizer_cache = ExactCache(optimizer.A, optimizer.params)
    optimizer.workspace = set_workspace(optimizer.A, optimizer.params)
end

function sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::Exactl1{T}) where {T<:Complex{<:AbstractFloat}} 

    params = optimizer.params
    sub_sample = optimizer.workspace.sub_sample
    sub_sample = Projector(sample)

    local_L::T = 0
    local_∇L::Array{T,3} = zeros(T,params.χ,params.χ,4)

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        local_L, local_∇L = one_body_Lindblad_term!(local_L, local_∇L, sample, sub_sample, j, optimizer)
    end

    local_L /= ρ_sample
    local_∇L ./= ρ_sample

    return local_L, local_∇L
end

function sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::Exactl2{T}) where {T<:Complex{<:AbstractFloat}} 

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

function update!(optimizer::Exact{T}, sample::Projector) where {T<:Complex{<:AbstractFloat}} #... the ensemble averages etc.

    params = optimizer.params
    A = optimizer.A
    data = optimizer.optimizer_cache
    ws = optimizer.workspace

    #Initialize auxiliary arrays:
    local_L::T = 0
    local_∇L::Array{T,3} = zeros(T,params.χ,params.χ,4)
    l_int::T = 0
    ws.local_∇L_diagonal_coeff = 0

    ws.L_set = L_MPO_products!(ws.L_set, sample, A, params, workspace)
    ws.R_set = R_MPO_products!(ws.R_set, sample, A, params, workspace)

    ρ_sample::T = tr(L_set[params.N+1])
    p_sample::T = ρ_sample*conj(ρ_sample)
    data.Z += p_sample

    ws.Δ = ∂MPO(sample, L_set, R_set, params, ws)./ρ_sample

    #Sweep lattice:
    local_L, local_∇L = sweep_Lindblad!(sample, ρ_sample, optimizer)

    #Add in diagonal part of the local derivative:
    local_∇L .+= ws.local_∇L_diagonal_coeff.*ws.Δ

    #Add in interaction terms:
    l_int = Ising_interaction_energy(optimizer.ising_op, sample, optimizer)
    local_L += l_int
    local_∇L .+= l_int*ws.Δ

    #Update L∂L* ensemble average:
    data.L∂L .+= p_sample*local_L*conj(local_∇L)

    #Update ΔLL ensemble average:
    data.ΔLL .+= p_sample*ws.Δ

    #Mean local Lindbladian:
    data.mlL += real(p_sample*local_L*conj(local_L))
end

function finalize!(optimizer::Exact{T}) where {T<:Complex{<:AbstractFloat}}

    data = optimizer.optimizer_cache

    data.mlL /= data.Z
    data.ΔLL .= conj.(data.ΔLL)
    data.ΔLL .*= data.mlL
    data.∇ = (data.L∂L-data.ΔLL)/data.Z
end

function compute_gradient!(optimizer::Exact{T}) where {T<:Complex{<:AbstractFloat}}

    initialize!(optimizer)

    for k in 1:optimizer.params.dim_H
        for l in 1:optimizer.params.dim_H
            sample = Projector(optimizer.basis[k], optimizer.basis[l])
            update!(optimizer, sample) 
        end
    end
end

function optimize!(optimizer::Exact{T}, δ::Float64) where {T<:Complex{<:AbstractFloat}}

    finalize!(optimizer)

    ∇ = optimizer.optimizer_cache.∇
    ∇ ./= maximum(abs.(∇))

    new_A = similar(optimizer.A)
    new_A = optimizer.A - δ*∇
    optimizer.A = new_A
    optimizer.A = normalize_MPO!(optimizer.params, optimizer.A)
end

function MPI_mean!(optimizer::Exact{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}

    par_data = optimizer.optimizer_cache

    MPI.Allreduce!(par_data.L∂L, +, mpi_cache.comm)
    MPI.Allreduce!(par_data.ΔLL, +, mpi_cache.comm)

    mlL = [par_data.mlL]
    MPI.Reduce!(mlL, +, mpi_cache.comm, root=0)

    if mpi_cache.rank == 0
        par_data.mlL = mlL[1]/mpi_cache.nworkers
        par_data.L∂L ./= mpi_cache.nworkers
        par_data.ΔLL ./= mpi_cache.nworkers
    end

end