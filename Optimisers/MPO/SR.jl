export SR, Optimize!, ComputeGradient!, MPI_mean!, MPI_normalize!


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
        0.0,#convert(UInt64,0),
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

    #Eigen operations:
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
    println("eigen_op\t", optimizer.ising_op)
    println("eigen_op\t", optimizer.dephasing_op)
    println("l1\t\t",optimizer.l1)
end


#Constructor:
function SR(sampler::MetropolisSampler, A::Array{T,3}, l1::Matrix{T}, ϵ::Float64, params::Parameters, ising_op::String="Ising", dephasing_op::String="Local") where {T<:Complex{<:AbstractFloat}} 
    #A = rand(ComplexF64,params.χ,params.χ,4)
    if ising_op=="Ising"
        if dephasing_op=="Local"
            optimizer = SRl1(A, sampler, SRCache(A, params), l1, Ising(), LocalDephasing(), params, ϵ, set_workspace(A, params))
        elseif dephasing_op=="Collective"
            optimizer = SRl1(A, sampler, SRCache(A, params), l1, Ising(), CollectiveDephasing(), params, ϵ, set_workspace(A, params))
        else
            error("Unrecognized eigen-operation")
        end
    elseif ising_op=="LongRangeIsing" || ising_op=="LRIsing" || ising_op=="Long Range Ising"
        @assert params.α>0
        optimizer = SRl1(A, sampler, SRCache(A, params), l1, LongRangeIsing(params), LocalDephasing(), params, ϵ, set_workspace(A, params))
    else
        error("Unrecognized eigen-operation")
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

    #Eigen operations:
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
    #A = rand(ComplexF64,params.χ,params.χ,4)
    if ising_op=="Ising"
        if dephasing_op=="Local"
            optimizer = SRl2(A, sampler, SRCache(A, params), l1, l2, Ising(), LocalDephasing(), params, ϵ, set_workspace(A, params))
        elseif dephasing_op=="Collective"
            optimizer = SRl2(A, sampler, SRCache(A, params), l1, l2, Ising(), CollectiveDephasing(), params, ϵ, set_workspace(A, params))
        else
            error("Unrecognized eigen-operation")
        end
    elseif ising_op=="LongRangeIsing" || ising_op=="LRIsing" || ising_op=="Long Range Ising"
        @assert params.α>0
        optimizer = SRl2(A, sampler, SRCache(A, params), l1, l2, LongRangeIsing(params), LocalDephasing(), params, ϵ, set_workspace(A, params))
    else
        error("Unrecognized eigen-operation")
    end
    return optimizer
end

function Initialize!(optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}}
    optimizer.optimizer_cache = SRCache(optimizer.A, optimizer.params)
    optimizer.workspace = set_workspace(optimizer.A, optimizer.params)
end

function Ising_interaction_energy(eigen_ops::Ising, sample::Projector, optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}} 

    A = optimizer.A
    params = optimizer.params

    l_int::T=0
    for j::UInt8 in 1:params.N-1
        l_int_ket = (2*sample.ket[j]-1)*(2*sample.ket[j+1]-1)
        l_int_bra = (2*sample.bra[j]-1)*(2*sample.bra[j+1]-1)
        l_int += l_int_ket-l_int_bra
    end
    l_int_ket = (2*sample.ket[params.N]-1)*(2*sample.ket[1]-1)
    l_int_bra = (2*sample.bra[params.N]-1)*(2*sample.bra[1]-1)
    l_int += l_int_ket-l_int_bra
    return 1.0im*params.J*l_int
    #return -1.0im*params.J*l_int
end

function Ising_interaction_energy(eigen_ops::LongRangeIsing, sample::Projector, optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}} 

    A = optimizer.A
    params = optimizer.params

    l_int_ket::T = 0.0
    l_int_bra::T = 0.0
    l_int::T = 0.0
    for i::Int16 in 1:params.N-1
        for j::Int16 in i+1:params.N
            l_int_ket = (2*sample.ket[i]-1)*(2*sample.ket[j]-1)
            l_int_bra = (2*sample.bra[i]-1)*(2*sample.bra[j]-1)
            dist = min(abs(i-j), abs(params.N+i-j))^eigen_ops.α
            l_int += (l_int_ket-l_int_bra)/dist
        end
    end
    return 1.0im*params.J*l_int/eigen_ops.Kac_norm
end

function OLDDephasing_term(dephasing_op::LocalDephasing, sample::Projector, optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}} 

    params = optimizer.params

    l::T=0
    for j::UInt8 in 1:params.N
        l_ket = (2*sample.ket[j]-1)
        l_bra = (2*sample.bra[j]-1)
        l += (l_ket*l_bra-1)
    end
    return params.γ_d*l
end

function Dephasing_term(dephasing_op::LocalDephasing, sample::Projector, optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}} 

    params = optimizer.params

    l::T=0
    for j::UInt8 in 1:params.N
        l_ket = (2*sample.ket[j]-1)
        l_bra = (2*sample.bra[j]-1)
        l += (l_ket*l_bra-1)
    end
    return params.γ_d*l
end


function OLDDephasing_term(dephasing_op::CollectiveDephasing, sample::Projector, optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}} 

    params = optimizer.params

    l::T=1
    for j::UInt8 in 1:params.N
        l_ket = (2*sample.ket[j]-1)
        l_bra = (2*sample.bra[j]-1)
        l *= l_ket*l_bra
    end
    l-=1
    return params.γ_d*l
end

function Dephasing_term(dephasing_op::CollectiveDephasing, sample::Projector, optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}} 

    params = optimizer.params

    l_ket::T=0
    l_bra::T=0
    for j::UInt8 in 1:params.N
        l_ket += (2*sample.ket[j]-1)
        l_bra += (2*sample.bra[j]-1)
        #l += (l_ket*l_bra-1)
    end
    return params.γ_d*(l_ket*l_bra-0.5*(l_ket^2+l_bra^2))
end

#### REPLACE WITH HOLY TRAITS ---

function SweepLindblad!(sample::Projector, ρ_sample::T, optimizer::SRl1{T}) where {T<:Complex{<:AbstractFloat}} 

    params = optimizer.params
    micro_sample = optimizer.workspace.micro_sample
    micro_sample = Projector(sample)

    temp_local_L::T = 0
    temp_local_∇L::Array{T,3} = zeros(T,params.χ,params.χ,4)
    #temp_local_L = optimizer.workspace.temp_local_L
    #temp_local_L = 0.0+0.0im
    #temp_local_∇L = optimizer.workspace.temp_local_∇L
    #temp_local_∇L = zeros(T,params.χ,params.χ,4)

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        temp_local_L, temp_local_∇L = one_body_Lindblad_term!(temp_local_L, temp_local_∇L, sample, micro_sample, j, optimizer)
    end

    temp_local_L  /= ρ_sample
    temp_local_∇L./= ρ_sample

    return temp_local_L, temp_local_∇L
end

function SweepLindblad!(sample::Projector, ρ_sample::T, optimizer::SRl2{T}) where {T<:Complex{<:AbstractFloat}} 

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

#### ---.

function UpdateSR!(optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}}
    S::Array{T,2} = optimizer.optimizer_cache.S
    avg_G::Vector{T} = optimizer.optimizer_cache.avg_G
    params::Parameters = optimizer.params
    workspace = optimizer.workspace
    
    G::Vector{T} = reshape(workspace.Δ,4*params.χ^2)
    conj_G = conj(G)
    avg_G.+= G
    mul!(workspace.plus_S,conj_G,transpose(G))
    S.+=workspace.plus_S   #use when l1 is unconjugated
    #S.+=conj(cache.plus_S)
end

#function Reconfigure!(data::SRCache{T}, N_MC::UInt64, ϵ::AbstractFloat, params::Parameters) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor
function Reconfigure!(optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor

    data = optimizer.optimizer_cache
    N_MC = optimizer.sampler.N_MC
    ϵ = optimizer.ϵ
    params = optimizer.params

    #Compute metric tensor:
    data.S./=N_MC
    data.avg_G./=N_MC
    conj_avg_G = conj(data.avg_G)
    data.S-=data.avg_G*transpose(conj_avg_G) ##THIS IS CORRECT
    #S-=real.(avg_G)*real.(transpose(conj_avg_G))

    #Regularize the metric tensor:
    data.S+=ϵ*Matrix{Int}(I, params.χ*params.χ*4, params.χ*params.χ*4)

    #Reconfigure gradient:
    grad::Array{eltype(data.S),3} = (data.L∂L-data.ΔLL)/N_MC
    flat_grad::Vector{eltype(data.S)} = reshape(grad,4*params.χ^2)
    flat_grad = inv(data.S)*flat_grad
    data.∇ = reshape(flat_grad,params.χ,params.χ,4)
end

function Finalize!(optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}}
    N_MC = optimizer.sampler.N_MC
    data = optimizer.optimizer_cache

    data.mlL /= N_MC
    data.ΔLL .= conj.(data.ΔLL) #remember to take the complex conjugate
    data.ΔLL .*= data.mlL
    #Reconfigure!(data,N_MC,optimizer.ϵ,optimizer.params)

    #optimizer.optimizer_cache.acceptance/=(optimizer.params.N*N_MC)
end

function ComputeGradient!(optimizer::SR{T}) where {T<:Complex{<:AbstractFloat}}

    Initialize!(optimizer)
    sample = optimizer.workspace.sample

    sample = MPO_Metropolis_burn_in(optimizer)

    for _ in 1:optimizer.sampler.N_MC

        #Generate sample:
        sample, acc = Mono_Metropolis_sweep_left(sample, optimizer)
        optimizer.optimizer_cache.acceptance += acc/(optimizer.params.N*optimizer.sampler.N_MC)

        #Compute local estimators:
        Update!(optimizer, sample) 

        #Update metric tensor:
        UpdateSR!(optimizer)
    end
    #Finalize!(optimizer)

    #Reconfigure!(optimizer.optimizer_cache,optimizer.sampler.N_MC,optimizer.ϵ,optimizer.params)
end

function Optimize!(optimizer::SR{T}, δ::Float64) where {T<:Complex{<:AbstractFloat}}

    #ComputeGradient!(optimizer)

    Finalize!(optimizer)

    Reconfigure!(optimizer)

    ∇  = optimizer.optimizer_cache.∇
    ∇./= maximum(abs.(∇))

    new_A = similar(optimizer.A)
    new_A = optimizer.A - δ*∇
    optimizer.A = new_A
    optimizer.A = normalize_MPO!(optimizer.params, optimizer.A)
end

"""
function MPI_mean!(optimizer::SR{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    par_cache = optimizer.optimizer_cache

    workers_sum!(par_cache.L∂L,mpi_cache.comm)
    workers_sum!(par_cache.ΔLL,mpi_cache.comm)
    workers_sum!(par_cache.mlL,par_cache.mlL,mpi_cache.comm)
    workers_sum!(par_cache.acceptance,par_cache.acceptance,mpi_cache.comm)
    workers_sum!(par_cache.S,mpi_cache.comm)
    workers_sum!(par_cache.avg_G,mpi_cache.comm)
    #ignore ∇???
end

function MPI_normalize!(optimizer::SR{T}, nworkers) where {T<:Complex{<:AbstractFloat}}
    par_cache = optimizer.optimizer_cache

    #par_cache.L∂L./=(nworkers)   #why not this too?
    #par_cache.ΔLL./=(nworkers)   #why not this too?
    par_cache.mlL/=(nworkers)
    par_cache.S./=(nworkers)
    par_cache.avg_G./=(nworkers)
    #par_cache.acceptance/=(nworkers)
end

"""
function MPI_mean!(optimizer::SR{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    par_cache = optimizer.optimizer_cache

    MPI.Allreduce!(par_cache.L∂L, +, mpi_cache.comm)
    MPI.Allreduce!(par_cache.ΔLL, +, mpi_cache.comm)
    MPI.Allreduce!(par_cache.S, +, mpi_cache.comm)
    MPI.Allreduce!(par_cache.avg_G, +, mpi_cache.comm)

    mlL = [par_cache.mlL]
    MPI.Reduce!(mlL, +, mpi_cache.comm, root=0)
    if mpi_cache.rank == 0
        par_cache.mlL = mlL[1]/mpi_cache.nworkers
        par_cache.L∂L./=mpi_cache.nworkers
        par_cache.ΔLL./=mpi_cache.nworkers
        par_cache.S./=mpi_cache.nworkers
        par_cache.avg_G./=mpi_cache.nworkers
    end

end

































#### NEED TO CLEAN UP LATER:



"""

mutable struct SR{T} <: Stochastic
    #Ensemble averages:
    L∂L::Array{T,3}
    ΔLL::Array{T,3}

    #Sums:
    mlL::T
    acceptance::UInt64

    #Gradient:
    ∇::Array{T,3}

    # Metric tensor:
    S::Array{T,2}
    avg_G::Array{T}
    #conj_avg_G::Array{T}
end

function set_SR(A,params)
    sr=SR(
        zeros(eltype(A),params.χ,params.χ,4),
        zeros(eltype(A),params.χ,params.χ,4),
        convert(eltype(A),0),
        convert(UInt64,0),
        zeros(eltype(A),params.χ,params.χ,4),
        zeros(eltype(A),4*params.χ^2,4*params.χ^2),
        #zeros(eltype(A),4*params.χ^2),
        zeros(eltype(A),4*params.χ^2)
    )  
    return sr
end

function MPO_flatten_index(i::UInt8,j::UInt8,s::UInt8,params::Parameters)
    return i+params.χ*(j-1)+params.χ^2*(s-1)
end

function sample_update_SR!(data::SR, params::Parameters, cache::Workspace)
    G = reshape(cache.Δ,4*params.χ^2)
    conj_G = conj(G)
    data.avg_G.+= G
    mul!(cache.plus_S,conj_G,transpose(G))
    data.S.+=cache.plus_S   #use when l1 is unconjugated
    #S.+=conj(cache.plus_S)
    #return data
end

"""

function ∇!(data::SR, N_MC::UInt64, ϵ::AbstractFloat, params::Parameters)

    #Compute metric tensor:
    data.S./=N_MC
    data.avg_G./=N_MC
    conj_avg_G = conj(data.avg_G)
    data.S-=data.avg_G*transpose(conj_avg_G) ##THIS IS CORRECT
    #S-=real.(avg_G)*real.(transpose(conj_avg_G))

    #Regularize the metric tensor:
    data.S+=ϵ*Matrix{Int}(I, params.χ*params.χ*4, params.χ*params.χ*4)

    #Reconfigure gradient:
    grad::Array{eltype(data.S),3} = (data.L∂L-data.ΔLL)/N_MC
    flat_grad::Vector{eltype(data.S)} = reshape(grad,4*params.χ^2)
    flat_grad = inv(data.S)*flat_grad
    data.∇ = reshape(flat_grad,params.χ,params.χ,4)
    #return grad
end

function SR_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, sampler::MetropolisSampler, ϵ::AbstractFloat, params::Parameters)
    N_MC = sampler.N_MC
    
    # Preallocate data cache:
    data = set_SR(A,params)

    # Preallocate auxiliary arrays:
    cache = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample::Projector = MPO_Metropolis_burn_in(A, params, cache)

    for _ in 1:N_MC

        #Generate sample:
        sample, acc = Mono_Metropolis_sweep_left(sample, A, params, cache)
        data.acceptance+=acc

        #Update data:
        update!(data, sample, A, l1, params, cache)

        #Update metric tensor:
        sample_update_SR!(data, params, cache)

    end

    #Finalize:
    data.mlL/=N_MC
    data.ΔLL.=conj.(data.ΔLL) #remember to take the complex conjugate
    data.ΔLL*=real(data.mlL)
    ∇!(data,N_MC,ϵ,params)

    return data.∇, real(data.mlL), data.acceptance/(N_MC*params.N)
end

function reweighted_sample_update_SR(p_sample::Float64, S::Array{<:Complex{<:AbstractFloat},2}, avg_G::Array{<:Complex{<:AbstractFloat}}, params::Parameters, cache::Workspace)
    G = reshape(cache.Δ,4*params.χ^2)
    conj_G = conj(G)
    avg_G.+= p_sample*G
    mul!(cache.plus_S,conj_G,transpose(G))
    S.+=cache.plus_S
    return S, avg_G
end

function reweighted_apply_SR(S::Array{<:Complex{<:AbstractFloat},2}, avg_G::Array{<:Complex{<:AbstractFloat}}, Z::Float64, ϵ::AbstractFloat, 
    L∇L::Array{<:Complex{<:AbstractFloat},3}, ΔLL::Array{<:Complex{<:AbstractFloat},3}, params::Parameters)

    #Compute metric tensor:
    S./=Z
    avg_G./=Z
    conj_avg_G = conj(avg_G)
    S-=avg_G*transpose(conj_avg_G)

    #Regularize the metric tensor:
    S+=ϵ*Matrix{Int}(I, params.χ*params.χ*4, params.χ*params.χ*4)

    #Find SR'd gradient:
    grad::Array{eltype(S),3} = (L∇L-ΔLL)/Z
    flat_grad::Vector{eltype(S)} = reshape(grad,4*params.χ^2)
    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad,params.χ,params.χ,4)

    return grad
end

function reweighted_SR_MPO_gradient(β::Float64, A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::Parameters)
    
    # Define ensemble averages:
    L∂L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    ΔLL::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    mean_local_Lindbladian::eltype(A) = 0
    Z::Float64 = 0

    # Preallocate cache:
    cache = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample = MPO_Metropolis_burn_in(A, params, cache)
    acceptance::UInt64=0

    # Metric tensor auxiliary arrays:
    S::Array{eltype(A),2} = zeros(eltype(A),4*params.χ^2,4*params.χ^2)
    Left::Array{eltype(A)} = zeros(eltype(A),4*params.χ^2)

    for _ in 1:N_MC

        #Initialize auxiliary arrays:
        local_L::eltype(A) = 0
        local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
        l_int::eltype(A) = 0
        cache.local_∇L_diagonal_coeff = 0

        #Generate sample:
        sample, acc = reweighted_Mono_Metropolis_sweep_left(β, sample, A, params, cache)
        acceptance+=acc

        ρ_sample::eltype(A) = tr(cache.R_set[params.N+1])
        p_sample::Float64=(ρ_sample*conj(ρ_sample))^(1-β)
        Z+=p_sample
        cache.L_set = L_MPO_strings!(cache.L_set, sample,A,params,cache)
        cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache)./ρ_sample

        #Calculate L∂L*:
        for j::UInt8 in 1:params.N
            #1-local part:
            lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,params,cache)
            local_L += lL
            local_∇L += l∇L
        end

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        #Add in diagonal part of the local derivative:
        local_∇L.+=cache.local_∇L_diagonal_coeff.*cache.Δ

        #Add in interaction terms:
        l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)
        local_L +=l_int
        local_∇L+=l_int*cache.Δ

        L∂L+=p_sample*local_L*conj(local_∇L)

        #ΔLL:
        ΔLL.+=cache.Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += p_sample*local_L*conj(local_L)

        #Update metric tensor:
        S, Left = reweighted_sample_update_SR(p_sample, S, Left, params, cache)
    end
    mean_local_Lindbladian/=Z
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate
    ΔLL.*=real(mean_local_Lindbladian)

    grad = reweighted_apply_SR(S,Left,Z,ϵ,L∂L,ΔLL,params)

    return grad, real(mean_local_Lindbladian), acceptance/(N_MC*params.N)
end



### DISTRIBUTED VERSION:

function one_worker_SR_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::Parameters)
    
    # Define ensemble averages:
    L∂L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    ΔLL::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    mean_local_Lindbladian::eltype(A) = 0

    # Preallocate cache:
    cache = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample = MPO_Metropolis_burn_in(A, params, cache)
    acceptance::UInt64=0

    # Metric tensor auxiliary arrays:
    S::Array{eltype(A),2} = zeros(eltype(A),4*params.χ^2,4*params.χ^2)
    Left::Array{eltype(A)} = zeros(eltype(A),4*params.χ^2)

    for _ in 1:N_MC

        #Initialize auxiliary arrays:
        local_L::eltype(A) = 0
        local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
        l_int::eltype(A) = 0
        cache.local_∇L_diagonal_coeff = 0

        #Generate sample:
        sample, acc = Mono_Metropolis_sweep_left(sample, A, params, cache)
        acceptance+=acc

        ρ_sample::eltype(A) = tr(cache.R_set[params.N+1])
        cache.L_set = L_MPO_strings!(cache.L_set, sample,A,params,cache)
        cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache)./ρ_sample

        #Calculate L∂L*:
        for j::UInt8 in 1:params.N
            #1-local part:
            lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,params,cache)
            local_L  += lL
            local_∇L.+= l∇L
        end

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        #Add in diagonal part of the local derivative:
        local_∇L.+=cache.local_∇L_diagonal_coeff.*cache.Δ

        #Add in interaction terms:
        l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)
        local_L +=l_int
        local_∇L+=l_int*cache.Δ

        L∂L+=local_L*conj(local_∇L)

        #ΔLL:
        ΔLL+=cache.Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

        #Update metric tensor:
        S, Left = sample_update_SR(S, Left, params, cache)
    end
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate

    return [L∂L, ΔLL, mean_local_Lindbladian, S, Left, acceptance]
end

function calculate_Kac_norm(d_max, α; offset=0.0) #periodic BCs only!
    N_K = offset
    #for i in 1:convert(Int64,floor(N/2))
    for i in 1:d_max
        N_K+=1/i^α
    end
    return N_K
end

function LR_one_worker_SR_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::Parameters)
    
    if mod(params.N,2)==0
        #error("ONLY ODD NUMBER OF SPINS SUPPORTED ATM")
    end

    # Define ensemble averages:
    L∂L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    ΔLL::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    mean_local_Lindbladian::eltype(A) = 0

    # Preallocate cache:
    cache = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample = MPO_Metropolis_burn_in(A, params, cache)
    acceptance::UInt64=0

    # Metric tensor auxiliary arrays:
    S::Array{eltype(A),2} = zeros(eltype(A),4*params.χ^2,4*params.χ^2)
    Left::Array{eltype(A)} = zeros(eltype(A),4*params.χ^2)

    N_K = calculate_Kac_norm(convert(Int64,floor(params.N/2)), params.α)

    #println("KAC = ", N_K)

    for _ in 1:N_MC

        #Initialize auxiliary arrays:
        local_L::eltype(A) = 0
        local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
        l_int::eltype(A) = 0
        cache.local_∇L_diagonal_coeff = 0

        #Generate sample:
        sample, acc = Mono_Metropolis_sweep_left(sample, A, params, cache)
        acceptance+=acc

        ρ_sample::eltype(A) = tr(cache.R_set[params.N+1])
        cache.L_set = L_MPO_strings!(cache.L_set, sample,A,params,cache)
        cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache)./ρ_sample

        #Calculate L∂L*:
        for j::UInt8 in 1:params.N
            #1-local part:
            lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,params,cache)
            local_L  += lL
            local_∇L.+= l∇L
        end

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        #Add in diagonal part of the local derivative:
        local_∇L.+=cache.local_∇L_diagonal_coeff.*cache.Δ

        #Add in interaction terms:
        #l_int = long_range_interaction(sample, A, params)/N_K
        l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)
        local_L +=l_int
        local_∇L+=l_int*cache.Δ

        L∂L+=local_L*conj(local_∇L)

        #ΔLL:
        ΔLL+=cache.Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

        #Update metric tensor:
        S, Left = sample_update_SR(S, Left, params, cache)
    end
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate

    return [L∂L, ΔLL, mean_local_Lindbladian, S, Left, acceptance]
end

function distributed_SR_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::Parameters)
    #output = [L∇L, ΔLL, mean_local_Lindbladian, S, Left, Right]

    #perform reduction:
    output = @distributed (+) for i=1:nworkers()
        #sample_with_SR_long_range(p, A, l1, N_MC, N_sweeps)
        #one_worker_SR_MPO_gradient(A, l1, convert(Int64,ceil(N_MC/nworkers())), ϵ, params)
        LR_one_worker_SR_MPO_gradient(A, l1, N_MC, ϵ, params)
        #one_worker_SR_MPO_gradient(A, l1, N_MC, ϵ, params)
    end

    L∂L=output[1]
    ΔLL=output[2]
    mean_local_Lindbladian=output[3]
    S=output[4]
    Left=output[5]
    acc=output[6]

    mean_local_Lindbladian/=(N_MC*nworkers())
    ΔLL*=mean_local_Lindbladian

    #Metric tensor:
    S./=(nworkers())
    Left./=(nworkers())

    grad = apply_SR(S,Left,N_MC,ϵ,L∂L,ΔLL,params)

    acc/=(N_MC*nworkers()*params.N)

    return grad, real(mean_local_Lindbladian), acc
end