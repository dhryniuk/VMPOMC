export SGD, Optimize!


mutable struct SGDCache{T} <: StochasticCache
    #Ensemble averages:
    L∂L::Array{T,3}
    ΔLL::Array{T,3}

    #Sums:
    mlL::T
    acceptance::UInt64

    #Gradient:
    ∇::Array{T,3}
end

function SGDCache(A::Array{T,3},params::parameters) where {T<:Complex{<:AbstractFloat}} 
    cache=SGDCache(
        zeros(T,params.χ,params.χ,4),
        zeros(T,params.χ,params.χ,4),
        convert(T,0),
        convert(UInt64,0),
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
    optimizer_cache::SGDCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}

    #Eigen operations:
    eigen_ops::EigenOperations

    #Parameters:
    params::parameters

    #Workspace:
    workspace::workspace{T}#Union{workspace,Nothing}

end

#Constructor:
function SGD(sampler::MetropolisSampler, l1::Matrix{<:Complex{<:AbstractFloat}}, params::parameters, eigen_op::String="Ising")
    A = rand(ComplexF64,params.χ,params.χ,4)
    if eigen_op=="Ising"
        optimizer = SGDl1(A, sampler, SGDCache(A, params), l1, Ising(), params, set_workspace(A, params))
    elseif eigen_op=="LongRangeIsing" || eigen_op=="LRIsing" || eigen_op=="Long Range Ising"
        @assert params.α>0
        optimizer = SGDl1(A, sampler, SGDCache(A, params), l1, LongRangeIsing(params), params, set_workspace(A, params))
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
    optimizer_cache::SGDCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}

    #2-local Lindbladian:
    l2::Matrix{T}

    #Eigen operations:
    eigen_ops::EigenOperations

    #Parameters:
    params::parameters

    #Workspace:
    workspace::workspace{T}#Union{workspace,Nothing}

end

#Constructor:
function SGD(sampler::MetropolisSampler, l1::Matrix{<:Complex{<:AbstractFloat}}, l2::Matrix{<:Complex{<:AbstractFloat}}, params::parameters, eigen_op::String="Ising")
    A = rand(ComplexF64,params.χ,params.χ,4)
    if eigen_op=="Ising"
        optimizer = SGDl2(A, sampler, SGDCache(A, params), l1, l2, Ising(), params, set_workspace(A, params))
    elseif eigen_op=="LongRangeIsing" || eigen_op=="LRIsing" || eigen_op=="Long Range Ising"
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

function Ising_interaction_energy(eigen_ops::Ising, sample::projector, optimizer::SGD{T}) where {T<:Complex{<:AbstractFloat}} 

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

function Ising_interaction_energy(eigen_ops::LongRangeIsing, sample::projector, optimizer::SGD{T}) where {T<:Complex{<:AbstractFloat}} 

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

function SweepLindblad!(sample::projector, ρ_sample::T, optimizer::SGDl1{T}, local_L::T, local_∇L::Array{T,3}) where {T<:Complex{<:AbstractFloat}} 

    params=optimizer.params
    micro_sample = optimizer.workspace.micro_sample
    micro_sample = projector(sample)

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        lL, l∇L = one_body_Lindblad_term(sample,micro_sample,j,optimizer)
        local_L += lL
        local_∇L += l∇L
    end

    local_L /=ρ_sample
    local_∇L/=ρ_sample

    return local_L, local_∇L
end

function SweepLindblad!(sample::projector, ρ_sample::T, optimizer::SGDl2{T}, local_L::T, local_∇L::Array{T,3}) where {T<:Complex{<:AbstractFloat}} 

    params=optimizer.params
    micro_sample = optimizer.workspace.micro_sample
    micro_sample = projector(sample)

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        lL, l∇L = one_body_Lindblad_term(sample,micro_sample,j,optimizer)
        local_L += lL
        local_∇L += l∇L
    end
    for j::UInt8 in 1:params.N-1
        lL, l∇L = two_body_Lindblad_term(sample,micro_sample,j,optimizer)
        local_L += lL
        local_∇L += l∇L
    end
    if params.N>2
        lL, l∇L = boundary_two_body_Lindblad_term(sample,micro_sample,optimizer)
        local_L += lL
        local_∇L += l∇L
    end

    local_L /=ρ_sample
    local_∇L/=ρ_sample

    return local_L, local_∇L
end

function Update!(optimizer::Stochastic{T}, sample::projector) where {T<:Complex{<:AbstractFloat}} #... the ensemble averages etc.

    params=optimizer.params
    A=optimizer.A
    #l1=optimizer.l1
    data=optimizer.optimizer_cache
    cache = optimizer.workspace

    #Initialize auxiliary arrays:
    local_L::T = 0
    local_∇L::Array{T,3} = zeros(T,params.χ,params.χ,4)
    l_int::T = 0
    cache.local_∇L_diagonal_coeff = 0

    ρ_sample::T = tr(cache.R_set[params.N+1])
    cache.L_set = L_MPO_strings!(cache.L_set, sample,A,params,cache)
    cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache)./ρ_sample

    #Sweep lattice:
    local_L, local_∇L = SweepLindblad!(sample, ρ_sample, optimizer, local_L, local_∇L)

    #Add in diagonal part of the local derivative:
    local_∇L.+=cache.local_∇L_diagonal_coeff.*cache.Δ

    #Add in Ising interaction terms:
    l_int = Ising_interaction_energy(optimizer.eigen_ops, sample, optimizer)
    #println(l_int); error()
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

    # Initialize sample and L_set for that sample:
    sample = MPO_Metropolis_burn_in(optimizer)

    for _ in 1:optimizer.sampler.N_MC

        #Generate sample:
        sample, acc = Mono_Metropolis_sweep_left(sample, optimizer)
        optimizer.optimizer_cache.acceptance += acc

        Update!(optimizer, sample) 
    end
    Finalize!(optimizer)
end

function Optimize!(optimizer::SGD{T}, δ) where {T<:Complex{<:AbstractFloat}}

    ComputeGradient!(optimizer)

    ∇ = optimizer.optimizer_cache.∇
    ∇./=maximum(abs.(∇))

    new_A = similar(optimizer.A)
    new_A = optimizer.A - δ*∇
    optimizer.A = new_A
    optimizer.A = normalize_MPO!(optimizer.params, optimizer.A)
end

































#### TO REWORK LATER:





function reweighted_SGD_MPO_gradient(β::Float64, A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, params::parameters)
    # Define ensemble averages:
    L∇L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    ΔLL::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    mean_local_Lindbladian::eltype(A) = 0
    Z::Float64 = 0

    # Preallocate cache:
    cache = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample = MPO_Metropolis_burn_in(A, params, cache)
    acceptance::UInt64=0

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
        p_sample=(ρ_sample*conj(ρ_sample))^(1-β)
        Z+=p_sample
        cache.L_set = L_MPO_strings!(cache.L_set, sample,A,params,cache)
        cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache)./ρ_sample

        #L∇L*:
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
        #l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)
        l_int = Ising_interaction_energy(optimizer.eigen_ops, sample, optimizer)
        local_L +=l_int
        local_∇L+=l_int*cache.Δ

        L∇L+=p_sample*local_L*conj(local_∇L)

        #ΔLL:
        ΔLL.+=cache.Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += p_sample*local_L*conj(local_L)

    end
    mean_local_Lindbladian/=Z
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate
    ΔLL.*=real(mean_local_Lindbladian)
    return (L∇L-ΔLL)/Z, real(mean_local_Lindbladian), acceptance/(N_MC*params.N)
end



### DISTRIBUTED VERSION:

function one_worker_SGD_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, params::parameters)
    
    # Define ensemble averages:
    L∂L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    ΔLL::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    mean_local_Lindbladian::eltype(A) = 0

    # Preallocate cache:
    cache = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample = MPO_Metropolis_burn_in(A, params, cache)
    acceptance::UInt64=0

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
    end
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate

    return [L∂L, ΔLL, mean_local_Lindbladian, acceptance]
end

function distributed_SGD_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, params::parameters)
    #output = [L∇L, ΔLL, mean_local_Lindbladian]

    #perform reduction:
    output = @distributed (+) for i=1:nworkers()
        one_worker_SGD_MPO_gradient(A, l1, N_MC, params)
    end

    L∂L=output[1]
    ΔLL=output[2]
    mean_local_Lindbladian=output[3]
    acc=output[4]

    mean_local_Lindbladian/=(N_MC*nworkers())
    ΔLL*=mean_local_Lindbladian

    acc/=(N_MC*nworkers()*params.N)

    return (L∂L-ΔLL)/(N_MC*nworkers()), real(mean_local_Lindbladian), acc
end