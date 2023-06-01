abstract type Stochastic <: OptimizerCache end

mutable struct SGD{T} <: Stochastic
    #Ensemble averages:
    L∂L::Array{T,3}
    ΔLL::Array{T,3}

    #Sums:
    mlL::T
    acceptance::UInt64

    #Gradient:
    ∇::Array{T,3}
end

function set_SGD(A,params)
    sgd=SGD(
        zeros(eltype(A),params.χ,params.χ,4),
        zeros(eltype(A),params.χ,params.χ,4),
        convert(eltype(A),0),
        convert(UInt64,0),
        zeros(eltype(A),params.χ,params.χ,4)
    )  
    return sgd
end

function ∇!(data::SGD, N_MC::UInt64)
    data.∇ = (data.L∂L-data.ΔLL)/N_MC
end


function update!(data::Stochastic, sample, A, l1, params::parameters, cache::workspace) #... the ensemble averages etc.
    #Initialize auxiliary arrays:
    local_L::eltype(A) = 0
    local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
    l_int::eltype(A) = 0
    cache.local_∇L_diagonal_coeff = 0

    #Generate sample:
    #sample, acc = Mono_Metropolis_sweep_left(sample, A, params, cache)
    #data.acceptance+=acc

    ρ_sample::eltype(A) = tr(cache.R_set[params.N+1])
    cache.L_set = L_MPO_strings!(cache.L_set, sample,A,params,cache)
    cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache)./ρ_sample

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        #1-local part:
        lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,params,cache)
        local_L += lL
        local_∇L .+= l∇L
    end

    local_L  /=ρ_sample
    local_∇L./=ρ_sample

    #Add in diagonal part of the local derivative:
    local_∇L.+=cache.local_∇L_diagonal_coeff.*cache.Δ

    #Add in Ising interaction terms:
    l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)
    local_L  +=l_int
    local_∇L.+=l_int*cache.Δ

    #Update L∂L* ensemble average:
    data.L∂L.+=local_L*conj(local_∇L)

    #Update ΔLL ensemble average:
    data.ΔLL.+=cache.Δ

    #Mean local Lindbladian:
    data.mlL += local_L*conj(local_L)
end


function SGD_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, sampler::MetropolisSampler, params::parameters)
        
    N_MC=sampler.N_MC

    # Preallocate data cache:
    data = set_SGD(A,params)

    # Preallocate auxiliary work cache:
    cache = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample::projector = MPO_Metropolis_burn_in(A, params, cache)

    for _ in 1:N_MC

        #Generate sample:
        sample, acc = Mono_Metropolis_sweep_left(sample, A, params, cache)
        data.acceptance+=acc

        #Update data:
        update!(data, sample, A, l1, params, cache)

    end

    #Finalize:
    data.mlL/=N_MC
    data.ΔLL.=conj.(data.ΔLL) #remember to take the complex conjugate
    data.ΔLL.*=real(data.mlL)
    ∇!(data,N_MC)

    return data.∇, data.mlL, data.acceptance/(N_MC*params.N)
end


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
        l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)
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