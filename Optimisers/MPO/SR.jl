export SR_MPO_gradient, distributed_SR_MPO_gradient, reweighted_SR_MPO_gradient

function MPO_flatten_index(i::UInt8,j::UInt8,s::UInt8,params::parameters)
    return i+params.χ*(j-1)+params.χ^2*(s-1)
end

function sample_update_SR(S::Array{<:Complex{<:AbstractFloat},2}, avg_G::Array{<:Complex{<:AbstractFloat}}, params::parameters, cache::workspace)
    G = reshape(cache.Δ,4*params.χ^2)
    conj_G = conj(G)
    avg_G.+= G
    mul!(cache.plus_S,conj_G,transpose(G))
    S.+=cache.plus_S   #use when l1 is unconjugated
    #S.+=conj(cache.plus_S)
    return S, avg_G
end

function apply_SR(S::Array{<:Complex{<:AbstractFloat},2}, avg_G::Array{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, 
    L∂L::Array{<:Complex{<:AbstractFloat},3}, ΔLL::Array{<:Complex{<:AbstractFloat},3}, params::parameters)

    #Compute metric tensor:
    S./=N_MC
    avg_G./=N_MC
    conj_avg_G = conj(avg_G)
    S-=avg_G*transpose(conj_avg_G)

    #Regularize the metric tensor:
    S+=ϵ*Matrix{Int}(I, params.χ*params.χ*4, params.χ*params.χ*4)

    #Find SR'd gradient:
    grad::Array{eltype(S),3} = (L∂L-ΔLL)/N_MC
    flat_grad::Vector{eltype(S)} = reshape(grad,4*params.χ^2)
    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad,params.χ,params.χ,4)

    return grad
end

function SR_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::parameters)
    
    # Define ensemble averages:
    L∂L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    ΔLL::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    mean_local_Lindbladian::eltype(A) = 0

    # Preallocate auxiliary arrays:
    cache = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample::projector = MPO_Metropolis_burn_in(A, params, cache)
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
        cache.L_set = L_MPO_strings(cache.L_set, sample,A,params,cache)
        cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache)./ρ_sample

        #L∂L*:
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
        l_int = Lindblad_Ising_interaction_energy(samcacheple, "periodic", A, params)
        local_L +=l_int
        local_∇L+=l_int*cache.Δ

        #Update L∂L* ensemble average:
        L∂L.+=local_L*conj(local_∇L)

        #Update ΔLL ensemble average:
        ΔLL.+=cache.Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

        #Update metric tensor:
        S, Left = sample_update_SR(S, Left, params, cache)
    end
    mean_local_Lindbladian/=N_MC
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate
    ΔLL*=real(mean_local_Lindbladian)

    #Reconfigure gradient:
    grad = apply_SR(S,Left,N_MC,ϵ,L∂L,ΔLL,params)

    return grad, real(mean_local_Lindbladian), acceptance/(N_MC*params.N)
end

function reweighted_sample_update_SR(p_sample::Float64, S::Array{<:Complex{<:AbstractFloat},2}, avg_G::Array{<:Complex{<:AbstractFloat}}, params::parameters, cache::workspace)
    G = reshape(cache.Δ,4*params.χ^2)
    conj_G = conj(G)
    avg_G.+= p_sample*G
    mul!(cache.plus_S,conj_G,transpose(G))
    S.+=cache.plus_S
    return S, avg_G
end

function reweighted_apply_SR(S::Array{<:Complex{<:AbstractFloat},2}, avg_G::Array{<:Complex{<:AbstractFloat}}, Z::Float64, ϵ::AbstractFloat, 
    L∇L::Array{<:Complex{<:AbstractFloat},3}, ΔLL::Array{<:Complex{<:AbstractFloat},3}, params::parameters)

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

function reweighted_SR_MPO_gradient(β::Float64, A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::parameters)
    
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
        local_L::ComplexF64 = 0
        local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
        l_int::ComplexF64 = 0
        cache.local_∇L_diagonal_coeff = 0

        #Generate sample:
        sample, acc = reweighted_Mono_Metropolis_sweep_left(β, sample, A, params, cache)
        acceptance+=acc

        ρ_sample::eltype(A) = tr(cache.R_set[params.N+1])
        p_sample::Float64=(ρ_sample*conj(ρ_sample))^(1-β)
        Z+=p_sample
        cache.L_set = L_MPO_strings(cache.L_set, sample,A,params,cache)
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

function one_worker_SR_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::parameters)
    
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
        cache.L_set = L_MPO_strings(cache.L_set, sample,A,params,cache)
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

function distributed_SR_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::parameters)
    #output = [L∇L, ΔLL, mean_local_Lindbladian, S, Left, Right]

    #perform reduction:
    output = @distributed (+) for i=1:nworkers()
        #sample_with_SR_long_range(p, A, l1, N_MC, N_sweeps)
        #one_worker_SR_MPO_gradient(A, l1, convert(Int64,ceil(N_MC/nworkers())), ϵ, params)
        one_worker_SR_MPO_gradient(A, l1, N_MC, ϵ, params)
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