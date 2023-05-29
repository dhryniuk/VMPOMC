export SR_cache

mutable struct SR_cache{T}
    L∂L::Array{T,3}
    ΔLL::Array{T,3}
    mean_local_Lindbladian::T
    S::Array{T,2}
    Left::Array{T}
    acc::UInt64
end

export set_SR_cache

function set_SR_cache(A,params)
    cache = SR_cache(
        zeros(eltype(A),params.χ,params.χ,4),
        zeros(eltype(A),params.χ,params.χ,4),
        0.0+0.0im,
        zeros(eltype(A),4*params.χ^2,4*params.χ^2),
        zeros(eltype(A),4*params.χ^2),
        0x000000000,
    )
    return cache
end

export MPI_SR_MPO_optimize!

function MPI_SR_MPO_optimize!(output::SR_cache,δ,A, N_MC, ϵ::AbstractFloat,  params::parameters, nworkers)

    L∂L=output.L∂L
    ΔLL=output.ΔLL
    mean_local_Lindbladian=output.mean_local_Lindbladian
    S=output.S
    Left=output.Left
    acc=output.acc

    mean_local_Lindbladian/=(N_MC*nworkers)
    ΔLL*=mean_local_Lindbladian

    #Metric tensor:
    S./=(nworkers)
    Left./=(nworkers)

    ∇ = apply_SR(S,Left,N_MC,ϵ,L∂L,ΔLL,params)
    ∇./=maximum(abs.(∇))

    acc/=(N_MC*nworkers*params.N)

    new_A=zeros(ComplexF64, params.χ,params.χ,4)
    new_A = A - δ*∇
    A = new_A
    A = normalize_MPO(MPOMC.params, A)
    return A, real(mean_local_Lindbladian), acc
end


function MPI_SR_mean!(par_cache::SR_cache, comm)
    workers_sum!(par_cache.L∂L,comm)
    workers_sum!(par_cache.ΔLL,comm)
    workers_sum!(par_cache.S,comm)
    workers_sum!(par_cache.Left,comm)
    workers_sum!(par_cache.mean_local_Lindbladian,par_cache.mean_local_Lindbladian,comm)
    workers_sum!(par_cache.acc,par_cache.acc,comm)
end


export reduced_one_worker_MPI_SR_MPO_gradient

function reduced_one_worker_MPI_SR_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::parameters, comm, par_cache::SR_cache)
    # Preallocate cache:
    cache = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample = MPO_Metropolis_burn_in(A, params, cache)

    for g in 1:N_MC

        #Initialize auxiliary arrays:
        local_L::eltype(A) = 0
        local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
        l_int::eltype(A) = 0
        cache.local_∇L_diagonal_coeff = 0

        #Generate sample:
        sample, ac = Mono_Metropolis_sweep_left(sample, A, params, cache)
        par_cache.acc+=ac

        #println("rank = $(MPI.Comm_rank(comm)), ", g, ": ", sample)

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

        par_cache.L∂L+=local_L*conj(local_∇L)

        #ΔLL:
        par_cache.ΔLL+=cache.Δ

        #Mean local Lindbladian:
        par_cache.mean_local_Lindbladian += local_L*conj(local_L)

        #Update metric tensor:
        par_cache.S, par_cache.Left = sample_update_SR(par_cache.S, par_cache.Left, params, cache)
    end
    par_cache.ΔLL.=conj.(par_cache.ΔLL) #remember to take the complex conjugate

    MPI_SR_mean!(par_cache, comm)
end
