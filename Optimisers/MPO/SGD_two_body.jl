#export SGD_MPO_gradient_two_body

function SGD_MPO_gradient_two_body(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, l2::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, params::parameters)
        
    # Define ensemble averages:
    L∂L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    mean_local_Lindbladian::ComplexF64 = 0

    # Preallocate cache:
    cache = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample::projector = MPO_Metropolis_burn_in(A, params, cache)
    acceptance::UInt64=0

    for _ in 1:N_MC

        #Initialize auxiliary arrays:
        local_L::ComplexF64 = 0
        local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
        l_int::ComplexF64 = 0
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
            local_L += lL
            local_∇L .+= l∇L
        end
        for j::UInt8 in 1:params.N-1
            lL, l∇L = two_body_Lindblad_term(sample,j,l2,A,params,cache)
            local_L += lL
            local_∇L += l∇L
        end
        if params.N>2
            lL, l∇L = boundary_two_body_Lindblad_term(sample,l2,A,params,cache)
            local_L += lL
            local_∇L += l∇L
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
        L∂L.+=local_L*conj(local_∇L)

        #Update ΔLL ensemble average:
        ΔLL.+=cache.Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

    end
    mean_local_Lindbladian/=N_MC
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate
    ΔLL.*=real(mean_local_Lindbladian)
    return (L∂L-ΔLL)/N_MC, real(mean_local_Lindbladian), acceptance/(N_MC*params.N)
end