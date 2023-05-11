export SGD_MPO_gradient, reweighted_SGD_MPO_gradient


function bad_SGD_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, params::parameters)
        
    # Define ensemble averages:
    L∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    mean_local_Lindbladian::ComplexF64 = 0

    # Preallocate auxiliary arrays:
    AUX = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample::projector = MPO_Metropolis_burn_in(A, params, AUX)
    acceptance::UInt64=0

    for _ in 1:N_MC

        sample, AUX.R_set, acc = Mono_Metropolis_sweep_left(sample, A, AUX.L_set, params, AUX)
        acceptance+=acc
        #for n in N_sweeps
        #    sample, L_set = Mono_Metropolis_sweep_right(params, sample, A, R_set)
        #    sample, R_set = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        #end
        ρ_sample::eltype(A) = tr(AUX.R_set[params.N+1])
        AUX.L_set = L_MPO_strings(AUX.L_set, sample,A,params,AUX)

        local_L::ComplexF64 = 0
        local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
        l_int::ComplexF64 = 0

        #L∇L*:
        for j::UInt8 in 1:params.N
            #1-local part:
            lL, l∇L = bad_one_body_Lindblad_term(sample,j,l1,A,AUX.L_set,AUX.R_set,params,AUX)
            local_L += lL
            local_∇L += l∇L

            #Update L_set:
            #mul!(AUX.L_set[j+1], AUX.L_set[j], @view(A[:,:,1+2*sample.ket[j]+sample.bra[j]]))
        end

        l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        AUX.Δ_MPO_sample = ∂MPO(sample, AUX.L_set, AUX.R_set, params, AUX)./ρ_sample

        #Add in interaction terms:
        local_L +=l_int
        local_∇L+=l_int*AUX.Δ_MPO_sample

        L∇L+=local_L*conj(local_∇L)

        #ΔLL:
        local_Δ=conj(AUX.Δ_MPO_sample)
        ΔLL+=local_Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

    end
    mean_local_Lindbladian/=N_MC
    ΔLL*=mean_local_Lindbladian
    return (L∇L-ΔLL)/N_MC, real(mean_local_Lindbladian), acceptance/(N_MC*params.N)
end

function SGD_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, params::parameters)
        
    # Define ensemble averages:
    L∂L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    mean_local_Lindbladian::ComplexF64 = 0

    # Preallocate auxiliary arrays:
    AUX = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample::projector = MPO_Metropolis_burn_in(A, params, AUX)
    acceptance::UInt64=0

    for _ in 1:N_MC

        #Initialize auxiliary arrays:
        local_L::ComplexF64 = 0
        local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
        l_int::ComplexF64 = 0
        AUX.local_∇L_diagonal_coeff = 0

        #Generate sample:
        sample, acc = Mono_Metropolis_sweep_left(sample, A, params, AUX)
        acceptance+=acc

        ρ_sample::eltype(A) = tr(AUX.R_set[params.N+1])
        AUX.L_set = L_MPO_strings(AUX.L_set, sample,A,params,AUX)
        AUX.Δ = ∂MPO(sample, AUX.L_set, AUX.R_set, params, AUX)./ρ_sample

        #Calculate L∂L*:
        for j::UInt8 in 1:params.N
            #1-local part:
            lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,params,AUX)
            local_L += lL
            local_∇L .+= l∇L
        end

        local_L  /=ρ_sample
        local_∇L./=ρ_sample

        #Add in diagonal part of the local derivative:
        local_∇L.+=AUX.local_∇L_diagonal_coeff.*AUX.Δ

        #Add in Ising interaction terms:
        l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)
        local_L  +=l_int
        local_∇L.+=l_int*AUX.Δ

        L∂L.+=local_L*conj(local_∇L)

        #ΔLL:
        ΔLL.+=AUX.Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

    end
    mean_local_Lindbladian/=N_MC
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate
    ΔLL.*=real(mean_local_Lindbladian)
    return (L∂L-ΔLL)/N_MC, real(mean_local_Lindbladian), acceptance/(N_MC*params.N)
end


function reweighted_SGD_MPO_gradient(β::Float64, A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, params::parameters)
    # Define ensemble averages:
    L∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    mean_local_Lindbladian::ComplexF64 = 0
    Z::Float64 = 0

    # Preallocate auxiliary arrays:
    AUX = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample = MPO_Metropolis_burn_in(A, params, AUX)
    acceptance::UInt64=0

    for _ in 1:N_MC

        #Initialize auxiliary arrays:
        local_L::ComplexF64 = 0
        local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
        l_int::ComplexF64 = 0
        AUX.local_∇L_diagonal_coeff = 0

        #Generate sample:
        sample, acc = reweighted_Mono_Metropolis_sweep_left(β, sample, A, params, AUX)
        acceptance+=acc

        ρ_sample::eltype(A) = tr(AUX.R_set[params.N+1])
        p_sample=(ρ_sample*conj(ρ_sample))^(1-β)
        Z+=p_sample
        AUX.L_set = L_MPO_strings(AUX.L_set, sample,A,params,AUX)
        AUX.Δ = ∂MPO(sample, AUX.L_set, AUX.R_set, params, AUX)./ρ_sample

        #L∇L*:
        for j::UInt8 in 1:params.N
            #1-local part:
            lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,params,AUX)
            local_L += lL
            local_∇L += l∇L
        end

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        #Add in diagonal part of the local derivative:
        local_∇L.+=AUX.local_∇L_diagonal_coeff.*AUX.Δ

        #Add in interaction terms:
        l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)
        local_L +=l_int
        local_∇L+=l_int*AUX.Δ

        L∇L+=p_sample*local_L*conj(local_∇L)

        #ΔLL:
        ΔLL.+=AUX.Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += p_sample*local_L*conj(local_L)

    end
    mean_local_Lindbladian/=Z
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate
    ΔLL.*=real(mean_local_Lindbladian)
    return (L∇L-ΔLL)/Z, real(mean_local_Lindbladian), acceptance/(N_MC*params.N)
end


"""
export partial_SGD_MPO_gradient

function partial_SGD_MPO_gradient(params::parameters, A::Array{ComplexF64,3}, l1::Matrix{ComplexF64}, N_MC::Int64)#, N_sweeps::Int64)
    L∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)

    mean_local_Lindbladian::ComplexF64 = 0

    # Initialize sample and L_set for that sample:
    sample, L_set = MPO_Metropolis_burn_in(params, A)
    acceptance::UInt64=0

    for _ in 1:N_MC

        sample, R_set, acc = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        acceptance+=acc
        #for n in N_sweeps
        #    sample, L_set = Mono_Metropolis_sweep_right(params, sample, A, R_set)
        #    sample, R_set = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        #end
        ρ_sample = tr(R_set[params.N+1])
        L_set = [ Matrix{ComplexF64}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
        L = Matrix{ComplexF64}(I, params.χ, params.χ)
        L_set[1] = L

        local_L::ComplexF64 = 0
        local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
        l_int::ComplexF64 = 0

        #L∇L*:
        for j::UInt16 in 1:params.N

            #1-local part:
            lL, l∇L = partial_one_body_Lindblad_term(params,sample,j,l1,A,L_set,R_set)
            #lL, l∇L = one_body_Lindblad_term(params,sample_ket,sample_bra,j,l1,A,L_set,R_set)
            local_L += lL
            local_∇L += l∇L

            #2-local part:
            #l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,params.N)+1]-1)
            #l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,params.N)+1]-1)
            #l_int += -1.0im*J*(l_int_α-l_int_β)
            #l_int += 1.0im*params.J*(l_int_α-l_int_β)

            #Update L_set:
            #L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
            L*=A[:,:,1+2*sample.ket[j]+sample.bra[j]]
            L_set[j+1] = L
        end

        l_int = Lindblad_Ising_interaction_energy(params, sample, "periodic")

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        Δ_MPO_sample = ∂MPO(params, sample, L_set, R_set)/ρ_sample

        #Add in interaction terms:
        local_L +=l_int
        local_∇L+=l_int*Δ_MPO_sample

        L∇L+=local_L*conj(local_∇L)

        #ΔLL:
        local_Δ=conj(Δ_MPO_sample)
        ΔLL+=local_Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

    end
    mean_local_Lindbladian/=N_MC
    ΔLL*=mean_local_Lindbladian
    return (L∇L-ΔLL)/N_MC, real(mean_local_Lindbladian), acceptance/(N_MC*params.N)
end
"""