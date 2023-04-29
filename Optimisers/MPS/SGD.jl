export SGD_MPS_gradient

function SGD_MPS_gradient(params::parameters, A::Array{Float64}, N_MC::Int64, h1::Matrix)

    # Initialize products:
    L∇L::Array{Float64,3} = zeros(Float64,params.χ,params.χ,2) #coupled product
    ΔLL::Array{Float64,3} = zeros(Float64,params.χ,params.χ,2) #uncoupled product

    mean_local_Hamiltonian::Float64 = 0

    # Initialize sample and L_set for that sample:
    sample, L_set = Metropolis_burn_in(params, A)
    acceptance::UInt64=0

    for _ in 1:N_MC
        sample, R_set, acc = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        acceptance+=acc
        ρ_sample = tr(R_set[params.N+1])

        local_E=0

        L_set = Vector{Matrix{Float64}}()
        L=Matrix{Float64}(I, params.χ, params.χ)
        push!(L_set,copy(L))

        e_field::Float64=0
        #L∇L*:
        for j::UInt16 in 1:params.N
            #1-local part (field):
            e_field += one_body_Hamiltonian_term(params, sample, j, h1, A, L_set, R_set)

            #Update L_set:
            L*=A[:,:,dINDEX2[sample[j]]]
            push!(L_set,copy(L))
        end
        e_field/=ρ_sample

        #Interaction term:
        e_int = Ising_interaction_energy(params, sample, "periodic")

        Δ_MPO_sample = ∂MPS(params, sample, L_set, R_set)/ρ_sample

        #Add in interaction terms:
        local_E = e_int+e_field
        L∇L += local_E*Δ_MPO_sample

        #ΔLL:
        local_Δ = Δ_MPO_sample
        ΔLL += local_Δ

        #Mean local Lindbladian:
        mean_local_Hamiltonian += local_E
    end

    mean_local_Hamiltonian/=N_MC
    ΔLL*=mean_local_Hamiltonian

    return (L∇L-ΔLL)/N_MC, mean_local_Hamiltonian, acceptance/(N_MC*params.N)
end

function SGD_MPS_gradient(params::parameters, A::Array{ComplexF64}, N_MC::Int64, h1::Matrix)

    # Initialize products:
    L∇L::Array{Float64,3} = zeros(Float64,params.χ,params.χ,2) #coupled product
    ΔLL::Array{Float64,3} = zeros(Float64,params.χ,params.χ,2) #uncoupled product

    mean_local_Hamiltonian::Float64 = 0

    # Initialize sample and L_set for that sample:
    sample, L_set = Metropolis_burn_in(params, A)
    acceptance::UInt64=0

    for _ in 1:N_MC
        sample, R_set, acc = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        acceptance+=acc
        ρ_sample = tr(R_set[params.N+1])

        local_E=0

        L_set = Vector{Matrix{ComplexF64}}()
        L=Matrix{ComplexF64}(I, params.χ, params.χ)
        push!(L_set,copy(L))

        e_field=0
        #L∇L*:
        for j in 1:params.N
            #1-local part (field):
            e_field += one_body_Hamiltonian_term(params, sample, j, h1, A, L_set, R_set)

            #Update L_set:
            L*=A[:,:,dINDEX2[sample[j]]]
            push!(L_set,copy(L))
        end
        e_field/=ρ_sample

        #Interaction term:
        e_int = Ising_interaction_energy(params, sample, "periodic")

        Δ_MPO_sample = conj( ∂MPS(params, sample, L_set, R_set)/ρ_sample )

        #Add in interaction terms:
        local_E  = e_int+e_field
        L∇L += local_E*Δ_MPO_sample

        #ΔLL:
        local_Δ = Δ_MPO_sample
        ΔLL += local_Δ

        #Mean local Lindbladian:
        mean_local_Hamiltonian += local_E
    end

    mean_local_Hamiltonian/=N_MC
    ΔLL*=mean_local_Hamiltonian

    return (L∇L-ΔLL)/N_MC, mean_local_Hamiltonian, acceptance/(N_MC*params.N)
end