export SR_MPS_gradient

function SR_MPS_gradient(params::Parameters, A::Array{Float64}, N_MC::Int64, ϵ::Float64, h1::Matrix)

    # Initialize products:
    L∇L::Array{Float64,3} = zeros(Float64,params.χ,params.χ,2) #coupled product
    ΔLL::Array{Float64,3} = zeros(Float64,params.χ,params.χ,2) #uncoupled product

    # Initialize metric tensor auxiliary arrays:
    S = zeros(Float64, 2*params.χ^2, 2*params.χ^2)
    G = zeros(Float64, params.χ, params.χ, 2)
    Left = zeros(Float64, params.χ, params.χ, 2)
    Right = zeros(Float64, params.χ, params.χ, 2)

    mean_local_Hamiltonian::Float64 = 0

    # Initialize sample and L_set for that sample:
    sample, L_set = Metropolis_burn_in(params, A)
    acceptance::UInt64=0

    for _ in 1:N_MC
        sample, R_set, acc = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        acceptance+=acc
        ρ_sample = tr(R_set[params.N+1])

        # Prepare new L_set of left MPS strings:
        L_set = [ Matrix{Float64}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
        L = Matrix{Float64}(I, params.χ, params.χ)
        L_set[1] = L

        e_field::Float64 = 0
        e_int::Float64 = 0

        #L∇L*:
        for j::UInt16 in 1:params.N
            #1-local part (field):
            e_field += one_body_Hamiltonian_term(params, sample, j, h1, A, L_set, R_set)

            #Update L_set:
            L*=A[:,:,2-sample[j]]
            L_set[j+1] = L
        end
        e_field/=ρ_sample

        #Interaction term:
        e_int = Ising_interaction_energy(params, sample, "periodic")

        Δ_MPO_sample = ∂MPS(params, sample, L_set, R_set)/ρ_sample

        #Add energies:
        local_E::Float64 = e_int+e_field
        L∇L += local_E*Δ_MPO_sample

        #ΔLL:
        local_Δ = Δ_MPO_sample
        ΔLL += local_Δ

        #Mean local Lindbladian:
        mean_local_Hamiltonian += local_E

        #Metric tensor:
        G = Δ_MPO_sample
        Left += G #change order of conjugation, but it shouldn't matter
        for s in 1:2, j in 1:params.χ, i in 1:params.χ, ss in 1:2, jj in 1:params.χ, ii in 1:params.χ
            @inbounds S[flatten_index(i,j,s,params),flatten_index(ii,jj,ss,params)] += conj(G[i,j,s])*G[ii,jj,ss]
        end
    end
    mean_local_Hamiltonian/=N_MC
    ΔLL*=mean_local_Hamiltonian

    #Metric tensor:
    S./=N_MC
    Left./=N_MC
    Right = conj(Left)
    for s in 1:2, j in 1:params.χ, i in 1:params.χ, ss in 1:2, jj in 1:params.χ, ii in 1:params.χ 
        @inbounds S[flatten_index(i,j,s,params),flatten_index(ii,jj,ss,params)] -= Left[i,j,s]*Right[ii,jj,ss]
    end
    S+=ϵ*Matrix{Int}(I, 2*params.χ^2, 2*params.χ^2)

    grad = (L∇L-ΔLL)/N_MC
    flat_grad = reshape(grad, 2*params.χ^2)

    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad, params.χ, params.χ,2 )

    return grad, mean_local_Hamiltonian, acceptance/(params.N*N_MC)
end

function SR_MPS_gradient(params::Parameters, A::Array{ComplexF64}, N_MC::Int64, ϵ::Float64, h1::Matrix)

    # Initialize products:
    L∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,2) #coupled product
    ΔLL::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,2) #uncoupled product

    # Initialize metric tensor auxiliary arrays:
    S = zeros(ComplexF64, 2*params.χ^2, 2*params.χ^2)
    G = zeros(ComplexF64, params.χ, params.χ, 2)
    Left = zeros(ComplexF64, params.χ, params.χ, 2)
    Right = zeros(ComplexF64, params.χ, params.χ, 2)

    mean_local_Hamiltonian::ComplexF64 = 0

    # Initialize sample and L_set for that sample:
    sample, L_set = Metropolis_burn_in(params, A)
    acceptance::UInt64=0

    for _ in 1:N_MC
        sample, R_set, acc = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        acceptance+=acc
        ρ_sample = tr(R_set[params.N+1])

        # Prepare new L_set of left MPS strings:
        L_set = [ Matrix{ComplexF64}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
        L = Matrix{ComplexF64}(I, params.χ, params.χ)
        L_set[1] = L

        e_field::ComplexF64 = 0
        e_int::ComplexF64 = 0

        #L∇L*:
        for j::UInt16 in 1:params.N
            #1-local part (field):
            e_field += one_body_Hamiltonian_term(params, sample, j, h1, A, L_set, R_set)

            #Update L_set:
            L*=A[:,:,2-sample[j]]
            L_set[j+1] = L
        end
        e_field/=ρ_sample

        #Interaction term:
        e_int = Ising_interaction_energy(params, sample, "periodic")

        Δ_MPO_sample = conj( ∂MPS(params, sample, L_set, R_set)/ρ_sample )

        #Add energies:
        local_E::ComplexF64 = e_int+e_field
        L∇L += local_E*Δ_MPO_sample

        #ΔLL:
        local_Δ = Δ_MPO_sample
        ΔLL += local_Δ

        #Mean local Lindbladian:
        mean_local_Hamiltonian += local_E

        #Metric tensor:
        G = Δ_MPO_sample
        Left += G #change order of conjugation, but it shouldn't matter
        for s in 1:2, j in 1:params.χ, i in 1:params.χ, ss in 1:2, jj in 1:params.χ, ii in 1:params.χ
            @inbounds S[flatten_index(i,j,s,params),flatten_index(ii,jj,ss,params)] += conj(G[i,j,s])*G[ii,jj,ss]
        end
    end
    mean_local_Hamiltonian/=N_MC
    ΔLL*=mean_local_Hamiltonian

    #Metric tensor:
    S./=N_MC
    Left./=N_MC
    Right = conj(Left)
    for s in 1:2, j in 1:params.χ, i in 1:params.χ, ss in 1:2, jj in 1:params.χ, ii in 1:params.χ 
        @inbounds S[flatten_index(i,j,s,params),flatten_index(ii,jj,ss,params)] -= Left[i,j,s]*Right[ii,jj,ss]
    end
    S+=ϵ*Matrix{Int}(I, 2*params.χ^2, 2*params.χ^2)

    grad = (L∇L-ΔLL)/N_MC
    flat_grad = reshape(grad, 2*params.χ^2)

    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad, params.χ, params.χ,2 )

    return grad, mean_local_Hamiltonian, acceptance/(params.N*N_MC)
end



export Exact_SR_MPS_gradient


function Exact_SR_MPS_gradient(params::Parameters, A::Array{Float64}, basis, h1::Matrix, ϵ::Float64)

    # Initialize products:
    L∇L::Array{Float64,3} = zeros(Float64,params.χ,params.χ,2) #coupled product
    ΔLL::Array{Float64,3} = zeros(Float64,params.χ,params.χ,2) #uncoupled product

    # Initialize metric tensor auxiliary arrays:
    S = zeros(Float64, 2*params.χ^2, 2*params.χ^2)
    avg_g = zeros(Float64, 2*params.χ^2)

    Z=0
    mean_local_Hamiltonian::Float64 = 0

    for k in 1:params.dim
        sample = basis[k]
        R_set = R_MPS_strings(params, sample, A)
        ρ_sample = tr(R_set[params.N+1])
        p_sample = ρ_sample*conj(ρ_sample)
        Z+=p_sample

        # Prepare new L_set of left MPS strings:
        L_set = [ Matrix{Float64}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
        L = Matrix{Float64}(I, params.χ, params.χ)
        L_set[1] = L

        e_field::Float64 = 0
        e_int::Float64 = 0

        #L∇L*:
        for j::UInt16 in 1:params.N
            #1-local part (field):
            e_field += one_body_Hamiltonian_term(params, sample, j, h1, A, L_set, R_set)

            #Update L_set:
            L*=A[:,:,2-sample[j]]
            L_set[j+1] = L
        end
        e_field/=ρ_sample

        #Interaction term:
        e_int = Ising_interaction_energy(params, sample, "periodic")

        Δ_MPO_sample = ∂MPS(params, sample, L_set, R_set)/ρ_sample

        #Add energies:
        local_E::Float64 = e_int+e_field
        L∇L += p_sample*local_E*Δ_MPO_sample

        #ΔLL:
        local_Δ = p_sample*Δ_MPO_sample
        ΔLL += local_Δ

        #Mean local Lindbladian:
        mean_local_Hamiltonian += p_sample*local_E

        #Metric tensor:
        g = reshape(Δ_MPO_sample,2*params.χ^2)
        avg_g += g #times p_α?
        S += g*transpose(g) 
    end
    mean_local_Hamiltonian/=Z
    ΔLL*=mean_local_Hamiltonian

    #Metric tensor:
    S./=Z
    avg_g./=Z
    S -= avg_g*transpose(avg_g)
    S+=ϵ*Matrix{Int}(I, 2*params.χ^2, 2*params.χ^2)

    grad = (L∇L-ΔLL)/Z
    flat_grad = reshape(grad, 2*params.χ^2)

    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad, params.χ, params.χ,2 )

    return grad, mean_local_Hamiltonian, 0
end


export old_Exact_SR_MPS_gradient

function old_Exact_SR_MPS_gradient(params::Parameters, A::Array{Float64}, basis, h1::Matrix, ϵ::Float64)

    # Initialize products:
    L∇L::Array{Float64,3} = zeros(Float64,params.χ,params.χ,2) #coupled product
    ΔLL::Array{Float64,3} = zeros(Float64,params.χ,params.χ,2) #uncoupled product

    # Initialize metric tensor auxiliary arrays:
    S = zeros(Float64, 2*params.χ^2, 2*params.χ^2)
    G = zeros(Float64, params.χ, params.χ, 2)
    Left = zeros(Float64, params.χ, params.χ, 2)
    Right = zeros(Float64, params.χ, params.χ, 2)

    Z=0
    mean_local_Hamiltonian::Float64 = 0

    for k in 1:params.dim
        sample = basis[k]
        R_set = R_MPS_strings(params, sample, A)
        ρ_sample = tr(R_set[params.N+1])
        p_sample = ρ_sample*conj(ρ_sample)
        Z+=p_sample

        # Prepare new L_set of left MPS strings:
        L_set = [ Matrix{Float64}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
        L = Matrix{Float64}(I, params.χ, params.χ)
        L_set[1] = L

        e_field::Float64 = 0
        e_int::Float64 = 0

        #L∇L*:
        for j::UInt16 in 1:params.N
            #1-local part (field):
            e_field -= one_body_Hamiltonian_term(params, sample, j, h1, A, L_set, R_set)

            #Update L_set:
            L*=A[:,:,2-sample[j]]
            L_set[j+1] = L
        end
        e_field/=ρ_sample

        #Interaction term:
        e_int = Ising_interaction_energy(params, sample, "periodic")

        Δ_MPO_sample = ∂MPS(params, sample, L_set, R_set)/ρ_sample

        #Add energies:
        local_E::Float64 = e_int+e_field
        L∇L += p_sample*local_E*Δ_MPO_sample

        #ΔLL:
        local_Δ = p_sample*Δ_MPO_sample
        ΔLL += local_Δ

        #Mean local Lindbladian:
        mean_local_Hamiltonian += p_sample*local_E

        #Metric tensor:
        G = Δ_MPO_sample
        Left += G #change order of conjugation, but it shouldn't matter
        for s in 1:2, j in 1:params.χ, i in 1:params.χ, ss in 1:2, jj in 1:params.χ, ii in 1:params.χ
            @inbounds S[flatten_index(i,j,s,params),flatten_index(ii,jj,ss,params)] += conj(G[i,j,s])*G[ii,jj,ss]
        end
    end
    mean_local_Hamiltonian/=Z
    ΔLL*=mean_local_Hamiltonian

    #Metric tensor:
    S./=Z
    Left./=Z
    Right = conj(Left)
    for s in 1:2, j in 1:params.χ, i in 1:params.χ, ss in 1:2, jj in 1:params.χ, ii in 1:params.χ 
        @inbounds S[flatten_index(i,j,s,params),flatten_index(ii,jj,ss,params)] -= Left[i,j,s]*Right[ii,jj,ss]
    end
    S+=ϵ*Matrix{Int}(I, 2*params.χ^2, 2*params.χ^2)

    grad = (L∇L-ΔLL)/Z
    flat_grad = reshape(grad, 2*params.χ^2)

    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad, params.χ, params.χ,2 )

    return grad, mean_local_Hamiltonian, 0
end
