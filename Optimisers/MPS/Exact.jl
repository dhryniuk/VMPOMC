export Exact_MPS_gradient

function one_body_Hamiltonian_term(params::parameters, sample::Vector{Bool}, j::UInt16, h1::Matrix, A::Array, 
    L_set::Union{Vector{Matrix{Float64}},Vector{Matrix{ComplexF64}}}, 
    R_set::Union{Vector{Matrix{Float64}},Vector{Matrix{ComplexF64}}})
    energy = 0
    s = dVEC2[sample[j]]
    bra_L = transpose(s)*h1
    for (i,k) in zip(1:2,2:-1:1)
        loc = bra_L[i]
        if loc!=0
            energy += loc*tr(L_set[j]*A[:,:,k]*R_set[params.N+1-j])
        end
    end
    return energy
end

function one_body_Hamiltonian_term(params::parameters, sample::Vector{Bool}, j::UInt16, h1::Matrix, A::Array, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})
    energy::ComplexF64 = 0
    s = dVEC2[sample[j]]
    bra_L = transpose(s)*h1
    for (i,k) in zip(1:2,2:-1:1)
        loc = bra_L[i]
        if loc!=0
            energy += loc*tr(L_set[j]*A[:,:,k]*R_set[params.N+1-j])
        end
    end
    return energy::ComplexF64
end


function Ising_interaction_energy(params::parameters, sample::Vector{Bool}, boundary_conditions)
    energy=0
    for j in 1:params.N-1
        energy -= (2*sample[j]-1) * (2*sample[j+1]-1)
    end
    if boundary_conditions=="periodic"
        energy -= (2*sample[params.N]-1) * (2*sample[1]-1)
    end
    return params.J*energy
end

function Exact_MPS_gradient(params::parameters, A::Array{Float64}, basis, h1::Matrix)
    L∇L::Array{Float64,3}=zeros(Float64,params.χ,params.χ,2) #coupled product
    ΔLL::Array{Float64,3}=zeros(Float64,params.χ,params.χ,2) #uncoupled product
    Z=0
    mean_local_Hamiltonian::Float64 = 0

    for _ in 1:params.dim*10
        #sample = basis[k]
        sample = rand(Bool,params.N)
        L_set = L_MPS_strings(params, sample, A)
        #println(sample)
        #display(ρ_sample)
        #display(L_set)
        #display(A)
        #println()
        R_set = R_MPS_strings(params, sample, A)
        ρ_sample = tr(L_set[params.N+1])
        p_sample = ρ_sample*conj(ρ_sample)
        Z+=p_sample

        L_set = Vector{Matrix{Float64}}()
        L=Matrix{Float64}(I, params.χ, params.χ)
        push!(L_set,copy(L))

        e_field=0
        #L∇L*:
        for j::UInt16 in 1:params.N
            #1-local part (field):
            e_field -= one_body_Hamiltonian_term(params, sample, j, h1, A, L_set, R_set)

            #Update L_set:
            L*=A[:,:,dINDEX2[sample[j]]]
            push!(L_set,copy(L))
        end
        e_field/=ρ_sample
        #Only divide e_field because probability amplitudes for e_int cancels out exactly since interactions act as eigenoperators of the computational basis

        #Interaction term:
        e_int = Ising_interaction_energy(params, sample, "periodic")

        #Calculate logarithmic derivative of MPS:
        Δ_MPO_sample = ∂MPS(params, sample, L_set, R_set)/ρ_sample

        #Add in interaction terms:
        local_E  = e_int+e_field
        L∇L += p_sample*Δ_MPO_sample*local_E

        #ΔLL:
        local_Δ = p_sample*Δ_MPO_sample
        ΔLL += local_Δ

        #Mean local Lindbladian:
        mean_local_Hamiltonian += p_sample*local_E
    end
"""
    println("L∇L")
    display(L∇L/Z)

    println("ΔLL")
    display(ΔLL/Z)

    println("mean_local_Hamiltonian= ", mean_local_Hamiltonian)
    println("Z= ", Z)
"""
    mean_local_Hamiltonian/=Z
    ΔLL*=mean_local_Hamiltonian

    return (L∇L-ΔLL)/Z, mean_local_Hamiltonian
end

function Exact_MPS_gradient(params::parameters, A::Array{ComplexF64}, basis, h1::Matrix)
    L∇L=zeros(ComplexF64,params.χ,params.χ,2) #coupled product
    ΔLL=zeros(ComplexF64,params.χ,params.χ,2) #uncoupled product
    Z=0
    mean_local_Hamiltonian = 0

    for k in 1:params.dim
        sample = basis[k]
        L_set = L_MPS_strings(params, sample, A)
        R_set = R_MPS_strings(params, sample, A)
        ρ_sample = tr(L_set[params.N+1])
        p_sample = ρ_sample*conj(ρ_sample)
        Z+=p_sample

        local_E=0

        L_set = Vector{Matrix{ComplexF64}}()
        L=Matrix{ComplexF64}(I, params.χ, params.χ)
        push!(L_set,copy(L))

        e_field=0
        #L∇L*:
        for j::UInt16 in 1:params.N

            #1-local part (field):
            e_field -= one_body_Hamiltonian_term(params, sample, j, h1, A, L_set, R_set)

            #Update L_set:
            L*=A[:,:,dINDEX2[sample[j]]]
            push!(L_set,copy(L))
        end
        e_field/=ρ_sample

        #Interaction term:
        e_int = Ising_interaction_energy(params, sample, "periodic")

        #Calculate logarithmic derivative of MPS:
        Δ_MPO_sample = conj( ∂MPS(params, sample, L_set, R_set)/ρ_sample )

        #Add in interaction terms:
        local_E  = e_int+e_field
        L∇L += p_sample*Δ_MPO_sample*local_E

        #ΔLL:
        ΔLL +=  p_sample*Δ_MPO_sample

        #Mean local Lindbladian:
        mean_local_Hamiltonian += p_sample*local_E
    end

    mean_local_Hamiltonian/=Z
    ΔLL*=mean_local_Hamiltonian

    return (L∇L-ΔLL)/Z, mean_local_Hamiltonian
end