export Exact_fMPS_gradient

function left_boundary_one_body_Hamiltonian_term(params::parameters, sample::Vector{Bool}, h1::Matrix, V::Array, 
    R_set::Union{Vector{Vector{Float64}},Vector{Vector{ComplexF64}}})
    energy = 0
    s = dVEC2[sample[1]]
    bra_L = transpose(s)*h1
    for (i,k) in zip(1:2,2:-1:1)
        loc = bra_L[i]
        if loc!=0
            energy -= loc*transpose(V[:,k])*R_set[params.N-1]
        end
    end
    return energy
end

function right_boundary_one_body_Hamiltonian_term(params::parameters, sample::Vector{Bool}, h1::Matrix, V::Array, 
    L_set::Union{Vector{Transpose{Float64, Vector{Float64}}},Vector{Transpose{ComplexF64, Vector{ComplexF64}}}})
    energy = 0
    s = dVEC2[sample[params.N]]
    bra_L = transpose(s)*h1
    for (i,k) in zip(1:2,2:-1:1)
        loc = bra_L[i]
        if loc!=0
            energy -= loc*L_set[params.N-1]*V[:,k]
        end
    end
    return energy
end

function one_body_Hamiltonian_term(params::parameters, sample::Vector{Bool}, j::UInt16, h1::Matrix, A::Array, 
    L_set::Union{Vector{Transpose{Float64, Vector{Float64}}},Vector{Transpose{ComplexF64, Vector{ComplexF64}}}}, 
    R_set::Union{Vector{Vector{Float64}},Vector{Vector{ComplexF64}}})
    energy = 0
    s = dVEC2[sample[j]]
    bra_L = transpose(s)*h1
    for (i,k) in zip(1:2,2:-1:1)
        loc = bra_L[i]
        if loc!=0
            energy += loc*tr(L_set[j-1]*A[:,:,k]*R_set[params.N-j])
        end
    end
    return energy
end

function Exact_fMPS_gradient(params::parameters, A::Array{Float64}, V::Array{Float64}, basis, h1::Matrix)
    L∇L_bulk::Array{Float64,3}=zeros(Float64,params.χ,params.χ,2) #coupled product
    ΔLL_bulk::Array{Float64,3}=zeros(Float64,params.χ,params.χ,2) #uncoupled product
    L∇L_boundary::Array{Float64}=zeros(Float64,params.χ,2) #coupled product
    ΔLL_boundary::Array{Float64}=zeros(Float64,params.χ,2) #uncoupled product
    Z=0

    mean_local_Hamiltonian::Float64 = 0

    for k in 1:params.dim#*10
        sample = basis[k]
        #sample=rand(Bool,params.N)
        #println(sample)
        L_set = L_fMPS_strings(params, sample, A, V)
        #display(L_set)
        #error()
        R_set = R_fMPS_strings(params, sample, A, V)
        ρ_sample = L_set[params.N-1]*R_set[1]
        #ρ_sample = MPS(params, sample, A)
        p_sample = ρ_sample*conj(ρ_sample)
        Z+=p_sample

        local_E=0

        L = [ transpose(Vector{Float64}(undef,params.χ)) for _ in 1:params.N-1 ]

        e_field::Float64=0
        #L∇L*:
            #left boundary:
            #1-local part (field):
            e_field -= left_boundary_one_body_Hamiltonian_term(params,sample,h1,V,R_set)

            #Update L_set:
            L=transpose(V[:,dINDEX2[sample[1]]])
            L_set[1] = copy(L)

            #bulk:
            for j::UInt16 in 2:params.N-1

                #1-local part (field):
                e_field-=one_body_Hamiltonian_term(params,sample,j,h1,A,L_set,R_set)

                #Update L_set:
                L*=A[:,:,dINDEX2[sample[j]]]
                L_set[j] = copy(L)
            end

            #right boundary:
            e_field-=right_boundary_one_body_Hamiltonian_term(params,sample,h1,V,L_set)

            #Ising interaction:
            e_int = Ising_interaction_energy(params,sample,"open")

        e_field/=ρ_sample

        #Differentiate the fMPS:
        Δ_bulk_sample, Δ_boundary_sample = ∂fMPS(params, sample, L_set, R_set)
        Δ_bulk_sample./=ρ_sample
        Δ_boundary_sample./=ρ_sample

        #Add in interaction terms:
        local_E  = e_int+e_field
        L∇L_bulk += p_sample*local_E*Δ_bulk_sample
        L∇L_boundary += p_sample*local_E*Δ_boundary_sample

        #ΔLL:
        local_Δ_bulk = p_sample*Δ_bulk_sample
        ΔLL_bulk += local_Δ_bulk
        local_Δ_boundary = p_sample*Δ_boundary_sample
        ΔLL_boundary += local_Δ_boundary

        #Mean local Lindbladian:
        mean_local_Hamiltonian += p_sample*local_E
    end

    mean_local_Hamiltonian/=Z
    ΔLL_bulk*=mean_local_Hamiltonian
    ΔLL_boundary*=mean_local_Hamiltonian

    return (L∇L_bulk-ΔLL_bulk)/Z, (L∇L_boundary-ΔLL_boundary)/Z, mean_local_Hamiltonian
end


function Exact_fMPS_gradient(params::parameters, A::Array{ComplexF64}, V::Array{ComplexF64}, basis, h1::Matrix)
    L∇L_bulk::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,2) #coupled product
    ΔLL_bulk::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,2) #uncoupled product
    L∇L_boundary::Array{ComplexF64}=zeros(ComplexF64,params.χ,2) #coupled product
    ΔLL_boundary::Array{ComplexF64}=zeros(ComplexF64,params.χ,2) #uncoupled product
    Z=0

    mean_local_Hamiltonian = 0

    for k in 1:params.dim
        sample = basis[k]
        #println(sample)
        L_set = L_fMPS_strings(params, sample, A, V)
        #display(L_set)
        #error()
        R_set = R_fMPS_strings(params, sample, A, V)
        ρ_sample = L_set[params.N-1]*R_set[1]
        #ρ_sample = MPS(params, sample, A)
        p_sample = ρ_sample*conj(ρ_sample)
        Z+=p_sample

        local_E=0

        L = [ transpose(Vector{ComplexF64}(undef,params.χ)) for _ in 1:params.N-1 ]

        e_field::ComplexF64=0
        #L∇L*:
            #left boundary:
            #1-local part (field):
            e_field -= left_boundary_one_body_Hamiltonian_term(params,sample,h1,V,R_set)

            #Update L_set:
            L=transpose(V[:,dINDEX2[sample[1]]])
            L_set[1] = copy(L)

            #bulk:
            for j::UInt16 in 2:params.N-1

                #1-local part (field):
                e_field-=one_body_Hamiltonian_term(params,sample,j,h1,A,L_set,R_set)

                #Update L_set:
                L*=A[:,:,dINDEX2[sample[j]]]
                L_set[j] = copy(L)
            end

            #right boundary:
            e_field-=right_boundary_one_body_Hamiltonian_term(params,sample,h1,V,L_set)

            #Ising interaction:
            e_int = Ising_interaction_energy(params,sample,"open")

        e_field/=ρ_sample

        #Differentiate the fMPS:
        Δ_bulk_sample, Δ_boundary_sample = conj.( ∂fMPS(params, sample, L_set, R_set) )
        Δ_bulk_sample./=ρ_sample
        Δ_boundary_sample./=ρ_sample

        #Add in interaction terms:
        local_E  = e_int+e_field
        L∇L_bulk += p_sample*local_E*Δ_bulk_sample
        L∇L_boundary += p_sample*local_E*Δ_boundary_sample

        #ΔLL:
        local_Δ_bulk = p_sample*Δ_bulk_sample
        ΔLL_bulk += local_Δ_bulk
        local_Δ_boundary = p_sample*Δ_boundary_sample
        ΔLL_boundary += local_Δ_boundary

        #Mean local Lindbladian:
        mean_local_Hamiltonian += p_sample*local_E
    end

    mean_local_Hamiltonian/=Z
    ΔLL_bulk*=mean_local_Hamiltonian
    ΔLL_boundary*=mean_local_Hamiltonian

    return (L∇L_bulk-ΔLL_bulk)/Z, (L∇L_boundary-ΔLL_boundary)/Z, mean_local_Hamiltonian
end