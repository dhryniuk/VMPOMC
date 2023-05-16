#export Exact_MPO_gradient_two_body

function two_body_Lindblad_term(sample::projector, k::UInt8, l2::Matrix, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, cache::workspace)
    local_L::eltype(A) = 0
    local_∇L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)

    s1::Matrix{eltype(A)} = dVEC_transpose[(sample.ket[k],sample.bra[k])]
    s2::Matrix{eltype(A)} = dVEC_transpose[(sample.ket[k+1],sample.bra[k+1])]
    s = kron(s1,s2)
    bra_L::Matrix{eltype(A)} = s*conj(l2)
    #@inbounds for i::UInt16 in 1:4, j::UInt16 in 1:4
    for (i::UInt8,state_i::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})
        for (j::UInt8,state_j::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})

            loc::eltype(A) = bra_L[j+4*(i-1)]
            if loc!=0
                local_L += loc*tr( (cache.L_set[k]*A[:,:,i])*A[:,:,j]*cache.R_set[(params.N-k)])
                micro_sample::projector = projector(sample)
                micro_sample.ket[k] = state_i[1]
                micro_sample.bra[k] = state_i[2]
                micro_sample.ket[k+1] = state_j[1]
                micro_sample.bra[k+1] = state_j[2]
                
                cache.micro_L_set = L_MPO_strings(cache.micro_L_set, micro_sample, A, params, cache)
                cache.micro_R_set = R_MPO_strings(cache.micro_R_set, micro_sample, A, params, cache)
                local_∇L.+= loc.*∂MPO(micro_sample, cache.micro_L_set, cache.micro_R_set, params, cache)
            end
        end
    end
    return local_L, local_∇L
end

function boundary_two_body_Lindblad_term(sample::projector, l2::Matrix, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, cache::workspace)

    #Need to find middle matrix product, by inverting the first tensor A:
    M = inv(A[:,:,dINDEX[(sample.ket[1],sample.bra[1])]])*cache.L_set[params.N]

    local_L::eltype(A) = 0
    local_∇L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)

    s1::Matrix{eltype(A)} = dVEC_transpose[(sample.ket[params.N],sample.bra[params.N])]
    s2::Matrix{eltype(A)} = dVEC_transpose[(sample.ket[1],sample.bra[1])]
    s = kron(s1,s2)
    bra_L::Matrix{eltype(A)} = s*conj(l2)
    #@inbounds for i::UInt16 in 1:4, j::UInt16 in 1:4
    for (i::UInt8,state_i::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})
        for (j::UInt8,state_j::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})

            loc::eltype(A) = bra_L[j+4*(i-1)]
            if loc!=0
                local_L += loc*tr( M*A[:,:,i]*A[:,:,j] )
                #local_L += loc*MPO(params,sample,A)
                micro_sample::projector = projector(sample)
                micro_sample.ket[1] = state_i[1]
                micro_sample.bra[1] = state_i[2]
                micro_sample.ket[params.N] = state_j[1]
                micro_sample.bra[params.N] = state_j[2]
                
                cache.micro_L_set = L_MPO_strings(cache.micro_L_set, micro_sample, A, params, cache)
                cache.micro_R_set = R_MPO_strings(cache.micro_R_set, micro_sample, A, params, cache)
                local_∇L.+= loc.*∂MPO(micro_sample, cache.micro_L_set, cache.micro_R_set, params, cache)
            end
        end
    end
    return local_L, local_∇L
end

function Exact_MPO_gradient_two_body(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, l2::Matrix{<:Complex{<:AbstractFloat}}, basis, params::parameters)
    
    # Define ensemble averages:
    L∂L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    ΔLL::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    Z::eltype(A) = 0
    mean_local_Lindbladian::eltype(A) = 0

    # Preallocate cache:
    cache = set_workspace(A,params)

    for k in 1:params.dim
        for l in 1:params.dim

            #Initialize auxiliary arrays:
            local_L::ComplexF64 = 0
            local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
            l_int::ComplexF64 = 0
            cache.local_∇L_diagonal_coeff = 0

            sample = projector(basis[k],basis[l])
            cache.L_set = L_MPO_strings(cache.L_set, sample,A,params,cache)
            cache.R_set = R_MPO_strings(cache.R_set, sample,A,params,cache)

            ρ_sample = tr(cache.L_set[params.N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z += p_sample

            cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache)./ρ_sample

            #Calculate L∂L*:
            for j::UInt8 in 1:params.N
                #1-local part:
                lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,params,cache)
                local_L += lL
                local_∇L += l∇L
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

            local_L /=ρ_sample
            local_∇L/=ρ_sample

            #Add in diagonal part of the local derivative:
            local_∇L.+=cache.local_∇L_diagonal_coeff.*cache.Δ

            #Add in interaction terms:
            l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)
            local_L +=l_int
            local_∇L+=l_int*cache.Δ

            #Update L∂L* ensemble average:
            L∂L+=p_sample*local_L*conj(local_∇L)
    
            #Update ΔLL ensemble average:
            ΔLL+=p_sample*cache.Δ
    
            #Mean local Lindbladian:
            mean_local_Lindbladian += p_sample*local_L*conj(local_L)
        end
    end
    mean_local_Lindbladian/=Z
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate
    ΔLL.*=real(mean_local_Lindbladian)
    return (L∂L-ΔLL)/Z, real(mean_local_Lindbladian)
end
