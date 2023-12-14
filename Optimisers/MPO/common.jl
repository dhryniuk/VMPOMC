function one_body_Lindblad_term!(local_L::T, local_∇L::Array{T,3}, sample::Projector, micro_sample::Projector, j::UInt8, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 

    l1 = optimizer.l1
    A = optimizer.A
    params = optimizer.params
    cache = optimizer.workspace

    s::Matrix{T} = cache.dVEC_transpose[(sample.ket[j],sample.bra[j])]
    mul!(cache.bra_L_l1, s, l1)

    #Iterate over all 4 one-body vectorized basis Projectors:
    @inbounds for (i,state) in zip(1:4,TPSC)
        loc = cache.bra_L_l1[i]
        if loc!=0
            #Compute estimator:
            mul!(cache.loc_1, cache.L_set[j], @view(A[:,:,i]))
            mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N+1-j)])
            local_L += loc.*tr(cache.loc_2)
            
            #Compute derivative:
            if state==(sample.ket[j],sample.bra[j])   #check if diagonal
                cache.local_∇L_diagonal_coeff += loc
            else
                micro_sample.ket[j] = state[1]
                micro_sample.bra[j] = state[2]
                cache.micro_L_set = L_MPO_strings!(cache.micro_L_set, micro_sample, A, params, cache)
                cache.micro_R_set = R_MPO_strings!(cache.micro_R_set, micro_sample, A, params, cache)
                local_∇L.+= loc.*∂MPO(micro_sample, cache.micro_L_set, cache.micro_R_set, params, cache)

                #Restore original micro_sample to minimize allocations:
                micro_sample.ket[j] = sample.ket[j]
                micro_sample.bra[j] = sample.bra[j]
            end
        end
    end
    return local_L, local_∇L
end

function two_body_Lindblad_term!(local_L::T, local_∇L::Array{T,3}, sample::Projector, micro_sample::Projector, k::UInt8, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 

    l2 = optimizer.l2
    A = optimizer.A
    params = optimizer.params
    cache = optimizer.workspace

    s1::Matrix{T} = cache.dVEC_transpose[(sample.ket[k],sample.bra[k])]
    s2::Matrix{T} = cache.dVEC_transpose[(sample.ket[k+1],sample.bra[k+1])]
    cache.s = kron(s1,s2)
    mul!(cache.bra_L_l2, cache.s, l2)
    for (i::UInt8,state_i::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})
        for (j::UInt8,state_j::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})
            loc::T = cache.bra_L_l2[j+4*(i-1)]
            if loc!=0
                
                #Compute estimator:
                mul!(cache.loc_1, cache.L_set[k], @view(A[:,:,i]))
                mul!(cache.loc_2, cache.loc_1, @view(A[:,:,j]))
                mul!(cache.loc_3, cache.loc_2, cache.R_set[(params.N-k)])
                local_L += loc.*tr(cache.loc_3)

                micro_sample.ket[k] = state_i[1]
                micro_sample.bra[k] = state_i[2]
                micro_sample.ket[k+1] = state_j[1]
                micro_sample.bra[k+1] = state_j[2]
                
                cache.micro_L_set = L_MPO_strings!(cache.micro_L_set, micro_sample, A, params, cache)
                cache.micro_R_set = R_MPO_strings!(cache.micro_R_set, micro_sample, A, params, cache)
                local_∇L.+= loc.*∂MPO(micro_sample, cache.micro_L_set, cache.micro_R_set, params, cache)

                #Restore original micro_sample to minimize allocations:
                micro_sample.ket[k] = sample.ket[k]
                micro_sample.bra[k] = sample.bra[k]
                micro_sample.ket[k+1] = sample.ket[k+1]
                micro_sample.bra[k+1] = sample.bra[k+1]
            end
        end
    end
    return local_L, local_∇L
end