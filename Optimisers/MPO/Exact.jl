export Exact_MPO_gradient#, one_body_Lindblad_term


function bad_one_body_Lindblad_term(sample::projector, j::UInt8, l1::Matrix{<:Complex{<:AbstractFloat}}, A::Array{<:Complex{<:AbstractFloat},3}, 
    L_set::Vector{<:Matrix{<:Complex{<:AbstractFloat}}}, R_set::Vector{<:Matrix{<:Complex{<:AbstractFloat}}}, params::parameters, AUX::workspace)
    
    local_L::eltype(A) = 0
    local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
    s::Matrix{eltype(A)} = dVEC_transpose[(sample.ket[j],sample.bra[j])]
    mul!(AUX.bra_L, s, conj.(l1))

    #Iterate over all 4 one-body vectorized basis projectors:
    @inbounds for (i,state) in zip(1:4,TPSC)
        loc = AUX.bra_L[i]
        if loc!=0
            #Compute estimator:
            mul!(AUX.loc_1, L_set[j], @view(A[:,:,i]))
            mul!(AUX.loc_2, AUX.loc_1, R_set[(params.N+1-j)])
            local_L += loc.*tr(AUX.loc_2)

            #display(AUX.R_set)

            #Compute derivative:
            micro_sample::projector = projector(sample)
            micro_sample.ket[j] = state[1]
            micro_sample.bra[j] = state[2]
            AUX.micro_L_set = L_MPO_strings(AUX.micro_L_set, micro_sample, A, params, AUX)
            AUX.micro_R_set = R_MPO_strings(AUX.micro_R_set, micro_sample, A, params, AUX)
            local_∇L.+= loc.*∂MPO(micro_sample, AUX.micro_L_set, AUX.micro_R_set, params, AUX)
        end
    end
    return local_L, local_∇L
end


### NEED TO TREAT DIAGONAL SEPARATELY
function one_body_Lindblad_term(sample::projector, j::UInt8, l1::Matrix{<:Complex{<:AbstractFloat}}, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, AUX::workspace)
    
    local_L::eltype(A) = 0
    local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
    s::Matrix{eltype(A)} = dVEC_transpose[(sample.ket[j],sample.bra[j])]
    #mul!(AUX.bra_L, s, conj.(l1))
    mul!(AUX.bra_L, s, l1)

    #Iterate over all 4 one-body vectorized basis projectors:
    @inbounds for (i,state) in zip(1:4,TPSC)
        loc = AUX.bra_L[i]
        if loc!=0
            #Compute estimator:
            mul!(AUX.loc_1, AUX.L_set[j], @view(A[:,:,i]))
            mul!(AUX.loc_2, AUX.loc_1, AUX.R_set[(params.N+1-j)])
            local_L += loc.*tr(AUX.loc_2)
            
            #Compute derivative:
            if state==(sample.ket[j],sample.bra[j])   #check if diagonal
                AUX.local_∇L_diagonal_coeff += loc
            else
                micro_sample::projector = projector(sample)
                micro_sample.ket[j] = state[1]
                micro_sample.bra[j] = state[2]
                AUX.micro_L_set = L_MPO_strings(AUX.micro_L_set, micro_sample, A, params, AUX)
                AUX.micro_R_set = R_MPO_strings(AUX.micro_R_set, micro_sample, A, params, AUX)
                local_∇L.+= loc.*∂MPO(micro_sample, AUX.micro_L_set, AUX.micro_R_set, params, AUX)
            end
        end
    end
    return local_L, local_∇L
end

function Lindblad_Ising_interaction_energy(sample::projector, boundary_conditions, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters)
    l_int::eltype(A)=0
    for j::UInt8 in 1:params.N-1
        l_int_ket = (2*sample.ket[j]-1)*(2*sample.ket[j+1]-1)
        l_int_bra = (2*sample.bra[j]-1)*(2*sample.bra[j+1]-1)
        l_int += l_int_ket-l_int_bra
    end
    if boundary_conditions=="periodic"
        l_int_ket = (2*sample.ket[params.N]-1)*(2*sample.ket[1]-1)
        l_int_bra = (2*sample.bra[params.N]-1)*(2*sample.bra[1]-1)
        l_int += l_int_ket-l_int_bra
    end
    return 1.0im*params.J*l_int
    #return -1.0im*params.J*l_int
end

"""
function two_body_Lindblad_term(params::parameters, sample::density_matrix, k::UInt16, l2::Matrix, A::Array{ComplexF64,3}, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})
    local_L::ComplexF64 = 0
    local_∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)

    s1::Matrix{ComplexF64} = dVEC_transpose[(sample.ket[k],sample.bra[k])]
    s2::Matrix{ComplexF64} = dVEC_transpose[(sample.ket[k+1],sample.bra[k+1])]
    s = kron(s1,s2)
    bra_L::Matrix{ComplexF64} = s*conj(l2)
    #@inbounds for i::UInt16 in 1:4, j::UInt16 in 1:4
    for (i::UInt16,state_i::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})
        for (j::UInt16,state_j::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})

            loc::ComplexF64 = bra_L[j+4*(i-1)]
            if loc!=0
                local_L += loc*tr( (L_set[k::UInt16]*A[:,:,i]::Matrix{ComplexF64})*A[:,:,j]::Matrix{ComplexF64}*R_set[(params.N-k)::Int64])
                micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                micro_sample.ket[k] = state_i[1]
                micro_sample.bra[k] = state_i[2]
                micro_sample.ket[k+1] = state_j[1]
                micro_sample.bra[k+1] = state_j[2]
                
                micro_L_set = L_MPO_strings(params, micro_sample, A)
                micro_R_set = R_MPO_strings(params, micro_sample, A)
                local_∇L+= loc*∂MPO(params, micro_sample,micro_L_set,micro_R_set)
            end
        end
    end
    return local_L, local_∇L
end

function boundary_two_body_Lindblad_term(params::parameters, sample::density_matrix, l2::Matrix, A::Array{ComplexF64,3}, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})

    #Need to find middle string, by inverting the first tensor A:
    M = inv(A[:,:,dINDEX[(sample.ket[1],sample.bra[1])]])*L_set[params.N]

    local_L::ComplexF64 = 0
    local_∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)

    s1::Matrix{ComplexF64} = dVEC_transpose[(sample.ket[params.N],sample.bra[params.N])]
    s2::Matrix{ComplexF64} = dVEC_transpose[(sample.ket[1],sample.bra[1])]
    s = kron(s1,s2)
    bra_L::Matrix{ComplexF64} = s*conj(l2)
    #@inbounds for i::UInt16 in 1:4, j::UInt16 in 1:4
    for (i::UInt16,state_i::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})
        for (j::UInt16,state_j::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})

            loc::ComplexF64 = bra_L[j+4*(i-1)]
            if loc!=0
                local_L += loc*tr( M*A[:,:,i]*A[:,:,j] )
                #local_L += loc*MPO(params,sample,A)
                micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                micro_sample.ket[1] = state_i[1]
                micro_sample.bra[1] = state_i[2]
                micro_sample.ket[params.N] = state_j[1]
                micro_sample.bra[params.N] = state_j[2]
                
                micro_L_set = L_MPO_strings(params, micro_sample, A)
                micro_R_set = R_MPO_strings(params, micro_sample, A)
                local_∇L+= loc*∂MPO(params, micro_sample, micro_L_set, micro_R_set)
            end
        end
    end
    return local_L, local_∇L
end
"""

function bad_Exact_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, basis, params::parameters)
    
    # Define ensemble averages:
    L∇L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    ΔLL::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    Z::eltype(A) = 0
    mean_local_Lindbladian::eltype(A) = 0

    # Preallocate auxiliary arrays:
    AUX = set_workspace(A,params)

    for k in 1:params.dim
        for l in 1:params.dim
            sample = projector(basis[k],basis[l])
            AUX.L_set = L_MPO_strings(AUX.L_set, sample,A,params,AUX)
            AUX.R_set = R_MPO_strings(AUX.R_set, sample,A,params,AUX)

            ρ_sample = tr(AUX.L_set[params.N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z += p_sample

            local_L::eltype(A) = 0
            local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
            l_int::eltype(A) = 0

            AUX.L_set[1] = Matrix{eltype(A)}(I, params.χ, params.χ)

            #Calculate L∂L*:
            for j::UInt8 in 1:params.N
                #1-local part:
                lL, l∇L = bad_body_Lindblad_term(sample,j,l1,A,AUX.L_set,AUX.R_set,params,AUX)
                local_L += lL
                local_∇L += l∇L
                
                #Update L_set:
                mul!(AUX.L_set[j+1], AUX.L_set[j], @view(A[:,:,1+2*sample.ket[j]+sample.bra[j]]))
            end

            l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)

            local_L /=ρ_sample
            local_∇L/=ρ_sample
    
            AUX.Δ_MPO_sample = ∂MPO(sample, AUX.L_set, AUX.R_set, params, AUX)./ρ_sample

            #Add in interaction terms:
            local_L +=l_int
            local_∇L+=l_int*AUX.Δ_MPO_sample

            L∇L+=p_sample*local_L*conj(local_∇L)
    
            #ΔLL:
            local_Δ=p_sample*conj(AUX.Δ_MPO_sample)
            ΔLL+=local_Δ
    
            #Mean local Lindbladian:
            mean_local_Lindbladian += p_sample*local_L*conj(local_L)
        end
    end
    mean_local_Lindbladian/=Z
    ΔLL*=mean_local_Lindbladian
    return (L∇L-ΔLL)/Z, real(mean_local_Lindbladian)
end

function Exact_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, basis, params::parameters)
    
    # Define ensemble averages:
    L∂L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    ΔLL::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    Z::eltype(A) = 0
    mean_local_Lindbladian::eltype(A) = 0

    # Preallocate auxiliary arrays:
    AUX = set_workspace(A,params)

    for k in 1:params.dim
        for l in 1:params.dim

            #Initialize auxiliary arrays:
            local_L::ComplexF64 = 0
            local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
            l_int::ComplexF64 = 0
            AUX.local_∇L_diagonal_coeff = 0

            sample = projector(basis[k],basis[l])
            AUX.L_set = L_MPO_strings(AUX.L_set, sample,A,params,AUX)
            AUX.R_set = R_MPO_strings(AUX.R_set, sample,A,params,AUX)

            ρ_sample = tr(AUX.L_set[params.N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z += p_sample

            AUX.Δ = ∂MPO(sample, AUX.L_set, AUX.R_set, params, AUX)./ρ_sample

            #Calculate L∂L*:
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

            #Update L∂L* ensemble average:
            L∂L+=p_sample*local_L*conj(local_∇L)
    
            #Update ΔLL ensemble average:
            ΔLL+=p_sample*AUX.Δ
    
            #Mean local Lindbladian:
            mean_local_Lindbladian += p_sample*local_L*conj(local_L)
        end
    end
    mean_local_Lindbladian/=Z
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate
    ΔLL.*=real(mean_local_Lindbladian)
    return (L∂L-ΔLL)/Z, real(mean_local_Lindbladian)
end


"""
export Two_body_Exact_MPO_gradient

function Two_body_Exact_MPO_gradient(params::parameters, A::Array{ComplexF64}, 
    l1::Matrix{ComplexF64}, l2::Matrix{ComplexF64}, basis)

    L∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    Z::ComplexF64 = 0
    mean_local_Lindbladian::ComplexF64 = 0

    for k in 1:params.dim
        for l in 1:params.dim
            sample = density_matrix(1,basis[k],basis[l])
            L_set = L_MPO_strings(params, sample, A)
            R_set = R_MPO_strings(params, sample, A)

            ρ_sample = tr(L_set[params.N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z += p_sample

            local_L::ComplexF64 = 0
            local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
            l_int::ComplexF64 = 0

            L_set = [ Matrix{ComplexF64}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
            L = Matrix{ComplexF64}(I, params.χ, params.χ)
            L_set[1] = L

            #L∇L*:
            for j::UInt16 in 1:params.N

                #1-local part:
                lL, l∇L = one_body_Lindblad_term(params,sample,j,l1,A,L_set,R_set)
                #lL, l∇L = one_body_Lindblad_term(params,sample_ket,sample_bra,j,l1,A,L_set,R_set)
                local_L += lL
                local_∇L += l∇L

                L*=A[:,:,1+2*sample.ket[j]+sample.bra[j]]
                L_set[j+1] = L
            end
            for j::UInt16 in 1:params.N-1
                lL, l∇L = two_body_Lindblad_term(params,sample,j,l2,A,L_set,R_set)
                local_L += lL
                local_∇L += l∇L
            end
            if params.N>2
                lL, l∇L = boundary_two_body_Lindblad_term(params,sample,l2,A,L_set,R_set)
                local_L += lL
                local_∇L += l∇L
            end

            #l_int = Lindblad_Ising_interaction_energy(params, sample, "periodic")

            local_L /=ρ_sample
            local_∇L/=ρ_sample
    
            Δ_MPO_sample = ∂MPO(params, sample, L_set, R_set)/ρ_sample
            #Δ_MPO_sample = derv_MPO(params, sample_ket, sample_bra, L_set, R_set)/ρ_sample
    
            #Add in interaction terms:
            local_L +=l_int
            local_∇L+=l_int*Δ_MPO_sample
    
            L∇L+=p_sample*local_L*conj(local_∇L)
            #L∇L+=p_sample*conj(local_L)*local_∇L
    
            #ΔLL:
            local_Δ=p_sample*conj(Δ_MPO_sample)
            #local_Δ=p_sample*Δ_MPO_sample
            ΔLL+=local_Δ
    
            #Mean local Lindbladian:
            mean_local_Lindbladian += p_sample*local_L*conj(local_L)
        end
    end
    mean_local_Lindbladian/=Z
    ΔLL*=mean_local_Lindbladian
    return (L∇L-ΔLL)/Z, real(mean_local_Lindbladian)
end
"""