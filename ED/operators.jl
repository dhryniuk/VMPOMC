

function one_body_Hamiltonian_term(params::Parameters, op1::Matrix{ComplexF64}, boundary_conditions)
    # vector of operators: [op1, id, ...]
    ops = fill(id, params.N)
    ops[1] = op1

    H::Matrix{ComplexF64} = zeros(ComplexF64, 2^params.N, 2^params.N)
    for _ in 1:params.N
        H += foldl(⊗, ops)
        ops = circshift(ops,1)
    end

    return H#params.h*H
end

function one_body_Hamiltonian_term(params::Parameters, op1::SparseMatrixCSC{ComplexF64, Int64}, boundary_conditions)
    # vector of operators: [op1, id, ...]
    ops = fill(sp_id, params.N)
    ops[1] = op1

    H::SparseMatrixCSC{ComplexF64, Int64} = spzeros(ComplexF64, 2^params.N, 2^params.N)
    for _ in 1:params.N
        H += foldl(⊗, ops)
        ops = circshift(ops,1)
    end

    return H#params.h*H
end

function two_body_Hamiltonian_term(params::Parameters, op1::Matrix{ComplexF64}, op2::Matrix{ComplexF64}, boundary_conditions)
    # vector of operators: [op1, op2, id, ...]
    ops = fill(id, params.N)
    ops[1] = op1
    ops[2] = op2

    H::Matrix{ComplexF64} = zeros(ComplexF64, 2^params.N, 2^params.N)
    for _ in 1:params.N-1
        H += foldl(⊗, ops)
        ops = circshift(ops,1)
    end
    if boundary_conditions=="periodic"
        H += foldl(⊗, ops)
    end

    return H#params.J*H
end

function two_body_Hamiltonian_term(params::Parameters, op1::SparseMatrixCSC{ComplexF64, Int64}, op2::SparseMatrixCSC{ComplexF64, Int64}, boundary_conditions)
    # vector of operators: [op1, op2, id, ...]
    ops = fill(sp_id, params.N)
    ops[1] = op1
    ops[2] = op2

    H::SparseMatrixCSC{ComplexF64, Int64} = zeros(ComplexF64, 2^params.N, 2^params.N)
    for _ in 1:params.N-1
        H += foldl(⊗, ops)
        ops = circshift(ops,1)
    end
    if boundary_conditions=="periodic"
        H += foldl(⊗, ops)
    end

    return H#params.J*H
end

function vectorize_Hamiltonian(params::Parameters, H::Matrix{ComplexF64})
    Id::Matrix{ComplexF64} = foldl(⊗, fill(id, params.N))
    return -1im*(H⊗Id - Id⊗transpose(H))
end

function vectorize_Hamiltonian(params::Parameters, H::SparseMatrixCSC{ComplexF64, Int64})
    Id::SparseMatrixCSC{ComplexF64, Int64} = foldl(⊗, fill(id, params.N))
    return -1im*(H⊗Id - Id⊗transpose(H))
end

function one_body_Lindbladian_term(params::Parameters, op1::Matrix{ComplexF64}, boundary_conditions)
    # vector of operators: [op1, id, ...]
    ops = fill(id, params.N)
    ops[1] = op1

    Id::Matrix{ComplexF64} = foldl(⊗, fill(id, params.N))

    L_D::Matrix{ComplexF64} = zeros(ComplexF64, 2^(2*params.N), 2^(2*params.N))
    for _ in 1:params.N
        Γ = foldl(⊗, ops)
        ops = circshift(ops,1)
        L_D += Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗Id/2 - Id⊗(transpose(Γ)*conj(Γ))/2
    end
    return params.γ*L_D
end

function one_body_Lindbladian_term(params::Parameters, op1::SparseMatrixCSC{ComplexF64, Int64}, boundary_conditions)
    # vector of operators: [op1, id, ...]
    ops = fill(sp_id, params.N)
    ops[1] = op1

    Id::SparseMatrixCSC{ComplexF64, Int64} = foldl(⊗, fill(sp_id, params.N))

    L_D::SparseMatrixCSC{ComplexF64, Int64} = spzeros(ComplexF64, 2^(2*params.N), 2^(2*params.N))
    for _ in 1:params.N
        Γ = foldl(⊗, ops)
        ops = circshift(ops,1)
        L_D += Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗Id/2 - Id⊗(transpose(Γ)*conj(Γ))/2
    end
    return params.γ*L_D
end

function LR_two_body_Hamiltonian_term(params::Parameters, op1::Matrix{ComplexF64}, op2::Matrix{ComplexF64}, boundary_conditions)
    H::Matrix{ComplexF64} = zeros(ComplexF64, 2^params.N, 2^params.N)
    if boundary_conditions=="periodic" && mod(params.N,2)==1
        N_K = calculate_Kac_norm(convert(Int64,floor(params.N/2)), params.α)
        for k in 1:convert(Int16,floor(params.N/2))
            ops = fill(id, params.N)
            ops[1] = op1
            ops[1+k] = op2
            dist = k^params.α
            for _ in 1:params.N
                H += 1/(dist*N_K)*foldl(⊗, ops)
                ops = circshift(ops,1)
            end
        end
    elseif boundary_conditions=="open"
        for k in 1:params.N-1
            ops = fill(id, params.N)
            ops[1] = op1
            ops[1+k] = op2
            dist = k^params.α
            for _ in 1:params.N-k
                H += 1/(dist*params.N_K)*foldl(⊗, ops)
                ops = circshift(ops,1)
            end
        end
    else
        return error("UNRECOGNIZED BOUNDARY CONDITIONS")
    end

    return params.J*H
end

function LR_two_body_Hamiltonian_term(params::Parameters, op1::SparseMatrixCSC{ComplexF64, Int64}, op2::SparseMatrixCSC{ComplexF64, Int64}, boundary_conditions)
    H::SparseMatrixCSC{ComplexF64, Int64} = spzeros(ComplexF64, 2^params.N, 2^params.N)
    if boundary_conditions=="periodic" && mod(params.N,2)==1
        N_K = calculate_Kac_norm(convert(Int64,floor(params.N/2)), params.α)
        for k in 1:convert(Int16,floor(params.N/2))
            ops = fill(sp_id, params.N)
            ops[1] = op1
            ops[1+k] = op2
            dist = k^params.α
            for _ in 1:params.N
                H += 1/(dist*N_K)*foldl(⊗, ops)
                ops = circshift(ops,1)
            end
        end
    elseif boundary_conditions=="open"
        for k in 1:params.N-1
            ops = fill(sp_id, params.N)
            ops[1] = op1
            ops[1+k] = op2
            dist = k^params.α
            for _ in 1:params.N-k
                H += 1/(dist*params.N_K)*foldl(⊗, ops)
                ops = circshift(ops,1)
            end
        end
    else
        return error("UNRECOGNIZED BOUNDARY CONDITIONS")
    end

    return params.J*H
end

function LR_Lindbladian_term(params::Parameters, op1::Matrix{ComplexF64}, op2::Matrix{ComplexF64}, boundary_conditions)
    Id = foldl(⊗, fill(id, params.N))
    L_D = zeros(ComplexF64, 2^(2*params.N), 2^(2*params.N))

    if boundary_conditions=="open"
        for j in 1:params.N

            # prepare vector of operators: [σ-, id, ...]
            dissip_term_ops_1 = fill(id, params.N)
            dissip_term_ops_1[1] =op1
            dissip_term_ops_2 = fill(id, params.N)
            dissip_term_ops_2[j] = op2

            # calculate distance:
            dist = j^params.α

            for _ in 1:params.N+1-j

                Γ1 = foldl(⊗, dissip_term_ops_1)
                Γ2 = foldl(⊗, dissip_term_ops_2)
                dissip_term_ops_1 = circshift(dissip_term_ops_1,1)
                dissip_term_ops_2 = circshift(dissip_term_ops_2,1)

                L_D += params.γ/(dist*params.N_K) * ( Γ1⊗transpose(Γ2) - (Γ2*Γ1)⊗Id/2 - Id⊗(transpose(Γ1)*transpose(Γ2))/2 )
            end
        end
    elseif boundary_conditions=="periodic" && mod(params.N,2)==1

        # recalculate Kac normalization constant:
        N_K = calculate_Kac_norm(convert(Int64,floor(params.N/2)), params.α, offset=1)

        for j in 1:convert(Int64,floor(params.N/2))+1

            # prepare vectors of operators: [σ-, id, ...]
            dissip_term_ops_1 = fill(id, params.N)
            dissip_term_ops_1[1] = op1
            dissip_term_ops_2 = fill(id, params.N)
            dissip_term_ops_2[j] = op2

            # calculate distance:
            dist = j^params.α

            for _ in 1:params.N#+1-j
                Γ1 = foldl(⊗, dissip_term_ops_1)
                Γ2 = foldl(⊗, dissip_term_ops_2)
                dissip_term_ops_1 = circshift(dissip_term_ops_1,1)
                dissip_term_ops_2 = circshift(dissip_term_ops_2,1)
                
                L_D += params.γ/(dist*N_K) * ( Γ1⊗transpose(Γ2) - (Γ2*Γ1)⊗Id/2 - Id⊗(transpose(Γ1)*transpose(Γ2))/2 )
            end
        end
    else
        return error("UNRECOGNIZED BOUNDARY CONDITIONS")
    end

    return L_D
end

function LR_Lindbladian_term(params::Parameters, op1::SparseMatrixCSC{ComplexF64, Int64}, op2::SparseMatrixCSC{ComplexF64, Int64}, boundary_conditions)
    Id = foldl(⊗, fill(sp_id, params.N))
    L_D = spzeros(ComplexF64, 2^(2*params.N), 2^(2*params.N))

    if boundary_conditions=="open"
        for j in 1:params.N

            # prepare vector of operators: [σ-, id, ...]
            dissip_term_ops_1 = fill(sp_id, params.N)
            dissip_term_ops_1[1] = op1
            dissip_term_ops_2 = fill(sp_id, params.N)
            dissip_term_ops_2[j] = op2

            # calculate distance:
            dist = j^params.α

            for _ in 1:params.N+1-j

                Γ1 = foldl(⊗, dissip_term_ops_1)
                Γ2 = foldl(⊗, dissip_term_ops_2)
                dissip_term_ops_1 = circshift(dissip_term_ops_1,1)
                dissip_term_ops_2 = circshift(dissip_term_ops_2,1)

                L_D += params.γ/(dist*params.N_K) * ( Γ1⊗transpose(Γ2) - (Γ2*Γ1)⊗Id/2 - Id⊗(transpose(Γ1)*transpose(Γ2))/2 )
            end
        end
    elseif boundary_conditions=="periodic" && mod(params.N,2)==1

        # recalculate Kac normalization constant:
        N_K = calculate_Kac_norm(convert(Int64,floor(params.N/2)), params.α, offset=1)

        for j in 1:convert(Int64,floor(params.N/2))+1

            # prepare vectors of operators: [σ-, id, ...]
            dissip_term_ops_1 = fill(sp_id, params.N)
            dissip_term_ops_1[1] = op1
            dissip_term_ops_2 = fill(sp_id, params.N)
            dissip_term_ops_2[j] = op2

            # calculate distance:
            dist = j^params.α

            for _ in 1:params.N#+1-j
                Γ1 = foldl(⊗, dissip_term_ops_1)
                Γ2 = foldl(⊗, dissip_term_ops_2)
                dissip_term_ops_1 = circshift(dissip_term_ops_1,1)
                dissip_term_ops_2 = circshift(dissip_term_ops_2,1)
                
                L_D += params.γ/(dist*N_K) * ( Γ1⊗transpose(Γ2) - (Γ2*Γ1)⊗Id/2 - Id⊗(transpose(Γ1)*transpose(Γ2))/2 )
            end
        end
    else
        return error("UNRECOGNIZED BOUNDARY CONDITIONS")
    end

    return L_D
end




function single_one_body_Lindbladian_term(pos::Int, params::Parameters, op1::Matrix{ComplexF64})
    # vector of operators: [op1, id, ...]
    ops = fill(id, params.N)
    ops[pos] = op1

    Id::Matrix{ComplexF64} = foldl(⊗, fill(id, params.N))

    L_D::Matrix{ComplexF64} = zeros(ComplexF64, 2^(2*params.N), 2^(2*params.N))
    Γ = foldl(⊗, ops)
    L_D += Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗Id/2 - Id⊗(transpose(Γ)*conj(Γ))/2
    return L_D
end

function single_one_body_Lindbladian_term(pos::Int, params::Parameters, op1::SparseMatrixCSC{ComplexF64, Int64})
    # vector of operators: [op1, id, ...]
    ops = fill(sp_id, params.N)
    ops[pos] = op1

    Id::SparseMatrixCSC{ComplexF64, Int64} = foldl(⊗, fill(sp_id, params.N))

    L_D::SparseMatrixCSC{ComplexF64, Int64} = spzeros(ComplexF64, 2^(2*params.N), 2^(2*params.N))
    Γ = foldl(⊗, ops)
    L_D += Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗Id/2 - Id⊗(transpose(Γ)*conj(Γ))/2
    return L_D
end