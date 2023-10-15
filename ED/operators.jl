

function one_body_Hamiltonian_term(params::Parameters, op1::Matrix{ComplexF64}, boundary_conditions)
    # vector of operators: [op1, id, ...]
    ops = fill(id, params.N)
    ops[1] = op1

    H::Matrix{ComplexF64} = zeros(ComplexF64, 2^params.N, 2^params.N)
    for _ in 1:params.N
        H += foldl(⊗, ops)
        ops = circshift(ops,1)
    end

    return H
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

    return H
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

    return H
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

    return H
end

function vectorize_Hamiltonian(params::Parameters, H::Matrix{ComplexF64})
    Id::Matrix{ComplexF64} = foldl(⊗, fill(id, params.N))
    return -1im*(H⊗Id - Id⊗transpose(H))
end

function vectorize_Hamiltonian(params::Parameters, H::SparseMatrixCSC{ComplexF64, Int64})
    Id::SparseMatrixCSC{ComplexF64, Int64} = foldl(⊗, fill(id, params.N))
    return -1im*(H⊗Id - Id⊗transpose(H))
end

function one_body_Lindbladian_term(op1::Matrix{ComplexF64}, params::Parameters)
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
    return L_D
end

function one_body_Lindbladian_term(op1::SparseMatrixCSC{ComplexF64, Int64}, params::Parameters)
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
    return L_D
end

function collective_Lindbladian_term(op1::SparseMatrixCSC{ComplexF64, Int64}, params::Parameters)
    # vector of operators: [op1, op1, ...]
    ops = fill(sp_id, params.N)
    ops[1] = op1

    Id::SparseMatrixCSC{ComplexF64, Int64} = foldl(⊗, fill(sp_id, params.N))

    L_D::SparseMatrixCSC{ComplexF64, Int64} = spzeros(ComplexF64, 2^(2*params.N), 2^(2*params.N))

    Γ = spzeros(ComplexF64, 2^(params.N), 2^(params.N))
    for _ in 1:params.N
        Γ += foldl(⊗, ops)
        ops = circshift(ops,1)
    end

    L_D = Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗Id/2 - Id⊗(transpose(Γ)*conj(Γ))/2

    return L_D
end

function wrong_collective_Lindbladian_term(op1::SparseMatrixCSC{ComplexF64, Int64}, params::Parameters)
    # vector of operators: [op1, op1, ...]
    ops = fill(op1, params.N)

    Id::SparseMatrixCSC{ComplexF64, Int64} = foldl(⊗, fill(sp_id, params.N))

    L_D::SparseMatrixCSC{ComplexF64, Int64} = spzeros(ComplexF64, 2^(2*params.N), 2^(2*params.N))

    Γ = foldl(⊗, ops)
    L_D += Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗Id/2 - Id⊗(transpose(Γ)*conj(Γ))/2

    return L_D
end

function LR_two_body_Hamiltonian_term(params::Parameters, op1::SparseMatrixCSC{ComplexF64, Int64}, op2::SparseMatrixCSC{ComplexF64, Int64}, boundary_conditions)
    H::SparseMatrixCSC{ComplexF64, Int64} = zeros(ComplexF64, 2^params.N, 2^params.N)
    N_K = calculate_Kac_norm(params)
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

    return H
end