#export DQIM, sparse_DQIM, LRInt_DQIM, sparse_LRInt_DQIM, LRDisp_DQIM, sparse_LRDisp_DQIM, make_one_body_Lindbladian
export DQIM, sparse_DQIM, sparse_DQIM_local_dephasing, sparse_DQIM_collective_dephasing, sparse_DQIM_LRI

function DQIM(params::Parameters, boundary_conditions)
    """
    Returns the matrix Lindblad superoperator for the dissipative transverse field Ising model in 1D
    """

    H_ZZ= params.J*two_body_Hamiltonian_term(params, sz, sz, boundary_conditions)
    H_X = params.hx*one_body_Hamiltonian_term(params, sx, boundary_conditions)
    #H_Z = params.hz*one_body_Hamiltonian_term(params, sz, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = params.γ*one_body_Lindbladian_term(sm, params)

    return L_H + L_D
end

function sparse_DQIM(params::Parameters, boundary_conditions)

    H_ZZ= params.J*two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = params.hx*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    #H_Z = params.hz*one_body_Hamiltonian_term(params, sp_sz, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = params.γ*one_body_Lindbladian_term(sp_sm, params)

    return L_H + L_D
end

function sparse_DQIM_local_dephasing(params::Parameters, boundary_conditions)

    H_ZZ= params.J*two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = params.hx*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    #display(Matrix(L_H))
    L_D = params.γ*one_body_Lindbladian_term(sp_sm, params)
    L_D_dephasing = params.γ_d*one_body_Lindbladian_term(sp_sz, params)
    #display(Matrix(L_D_dephasing))

    return L_H + L_D + L_D_dephasing
end

function sparse_DQIM_collective_dephasing(params::Parameters, boundary_conditions)

    H_ZZ= params.J*two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = params.hx*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = params.γ*one_body_Lindbladian_term(sp_sm, params)
    #display(L_D)
    L_D_dephasing = params.γ_d*(collective_Lindbladian_term(sp_sz, params)+0.0001*one_body_Lindbladian_term(sp_sz, params))
    #display(L_D_dephasing)

    return L_H + L_D + L_D_dephasing
end

export Lamacraft_local_dephasing
export Lamacraft_collective_dephasing

function Lamacraft_local_dephasing(params::Parameters, boundary_conditions)

    H_Z = params.hx*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_Z)
    L_Dm = params.γ*one_body_Lindbladian_term(sp_sm, params)
    L_Dp = params.γ*one_body_Lindbladian_term(sp_sp, params)
    L_D_dephasing = params.γ_d*one_body_Lindbladian_term(sp_sz, params)

    return L_H + L_Dm + L_Dp + L_D_dephasing
end

function Lamacraft_collective_dephasing(params::Parameters, boundary_conditions)

    H_Z = params.hx*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_Z)
    L_Dm = params.γ*one_body_Lindbladian_term(sp_sm, params)
    L_Dp = params.γ*one_body_Lindbladian_term(sp_sp, params)
    L_D_dephasing = params.γ_d*collective_Lindbladian_term(sp_sz, params)

    return L_H + L_Dm + L_Dp + L_D_dephasing
end

function sparse_DQIM_LRI(params::Parameters, boundary_conditions)

    H_ZZ= params.J*LR_two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = params.hx*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    #H_Z = params.hz*one_body_Hamiltonian_term(params, sp_sz, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = params.γ*one_body_Lindbladian_term(sp_sm, params)

    return L_H + L_D
end