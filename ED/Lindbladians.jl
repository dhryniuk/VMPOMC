export DQIM, sparse_DQIM, DQIM_LRI, sparse_DQIM_LRI, DQIM_LRL, sparse_DQIM_LRL, DQIM_LRD, sparse_DQIM_LRD, make_one_body_Lindbladian

export DQIM_x, sparse_DQIM_x, DQIM_LRI_x, sparse_DQIM_LRI_x, DQIM_LRL_x, sparse_DQIM_LRL_x, DQIM_LRD_x, sparse_DQIM_LRD_x

export sparse_DQIMe

function DQIM(params::Parameters, boundary_conditions)
    """
    Returns the matrix Lindblad superoperator for the dissipative transverse field Ising model in 1D
    """

    H_ZZ= params.J*two_body_Hamiltonian_term(params, sz, sz, boundary_conditions)
    H_X = params.hx*one_body_Hamiltonian_term(params, sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = params.γ*one_body_Lindbladian_term(params, sm, boundary_conditions)

    return L_H + L_D
end

function sparse_DQIM(params::Parameters, boundary_conditions)

    H_ZZ= params.J*two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = params.hx*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    #H_Z = params.hz*one_body_Hamiltonian_term(params, sp_sz, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = params.γ*one_body_Lindbladian_term(params, sp_sm, boundary_conditions)

    return L_H + L_D
end

export sparse_DQIM_local_dephasing, sparse_DQIM_collective_dephasing

function sparse_DQIM_local_dephasing(params::Parameters, boundary_conditions)

    H_ZZ= params.J*two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = params.hx*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    #display(Matrix(L_H))
    L_D = params.γ*one_body_Lindbladian_term(params, sp_sm, boundary_conditions)
    L_D_dephasing = params.γ_d*one_body_Lindbladian_term(params, sp_sz, boundary_conditions)
    #display(Matrix(L_D_dephasing))

    return L_H + L_D + L_D_dephasing
end

function sparse_DQIM_collective_dephasing(params::Parameters, boundary_conditions)

    H_ZZ= params.J*two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = params.hx*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = params.γ*one_body_Lindbladian_term(params, sp_sm, boundary_conditions)
    #display(L_D)
    L_D_dephasing = params.γ_d*(collective_Lindbladian_term(sp_sz, params)+0.000001*one_body_Lindbladian_term(params, sp_sz, boundary_conditions))
    #display(L_D_dephasing)

    return L_H + L_D + L_D_dephasing
end

function DQIM_LRI(params::Parameters, boundary_conditions)
    """
    Returns the matrix Lindblad superoperator for the dissipative transverse field Ising model in 1D with long range coherent interactions
    """
    
    H_int = LR_two_body_Hamiltonian_term(params, sz, sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_int + H_X)
    L_D = one_body_Lindbladian_term(params, sm, boundary_conditions)

    return L_H + L_D
end

function sparse_DQIM_LRI(params::Parameters, boundary_conditions)
    
    H_int = LR_two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_int + H_X)
    L_D = one_body_Lindbladian_term(params, sp_sm, boundary_conditions)

    return L_H + L_D
end



function make_one_body_Lindbladian(H, Γ)
    L_H = -1im*(H⊗id - id⊗transpose(H))
    L_D = Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗id/2 - id⊗(transpose(Γ)*conj(Γ))/2
    return L_H + L_D
end

export one_body_Lindblad_term, one_body_Hamiltonian_term

function one_body_Hamiltonian_term(H)
    L_H = -1im*(H⊗id - id⊗transpose(H))
    return L_H
end

function one_body_Lindblad_term(Γ)
    L_D = Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗id/2 - id⊗(transpose(Γ)*conj(Γ))/2
    return L_D
end














export XXZ, sparse_XXZ

function XXZ(params::Parameters, boundary_conditions)
    H_int = two_body_Hamiltonian_term(params, sx, sx, boundary_conditions) + two_body_Hamiltonian_term(params, sy, sy, boundary_conditions) + params.α*two_body_Hamiltonian_term(params, sz, sz, boundary_conditions)
    #H_X = one_body_Hamiltonian_term(params, sz, boundary_conditions)
    L_Hint = vectorize_Hamiltonian(params, H_int)
    #L_H2 = vectorize_Hamiltonian(params, H_X)
    #display(real(L_H2))
    #L_D = one_body_Lindbladian_term(params, sm, boundary_conditions)
    L_D = params.γ*single_one_body_Lindbladian_term(1,params,sp) + params.γ*single_one_body_Lindbladian_term(params.N,params,sm)

    return L_Hint + L_D
end

function sparse_XXZ(params::Parameters, boundary_conditions)
    H_int = two_body_Hamiltonian_term(params, sp_sx, sp_sx, boundary_conditions) + two_body_Hamiltonian_term(params, sp_sy, sp_sy, boundary_conditions) + params.α*two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    #H_X = one_body_Hamiltonian_term(params, sz, boundary_conditions)
    L_Hint = vectorize_Hamiltonian(params, H_int)
    #L_H2 = vectorize_Hamiltonian(params, H_X)
    #display(real(L_H2))
    #L_D = one_body_Lindbladian_term(params, sm, boundary_conditions)
    L_D = params.γ*single_one_body_Lindbladian_term(1,params,sp_sp) + params.γ*single_one_body_Lindbladian_term(params.N,params,sp_sm)

    return L_Hint + L_D
end

export XYZ_Lindbald

function XYZ_Lindbald(params::Parameters, boundary_conditions)
    H_int = params.Jx*two_body_Hamiltonian_term(params, sx, sx, boundary_conditions) + params.Jy*two_body_Hamiltonian_term(params, sy, sy, boundary_conditions) + params.J*two_body_Hamiltonian_term(params, sz, sz, boundary_conditions)
    H_X = params.hx*one_body_Hamiltonian_term(params, sx, boundary_conditions)
    H_Z = params.hz*one_body_Hamiltonian_term(params, sz, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_int+H_X+H_Z)
    #L_H2 = vectorize_Hamiltonian(params, H_X)
    L_D = one_body_Lindbladian_term(params, sm, boundary_conditions)
    return L_H + L_D
end