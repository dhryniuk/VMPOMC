#using LinearAlgebra
#using NPZ

id = [1 0; 0 1]
sx = [0 1; 1 0]
sy = [0 -1im; 1im 0]
sz = [1 0; 0 -1]
sp = (sx+1im*sy)/2
sm = (sx-1im*sy)/2

⊗(x,y) = kron(x,y)

function make_Liouvillian(H, Γ)
    L_H = -1im*(H⊗id - id⊗transpose(H))
    L_D = Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗id/2 - id⊗(transpose(Γ)*conj(Γ))/2
    return L_H + L_D
end

function TransverseFieldIsing(;N,h)
    id = [1 0; 0 1]
    σˣ = [0 1; 1 0]
    σᶻ = [1 0; 0 -1]
    
    # vector of operators: [σᶻ, σᶻ, id, ...]
    first_term_ops = fill(id, N)
    first_term_ops[1] = σᶻ
    first_term_ops[2] = σᶻ
    
    # vector of operators: [σˣ, id, ...]
    second_term_ops = fill(id, N)
    second_term_ops[1] = σˣ
    
    H = zeros(Int, 2^N, 2^N)
    for i in 1:N#-1     #-1 for OBCs
        # tensor multiply all operators
        H -= foldl(⊗, first_term_ops)
        # cyclic shift the operators
        first_term_ops = circshift(first_term_ops,1)
    end
    
    for i in 1:N
        H -= h*foldl(⊗, second_term_ops)
        second_term_ops = circshift(second_term_ops,1)
    end
    H
end

function DQIM(N,J,h,γ)
    # vector of operators: [σᶻ, σᶻ, id, ...]
    first_term_ops = fill(id, N)
    if J!=0
        first_term_ops[1] = sz
        first_term_ops[2] = sz
    end
    
    # vector of operators: [σˣ, id, ...]
    second_term_ops = fill(id, N)
    second_term_ops[1] = sx
    
    H = zeros(ComplexF64, 2^N, 2^N)
    for i in 1:N
        H += J*foldl(⊗, first_term_ops)
        first_term_ops = circshift(first_term_ops,1)
    end
    
    for i in 1:N
        H += h*foldl(⊗, second_term_ops)
        second_term_ops = circshift(second_term_ops,1)
    end

    Id = foldl(⊗, fill(id, N))
    L_H = -1im*(H⊗Id - Id⊗transpose(H))

    # vector of operators: [σ-, id, ...]
    dissip_term_ops = fill(id, N)
    dissip_term_ops[1] = sm

    L_D = zeros(ComplexF64, 2^(2*N), 2^(2*N))
    for i in 1:N
        Γ = γ*foldl(⊗, dissip_term_ops)
        dissip_term_ops = circshift(dissip_term_ops,1)
        L_D += Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗Id/2 - Id⊗(transpose(Γ)*conj(Γ))/2
    end

    return L_H + L_D
end

function construct_vec_density_matrix_basis(N)
    one_body = []
    for i in [0,1]
        for j in [0,1]
            push!(one_body,density_matrix(1,[i],[j]))
        end
    end
    BASIS=one_body
    for k in 2:N
        NEW_BASIS=[]
        for state in BASIS
            for state2 in one_body
                new_state = density_matrix(1,vcat(state.ket,state2.ket),vcat(state.bra,state2.bra))
                push!(NEW_BASIS,new_state)
            end
        end
        BASIS=NEW_BASIS
    end
    return BASIS
end


function own_version_DQIM(N,J,h,γ,basis)

    # calculate one-body lindbladian:
    l = make_Liouvillian(h*sx, γ*sm)

    # vector of operators:
    one_body_term_ops::Array{Matrix{ComplexF64}} = fill(id⊗id, N)
    one_body_term_ops[1] = l
    
    L = zeros(ComplexF64, 4^N, 4^N)
    for i in 1:N
        L += foldl(⊗, one_body_term_ops)
        one_body_term_ops = circshift(one_body_term_ops,1)
    end
    println(length(basis))
    # interaction:
    for i in 1:length(basis)
        α=basis[i].ket
        β=basis[i].bra

        l_int=0
        for k in 1:N
            l_int_α = (2*α[k]-1)*(2*α[mod(k-2,N)+1]-1)
            l_int_β = (2*β[k]-1)*(2*β[mod(k-2,N)+1]-1)
            l_int += -1.0im*J*(l_int_α-l_int_β)
        end

        #index = dim*(i-1)+j
        #index = dim*(j-1)+i
        #L[index,index] += l_int
        L[i,i] += l_int
    end

    #= # interaction:
    for i in 1:dim
        α=basis[i]
        for j in 1:dim
            β=basis[j]

            l_int=0
            for k in 1:N
                l_int_α = (2*α[k]-1)*(2*α[mod(k-2,N)+1]-1)
                l_int_β = (2*β[k]-1)*(2*β[mod(k-2,N)+1]-1)
                l_int += -1.0im*J*(l_int_α-l_int_β)
            end

            index = dim*(i-1)+j
            #index = dim*(j-1)+i
            L[index,index] += l_int
        end
    end =#

    return L
end

function normalize_own_density_matrix(ρ,basis)
    norm=0
    for i in 1:length(basis)
        state=basis[i]
        if state.ket==state.bra
            norm+=ρ[i]
        end
    end
    ρ./=norm
    return ρ
end

function own_z_magnetization(ρ,N,basis)
    function average_spins(state)
        average_spin=0
        for j in 1:N
            average_spin+=2*state[j]-1
        end
        return average_spin
    end

    M_z=0
    ρ_flat = reshape(ρ,length(basis))
    for i in 1:length(basis)
        state=basis[i]
        if state.ket==state.bra
            M_z+=average_spins(state.ket)*ρ_flat[i]
        end
    end
    return M_z
end

function calculate_magnetization(ρ,N)
    first_term_ops = fill(id, N)
    first_term_ops[1] = sx

    m_z::ComplexF64=0
    for i in 1:N
        m_z += tr(ρ*foldl(⊗, first_term_ops))
        first_term_ops = circshift(first_term_ops,1)
    end
    #m_z/=N

    return m_z
end

#N=4
#H=TransverseFieldIsing(;N=4,h=1)
#L=DQIM(N,1,1,1)
#vals, vecs = eigen(L)
#vals, vecs = eigen_sparse(L)
#display(L)
#display(vecs[:,1])
#ρ=reshape(vecs[:,1],2^N,2^N)
#ρ./=tr(ρ)
#ρ=round.(ρ,digits = 12)
#display(ρ)
#l = make_Liouvillian(H, Γ, id)

#npzwrite("rho_real.npy", real.(ρ))
#npzwrite("rho_imag.npy", imag.(ρ))