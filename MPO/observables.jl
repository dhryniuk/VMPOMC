export calculate_z_magnetization, calculate_x_magnetization, calculate_y_magnetization, tensor_calculate_z_magnetization, calculate_spin_spin_correlation, calculate_steady_state_structure_factor

#temporary:
export hermetize_MPO, increase_bond_dimension, L_MPO_products!, density_matrix, calculate_purity, calculate_Renyi_entropy, tensor_purity


export compute_density_matrix
function compute_density_matrix(params::Parameters, A::Array{ComplexF64}, basis)
    
    ρ = zeros(ComplexF64,length(basis), length(basis))
    k = 0
    for ket in basis
        k += 1
        b = 0
        for bra in basis
            b += 1
            sample = Projector(ket, bra)
            p = MPO(params, sample, A)
            ρ[k, b] = p
        end
    end
    return ρ
end

function hermetize_MPO(params::Parameters, A::Array{ComplexF64})

    A = reshape(A,params.χ,params.χ,2,2)
    new_A = deepcopy(A)
    new_A[:,:,1,2] = 0.5*(A[:,:,1,2]+A[:,:,2,1])
    new_A[:,:,2,1] = conj(new_A[:,:,1,2])
    new_A[:,:,1,1] = real(new_A[:,:,1,1])
    new_A[:,:,2,2] = real(new_A[:,:,2,2])
    return reshape(new_A,params.χ,params.χ,4)#::ComplexF64
end

function calculate_x_magnetization(params::Parameters, A::Array{ComplexF64})

    mp = Matrix{Int}(I, params.χ, params.χ)
    for i in 1:params.N-1
        mp *= A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]]
    end
    mp *= A[:,:,dINDEX[(1,0)]]+A[:,:,dINDEX[(0,1)]]
    return tr(mp)
end

function calculate_y_magnetization(params::Parameters, A::Array{ComplexF64})

    mp = Matrix{Int}(I, params.χ, params.χ)
    for i in 1:params.N-1
        mp *= A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]]
    end
    mp *= (A[:,:,dINDEX[(1,0)]]-A[:,:,dINDEX[(0,1)]])*1im
    return -tr(mp)
end

function calculate_z_magnetization(params::Parameters, A::Array{ComplexF64})

    mp = Matrix{Int}(I, params.χ, params.χ)
    for i in 1:params.N-1
        mp *= A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]]
    end
    mp *= -A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]]
    return tr(mp)
end

function tensor_calculate_z_magnetization(params::Parameters, A::Array{ComplexF64})
    
    A = reshape(A,params.χ,params.χ,2,2)
    B = zeros(ComplexF64,params.χ,params.χ)
    D = zeros(ComplexF64,params.χ,params.χ)
    @tensor B[a,b] = A[a,b,c,d]*sz[c,d]
    C = deepcopy(B)
    for _ in 1:params.N-1
        @tensor D[a,b] = C[a,c]*A[c,b,e,e]
        C = deepcopy(D)
    end
    return @tensor C[a,a]
end

export tensor_calculate_magnetization

function tensor_calculate_magnetization(params::Parameters, A::Array{ComplexF64,4}, op::Array{ComplexF64})
    
    B = zeros(ComplexF64,params.χ,params.χ)
    D = zeros(ComplexF64,params.χ,params.χ)
    @tensor B[a,b] = A[a,b,c,d]*op[c,d]
    C = deepcopy(B)
    for _ in 1:params.N-1
        @tensor D[a,b] = C[a,c]*A[c,b,e,e]
        C=deepcopy(D)
    end
    return @tensor C[a,a]
end

function tensor_calculate_magnetization(N::Int64, χ::Int64, A::Array{ComplexF64,4}, op::Array{ComplexF64})

    B = zeros(ComplexF64,χ,χ)
    D = zeros(ComplexF64,χ,χ)
    @tensor B[a,b] = A[a,b,c,d]*op[c,d]
    C = deepcopy(B)
    for _ in 1:N-1
        @tensor D[a,b] = C[a,c]*A[c,b,e,e]
        C = deepcopy(D)
    end
    return @tensor C[a,a]
end


export tensor_calculate_correlation

function tensor_calculate_correlation(params::Parameters, A::Array{ComplexF64,4}, op::Array{ComplexF64})

    B = zeros(ComplexF64,params.χ,params.χ)
    D = zeros(ComplexF64,params.χ,params.χ)
    @tensor B[a,b] = A[a,b,c,d]*op[c,d]
    T = deepcopy(B)
    C = deepcopy(B)
    @tensor C[a,b] = B[a,c]*T[c,b]
    for _ in 1:params.N-2
        @tensor D[a,b] = C[a,c]*A[c,b,e,e]
        C=deepcopy(D)
    end
    return @tensor C[a,a]
end

function tensor_calculate_correlation(N::Int64, χ::Int64, A::Array{ComplexF64,4}, op::Array{ComplexF64})

    B = zeros(ComplexF64,χ,χ)
    D = zeros(ComplexF64,χ,χ)
    @tensor B[a,b] = A[a,b,c,d]*op[c,d]
    T = deepcopy(B)
    C = deepcopy(B)
    @tensor C[a,b] = B[a,c]*T[c,b]
    for _ in 1:N-2
        @tensor D[a,b] = C[a,c]*A[c,b,e,e]
        C = deepcopy(D)
    end
    return @tensor C[a,a]
end

export tensor_calculate_nnn_correlation

function tensor_calculate_nnn_correlation(params::Parameters, A::Array{ComplexF64,4}, op::Array{ComplexF64})

    B = zeros(ComplexF64,params.χ,params.χ)
    @tensor B[a,b] = A[a,b,c,d]*op[c,d]
    T = deepcopy(B)
    @tensor B[a,b] := B[a,c]*A[c,b,e,e]
    @tensor B[a,b] := B[a,c]*T[c,b]
    for _ in 1:params.N-3
        @tensor B[a,b] := B[a,c]*A[c,b,e,e]
    end
    return @tensor B[a,a]
end

function tensor_calculate_nnn_correlation(N::Int64, χ::Int64, A::Array{ComplexF64,4}, op::Array{ComplexF64})
    
    B = zeros(ComplexF64,χ,χ)
    @tensor B[a,b] = A[a,b,c,d]*op[c,d]
    T = deepcopy(B)
    @tensor B[a,b] := B[a,c]*A[c,b,e,e]
    @tensor B[a,b] := B[a,c]*T[c,b]
    for _ in 1:N-3
        @tensor B[a,b] := B[a,c]*A[c,b,e,e]
    end
    return @tensor B[a,a]
end

function increase_bond_dimension(params::Parameters, A::Array{ComplexF64}, step::Int)
    
    params.χ += step
    new_A = 0.001*rand(ComplexF64,params.χ,params.χ,4)
    for i in 1:4
        for j in 1:params.χ-step
            for k in 1:params.χ-step
                new_A[j,k,i] = A[j,k,i]
            end
        end
    end
    new_A ./= normalize_MPO!(MPOMC.params, new_A)
    return new_A
end

function calculate_spin_spin_correlation(params::Parameters, A::Array{ComplexF64}, op, dist::Int)
    A = reshape(A,params.χ,params.χ,2,2)
    B = zeros(ComplexF64,params.χ,params.χ)
    D = zeros(ComplexF64,params.χ,params.χ)
    E = zeros(ComplexF64,params.χ,params.χ)
    @tensor B[a,b] = A[a,b,f,e]*op[e,f]#conj(A[a,b,e,f])*A[u,v,e,f]
    @tensor D[a,b] = A[a,b,f,f]
    C = deepcopy(B)
    for _ in 1:dist-1
        @tensor E[a,b] = C[a,c]*D[c,b]
        C = deepcopy(E)
    end
    @tensor E[a,b] = C[a,c]*B[c,b]
    C = deepcopy(E)
    for _ in 1:params.N-1-dist
        @tensor E[a,b] = C[a,c]*D[c,b]
        C = deepcopy(E)
    end
    return @tensor C[a,a]
end

function calculate_purity(params::Parameters, A::Array{ComplexF64})
    p = Matrix{Int}(I, params.χ, params.χ)
    for _ in 1:params.N
        p *= ( ct(A[:,:,dINDEX[(1,1)]])*A[:,:,dINDEX[(1,1)]] 
        + ct(A[:,:,dINDEX[(0,0)]])*A[:,:,dINDEX[(0,0)]] 
        + ct(A[:,:,dINDEX[(0,1)]])*A[:,:,dINDEX[(1,0)]] 
        + ct(A[:,:,dINDEX[(1,0)]])*A[:,:,dINDEX[(0,1)]] )
    end
    return tr(p)
end

function calculate_Renyi_entropy(params::Parameters, A::Array{ComplexF64})
    return -log2(calculate_purity(params, A))
end

function tensor_purity(params::Parameters, A::Array{ComplexF64})
    A=reshape(A,params.χ,params.χ,2,2)
    B=rand(ComplexF64,params.χ,params.χ,params.χ,params.χ)
    @tensor B[a,b,u,v] = A[a,b,f,e]*A[u,v,e,f]#conj(A[a,b,e,f])*A[u,v,e,f]
    C=deepcopy(B)
    D=deepcopy(B)
    for _ in 1:params.N-1
        @tensor D[a,b,u,v] = C[a,c,u,d]*B[c,b,d,v]
        C=deepcopy(D)
        #B=C
    end
    return @tensor C[a,a,u,u]
    #return tensortrace(B,(1,1,2,2))[1]*params.N
end

### N=2 ONLY!
function tp(A::Array{ComplexF64})
    A=reshape(A,MPOMC.params.χ,MPOMC.params.χ,2,2)
    B=rand(ComplexF64,2,2,2,2)
    @tensor B[a,b,u,v] = A[e,f,a,b]*A[f,e,u,v]
    return @tensor B[a,b,u,v]*B[b,a,v,u]
end

function tp_across_3(A::Array{ComplexF64})
    A=reshape(A,MPOMC.params.χ,MPOMC.params.χ,2,2)    
    B=rand(ComplexF64,MPOMC.params.χ,MPOMC.params.χ,MPOMC.params.χ,MPOMC.params.χ)
    @tensor B[a,b,u,v] = A[a,b,f,e]*A[u,v,e,f]
    #return @tensor B[a,b,u,v]*B[b,a,v,u] #N=2
    return @tensor B[a,c,u,d]*B[c,b,d,v]*B[b,a,v,u] #N=3
end

export one_body_reduced_density_matrix

function one_body_reduced_density_matrix(params::Parameters, A::Array{ComplexF64})
    A=reshape(A,params.χ,params.χ,2,2)
    ρ_1=zeros(ComplexF64,2,2)
    #ρ_1 = deepcopy(A)
    B = deepcopy(A)
    C = similar(B)
    for _ in 1:params.N-1
        @tensor C[a,g,c,d] = B[a,b,c,d]*A[b,g,e,e]
        B=C
    end
    @tensor ρ_1[c,d] = B[a,a,c,d]
    return ρ_1
end