export normalize_MPO, normalize_MPS, calculate_z_magnetization, calculate_x_magnetization, calculate_y_magnetization, double_bond_dimension

#temporary:
export hermetize_MPO, increase_bond_dimension, L_MPO_strings, density_matrix, calculate_purity, calculate_Renyi_entropy, tensor_purity

mutable struct parameters
    N::Int
    dim::Int
    χ::Int
    J::Float64
    h::Float64
    γ::Float64
    α::Int
end

mutable struct density_matrix#{Coeff<:Int64, Vec<:Vector{Float64}}
    coeff::ComplexF64
    ket::Vector{Bool}
    bra::Vector{Bool}
    #ket::Vector{Int8}
    #bra::Vector{Int8}
    #density_matrix(coeff,ket,bra) = new(coeff,ket,bra)
end

function MPO(params::parameters, sample::density_matrix, A::Array{ComplexF64})
    MPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    for i in 1:params.N
        MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
    end
    return tr(MPO)::ComplexF64
end

#MPO string beginning as site l and ending at site r:
function MPO_string(sample::density_matrix, A::Array{ComplexF64},l,r)
    MPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    for i in l:r
        MPO*=A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
    end
    return MPO
end

#Left strings of MPOs:
function L_MPO_strings(params::parameters, sample::density_matrix, A::Array{ComplexF64})
    L = Vector{Matrix{ComplexF64}}()
    MPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    push!(L,copy(MPO))
    for i::UInt8 in 1:params.N
        MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        #MPO*=A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
        push!(L,copy(MPO))
    end
    return L
end

#Right strings of MPOs:
function R_MPO_strings(params::parameters, sample::density_matrix, A::Array{ComplexF64})
    R = Vector{Matrix{ComplexF64}}()
    MPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    push!(R,copy(MPO))
    for i::UInt8 in params.N:-1:1
        MPO=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]*MPO

        # MATRIX MULTIPLICATION IS NOT COMMUTATIVE, IDIOT

        push!(R,copy(MPO))
    end
    return R
end

function normalize_MPO(params::parameters, A::Array{ComplexF64})
    MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])^params.N
    return tr(MPO)^(1/params.N)#::ComplexF64
end

function hermetize_MPO(params::parameters, A::Array{ComplexF64})
    A=reshape(A,params.χ,params.χ,2,2)
    new_A = deepcopy(A)
    new_A[:,:,1,2]=0.5*(A[:,:,1,2]+A[:,:,2,1])
    new_A[:,:,2,1]=conj(new_A[:,:,1,2])
    new_A[:,:,1,1]=real(new_A[:,:,1,1])
    new_A[:,:,2,2]=real(new_A[:,:,2,2])
    return reshape(new_A,params.χ,params.χ,4)#::ComplexF64
end

function B_list(m, sample::density_matrix, A::Array{ComplexF64}) #FIX m ORDERING
    B_list=Matrix{ComplexF64}[Matrix{Int}(I, χ, χ)]
    for j::UInt8 in 1:params.N-1
        push!(B_list,A[:,:,dINDEX[(sample.ket[mod(m+j-1,N)+1],sample.bra[mod(m+j-1,N)+1])]])
    end
    return B_list
end

function derv_MPO(params::parameters, sample::density_matrix, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})
    ∇=zeros(ComplexF64, params.χ, params.χ,4)
    #L_set = L_MPO_strings(sample, A)
    #R_set = R_MPO_strings(sample, A)
    for m::UInt8 in 1:params.N
        B = R_set[params.N+1-m]*L_set[m]
        for i in 1:params.χ
            for j in 1:params.χ
                ∇[i,j,dINDEX[(sample.ket[m],sample.bra[m])]] += B[i,j] + B[j,i]
            end
            ∇[i,i,:]./=2
        end
    end
    return ∇
end


function calculate_x_magnetization(params::parameters, A::Array{ComplexF64})
    mp=Matrix{Int}(I, params.χ, params.χ)
    for i in 1:params.N-1
        mp*=A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]]
    end
    mp*=A[:,:,dINDEX[(1,0)]]+A[:,:,dINDEX[(0,1)]]
    return tr(mp)
end

function calculate_y_magnetization(params::parameters, A::Array{ComplexF64})
    mp=Matrix{Int}(I, params.χ, params.χ)
    for i in 1:params.N-1
        mp*=A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]]
    end
    mp*=(A[:,:,dINDEX[(1,0)]]-A[:,:,dINDEX[(0,1)]])*1im
    return -tr(mp)
end

function calculate_z_magnetization(params::parameters, A::Array{ComplexF64})
    mp=Matrix{Int}(I, params.χ, params.χ)
    for i in 1:params.N-1
        mp*=A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]]
    end
    mp*=-A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]]
    return tr(mp)
end

function double_bond_dimension(params::parameters, A::Array{ComplexF64})
    params.χ*=2
    new_A = Array{ComplexF64}(undef, params.χ,params.χ,4)#2,2)
    for i in 1:4
        new_A[:,:,i] = kron(A[:,:,i], [1 0.99;0.99 1])
    end
    #for i in 1:2
    #    for j in 1:2
    #        new_A[:,:,i,j] = kron(A[:,:,i,j], [1 1;1 1])
    #    end
    #end
    new_A./=normalize_MPO(MPOMC.params, new_A)
    return new_A
end

function increase_bond_dimension(params::parameters, A::Array{ComplexF64}, step::Int)
    params.χ+=step
    new_A = 0.001*rand(ComplexF64,params.χ,params.χ,4)#2,2)
    for i in 1:4
        for j in 1:params.χ-step
            for k in 1:params.χ-step
                new_A[j,k,i] = A[j,k,i]
            end
        end
    end
    new_A./=normalize_MPO(MPOMC.params, new_A)
    return new_A
end

function calculate_purity(params::parameters, A::Array{ComplexF64})
    p = Matrix{Int}(I, params.χ, params.χ)
    for _ in 1:params.N
        p *= ( ct(A[:,:,dINDEX[(1,1)]])*A[:,:,dINDEX[(1,1)]] 
        + ct(A[:,:,dINDEX[(0,0)]])*A[:,:,dINDEX[(0,0)]] 
        + ct(A[:,:,dINDEX[(0,1)]])*A[:,:,dINDEX[(1,0)]] 
        + ct(A[:,:,dINDEX[(1,0)]])*A[:,:,dINDEX[(0,1)]] )
    end
    return tr(p)
end

function calculate_Renyi_entropy(params::parameters, A::Array{ComplexF64})
    return -log2(calculate_purity(params, A))
end

function tensor_purity(params::parameters, A::Array{ComplexF64})
    A=reshape(A,params.χ,params.χ,2,2)
    B=rand(ComplexF64,params.χ,params.χ,params.χ,params.χ)
    @tensor B[a,b,u,v] = A[a,b,f,e]*A[u,v,e,f]#conj(A[a,b,e,f])*A[u,v,e,f]
    C=deepcopy(B)
    for _ in 1:params.N-1
        @tensor C[a,b,u,v] = C[a,c,u,d]*B[c,b,d,v]
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
