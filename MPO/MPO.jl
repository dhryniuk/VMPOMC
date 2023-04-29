export normalize_MPO, normalize_MPS, calculate_z_magnetization, calculate_x_magnetization, calculate_y_magnetization, double_bond_dimension

#temporary:
export hermetize_MPO, increase_bond_dimension, L_MPO_strings, density_matrix, calculate_purity, calculate_Renyi_entropy, tensor_purity

mutable struct density_matrix#{Coeff<:Int64, Vec<:Vector{Float64}}
    coeff::ComplexF64
    ket::Vector{Bool}
    bra::Vector{Bool}
    #ket::Vector{Int8}
    #bra::Vector{Int8}
    #density_matrix(coeff,ket,bra) = new(coeff,ket,bra)
end

Base.:*(x::density_matrix, y::density_matrix) = density_matrix(x.coeff * y.coeff, vcat(x.ket, y.ket), vcat(x.bra, y.bra))

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

#export L_MPO_strings, R_MPO_strings

#Left strings of MPOs:
function L_MPO_strings(params::parameters, sample::density_matrix, A::Array{ComplexF64})
    MPO::Matrix{ComplexF64} = Matrix{ComplexF64}(I, params.χ, params.χ)
    #L = Vector{Matrix{ComplexF64}}()
    #push!(L,copy(MPO))


    # CONSIDER REMOVING PREALLOCATION:
    L::Vector{Matrix{ComplexF64}} = [ Matrix{ComplexF64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    
    
    #L[1] = copy(MPO)    ### IT SEEMS COPY IS VERY IMPORTANT!
    L[1] = MPO
    for i::UInt8 in 1:params.N
        #MPO *= A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        MPO *= A[:,:,1+2*sample.ket[i]+sample.bra[i]]
        #MPO*=A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
        #push!(L,copy(MPO))
        #L[i+1] = copy(MPO)
        L[i+1] = MPO
    end
    return L
end

#Right strings of MPOs:
function R_MPO_strings(params::parameters, sample::density_matrix, A::Array{ComplexF64})
    MPO::Matrix{ComplexF64} = Matrix{ComplexF64}(I, params.χ, params.χ)
    #R = Vector{Matrix{ComplexF64}}()
    #push!(R,copy(MPO))
    R::Vector{Matrix{ComplexF64}} = [ Matrix{ComplexF64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    #R[1] = copy(MPO) 
    R[1] = MPO
    for i::UInt8 in params.N:-1:1
        #MPO = A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]*MPO
        MPO = A[:,:,1+2*sample.ket[i]+sample.bra[i]]*MPO

        # MATRIX MULTIPLICATION IS NOT COMMUTATIVE, IDIOT

        #push!(R,copy(MPO))
        #R[params.N+2-i] = copy(MPO)
        R[params.N+2-i] = MPO
    end
    return R
end

#Left strings of MPOs:
function L_MPO_strings_without_preallocation(L::Vector{Matrix{ComplexF64}}, AUX_ID::Matrix{ComplexF64}, params::parameters, sample::density_matrix, A::Array{ComplexF64})
    #L[1] = Matrix{ComplexF64}(I, params.χ, params.χ)
    L[1] = AUX_ID
    for i::UInt8 in 1:params.N
        idx = 1+2*sample.ket[i]+sample.bra[i]
        mul!(L[i+1],L[i],@view(A[:,:,idx]))
    end
    return L
end

#Right strings of MPOs:
function R_MPO_strings_without_preallocation(R::Vector{Matrix{ComplexF64}}, AUX_ID::Matrix{ComplexF64}, params::parameters, sample::density_matrix, A::Array{ComplexF64})
    #R[1] = Matrix{ComplexF64}(I, params.χ, params.χ)
    R[1] = AUX_ID
    for i::UInt8 in params.N:-1:1
        idx = 1+2*sample.ket[i]+sample.bra[i]
        mul!(R[params.N+2-i],@view(A[:,:,idx]),R[params.N+1-i])
    end
    return R
end


function normalize_MPO(params::parameters, A::Array{ComplexF64})
    #MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])^params.N
    MPO=(A[:,:,1]+A[:,:,4])^params.N
    return A./=tr(MPO)^(1/params.N)#::ComplexF64
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

function ∂MPO(params::parameters, sample::density_matrix, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})
    ∂::Array{ComplexF64,3}=zeros(ComplexF64, params.χ, params.χ, 4)
    #L_set = L_MPO_strings(sample, A)
    #R_set = R_MPO_strings(sample, A)
    B::Matrix{ComplexF64} = zeros(params.χ,params.χ)
    for m::UInt8 in 1:params.N
        #B = R_set[params.N+1-m]*L_set[m]
        mul!(B,R_set[params.N+1-m],L_set[m])
        for i::UInt8 in 1:params.χ
            for j::UInt8 in 1:params.χ
                #@inbounds ∂[i,j,dINDEX[(sample.ket[m],sample.bra[m])]] += B[j,i] # + B[i,j]
                @inbounds ∂[i,j,1+2*sample.ket[m]+sample.bra[m]] += B[j,i] # + B[i,j]
            end
            #@inbounds ∂[i,i,:]./=2
        end
    end
    return ∂
end

function ∂MPO_without_preallocation(B::Matrix{ComplexF64}, params::parameters, sample::density_matrix, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})
    ∂::Array{ComplexF64,3}=zeros(ComplexF64, params.χ, params.χ, 4)
    for m::UInt8 in 1:params.N
        mul!(B,R_set[params.N+1-m],L_set[m])
        for i::UInt8 in 1:params.χ
            for j::UInt8 in 1:params.χ
                @inbounds ∂[i,j,1+2*sample.ket[m]+sample.bra[m]] += B[j,i]
            end
        end
    end
    return ∂
end

function derv_MPO(params::parameters, sample_ket::Array{Bool}, sample_bra::Array{Bool}, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})
    ∇=zeros(ComplexF64, params.χ, params.χ,4)
    #L_set = L_MPO_strings(sample, A)
    #R_set = R_MPO_strings(sample, A)
    for m::UInt8 in 1:params.N
        B = R_set[params.N+1-m]*L_set[m]
        for i::UInt8 in 1:params.χ
            for j::UInt8 in 1:params.χ
                @inbounds ∇[i,j,dINDEX[(sample_ket[m],sample_bra[m])]] += B[j,i] # + B[i,j]
            end
            #@inbounds ∇[i,i,:]./=2
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

export tensor_calculate_z_magnetization

function tensor_calculate_z_magnetization(params::parameters, A::Array{ComplexF64})
    A=reshape(A,params.χ,params.χ,2,2)
    B=zeros(ComplexF64,params.χ,params.χ)
    D=zeros(ComplexF64,params.χ,params.χ)
    @tensor B[a,b]=A[a,b,c,d]*sz[c,d]
    C=deepcopy(B)
    for _ in 1:params.N-1
        @tensor D[a,b] = C[a,c]*A[c,b,e,e]
        C=deepcopy(D)
    end
    return @tensor C[a,a]
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

export calculate_spin_spin_correlation

function calculate_spin_spin_correlation(params::parameters, A::Array{ComplexF64}, op, dist::Int)
    A=reshape(A,params.χ,params.χ,2,2)
    B=zeros(ComplexF64,params.χ,params.χ)
    D=zeros(ComplexF64,params.χ,params.χ)
    E=zeros(ComplexF64,params.χ,params.χ)
    @tensor B[a,b] = A[a,b,f,e]*op[e,f]#conj(A[a,b,e,f])*A[u,v,e,f]
    @tensor D[a,b] = A[a,b,f,f]
    C=deepcopy(B)
    for _ in 1:dist-1
        @tensor E[a,b] = C[a,c]*D[c,b]
        C=deepcopy(E)
    end
    @tensor E[a,b] = C[a,c]*B[c,b]
    C=deepcopy(E)
    for _ in 1:params.N-1-dist
        @tensor E[a,b] = C[a,c]*D[c,b]
        C=deepcopy(E)
    end
    return @tensor C[a,a]
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




























#vectorized samples version:

function MPO(params::parameters, sample_ket::Array{Bool}, sample_bra::Array{Bool}, A::Array{ComplexF64})
    MPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    for i in 1:params.N
        MPO*=A[:,:,dINDEX[(sample_ket[i],sample_bra[i])]]
    end
    return tr(MPO)::ComplexF64
end

#MPO string beginning as site l and ending at site r:
function MPO_string(sample_ket::Array{Bool}, sample_bra::Array{Bool}, A::Array{ComplexF64},l,r)
    MPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    for i in l:r
        MPO*=A[:,:,dINDEX2[sample_ket[i]],dINDEX2[sample_bra[i]]]
    end
    return MPO
end

#Left strings of MPOs:
function L_MPO_strings(params::parameters, sample_ket::Array{Bool}, sample_bra::Array{Bool}, A::Array{ComplexF64})
    MPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    #L = Vector{Matrix{ComplexF64}}()
    #push!(L,copy(MPO))
    L = [ Matrix{ComplexF64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    L[1] = copy(MPO)    ### IT SEEMS COPY IS VERY IMPORTANT!
    for i::UInt8 in 1:params.N
        MPO *= A[:,:,dINDEX[(sample_ket[i],sample_bra[i])]]
        #MPO*=A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
        #push!(L,copy(MPO))
        L[i+1] = copy(MPO)
    end
    return L
end

#Right strings of MPOs:
function R_MPO_strings(params::parameters, sample_ket::Array{Bool}, sample_bra::Array{Bool}, A::Array{ComplexF64})
    MPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    #R = Vector{Matrix{ComplexF64}}()
    #push!(R,copy(MPO))
    R = [ Matrix{ComplexF64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    R[1] = copy(MPO) 
    for i::UInt8 in params.N:-1:1
        MPO = A[:,:,dINDEX[(sample_ket[i],sample_bra[i])]]*MPO

        # MATRIX MULTIPLICATION IS NOT COMMUTATIVE, IDIOT

        #push!(R,copy(MPO))
        R[params.N+2-i] = copy(MPO)
    end
    return R
end































export open_MPO, open_L_MPO_strings, open_R_MPO_strings, open_derv_MPO, normalize_open_MPO


function open_MPO(params::parameters, sample::density_matrix, A::Array{ComplexF64}, V::Array{ComplexF64})
    MPO = transpose( V[:,dINDEX[(sample.ket[1],sample.bra[1])]] ) #left boundary
    for i::UInt16 in 2:params.N-1 #bulk
        MPS*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])] ]
    end
    MPS*= V[:,dINDEX[(sample.ket[params.N],sample.bra[params.N])] ]#right boundary
    return MPS::ComplexF64
end

#Left strings of MPSs:
function open_L_MPO_strings(params::parameters, sample::density_matrix, A::Array{ComplexF64}, V::Array{ComplexF64})
    MPO = transpose( V[:,dINDEX[(sample.ket[1],sample.bra[1])] ] )#left boundary
    L = [ transpose(Vector{ComplexF64}(undef,params.χ)) for _ in 1:params.N-1 ]
    L[1] = copy(MPO)
    for i::UInt16 in 2:params.N-1
        MPO *= A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        L[i] = copy(MPO)
    end
    #MPS *= V[:,2-sample[params.N]]
    #L[params.N] = copy(MPS)
    return L
end

#Right strings of MPSs:
function open_R_MPO_strings(params::parameters, sample::density_matrix, A::Array{ComplexF64}, V::Array{ComplexF64})
    MPO = V[:,dINDEX[(sample.ket[params.N],sample.bra[params.N])]] #left boundary
    R = [ Vector{ComplexF64}(undef,params.χ) for _ in 1:params.N-1 ]
    R[1] = copy(MPO)
    for i::UInt16 in params.N-1:-1:2
        MPO = A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]*MPO
        #R[i+1] = MPS  MIGHT BE WRONG!!
        R[params.N+1-i] = copy(MPO)
    end
    #MPS = transpose(V[:,2-sample[1]])*MPS
    #R[params.N] = copy(MPS)
    return R
end

#Claculates the matrix of all derivatives of all elements of the tensor : 
function open_derv_MPO(params::parameters, sample::density_matrix, L_set, R_set)
    ∇_bulk::Array{ComplexF64,3}=zeros(ComplexF64, params.χ, params.χ, 4)
    ∇_boundary::Array{ComplexF64,2}=zeros(ComplexF64, params.χ, 4)
    for m::UInt16 in 2:params.N-1
        #B = L_set[m]*R_set[params.N+1-m]
        for i::UInt8 in 1:params.χ
            for j::UInt8 in 1:params.χ 
                @inbounds ∇_bulk[i,j,dINDEX[(sample.ket[m],sample.bra[m])]] += L_set[m-1][i]*R_set[params.N-m][j] #B[j,i] # + B[i,j]
            end
            #@inbounds ∇[i,i,:]./=2
        end
    end
    for i::UInt8 in 1:params.χ
        ∇_boundary[i,dINDEX[(sample.ket[1],sample.bra[1])]] += R_set[params.N-1][i]
        ∇_boundary[i,dINDEX[(sample.ket[params.N],sample.bra[params.N])]] += L_set[params.N-1][i]
    end
    return ∇_bulk, ∇_boundary
end

function normalize_open_MPO(params::parameters, A::Array{ComplexF64}, V::Array{ComplexF64})
    MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])^(params.N-2)
    return ( transpose(V[:,dINDEX[(1,1)]]+V[:,dINDEX[(0,0)]]) * MPO * (V[:,dINDEX[(1,1)]]+V[:,dINDEX[(0,0)]]) )^(1/params.N)#::ComplexF64
end