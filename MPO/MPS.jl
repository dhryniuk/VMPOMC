export normalize_MPS

function MPS(params::parameters, sample::Vector{Bool}, A::Array{Float64})
    MPS=Matrix{Float64}(I, params.χ, params.χ)
    for i::UInt16 in 1:params.N
        MPS*=A[:,:,2-sample[i]]
    end
    return tr(MPS)::Float64
end

#Left strings of MPSs:
function L_MPS_strings(params::parameters, sample::Vector{Bool}, A::Array{Float64})
    MPS=Matrix{Float64}(I, params.χ, params.χ)
    L = [ Matrix{Float64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    L[1] = copy(MPS)
    for i::UInt16 in 1:params.N
        MPS *= A[:,:,2-sample[i]]
        L[i+1] = copy(MPS)
    end
    return L
end

#Right strings of MPSs:
function R_MPS_strings(params::parameters, sample::Vector{Bool}, A::Array{Float64})
    MPS=Matrix{Float64}(I, params.χ, params.χ)
    R = [ Matrix{Float64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    R[1] = copy(MPS)
    for i::UInt16 in params.N:-1:1
        MPS = A[:,:,2-sample[i]]*MPS
        #R[i+1] = MPS  MIGHT BE WRONG!!
        R[params.N+2-i] = copy(MPS)
    end
    return R
end

#Claculates the matrix of all derivatives of all elements of the tensor : 
function derv_MPS(params::parameters, sample::Vector{Bool}, L_set::Vector{Matrix{Float64}}, R_set::Vector{Matrix{Float64}})
    ∇::Array{Float64,3}=zeros(Float64, params.χ, params.χ, 2)
    for m::UInt16 in 1:params.N
        B = R_set[params.N+1-m]*L_set[m]
        for i::UInt8 in 1:params.χ
            for j::UInt8 in 1:params.χ 
                @inbounds ∇[i,j,2-sample[m]] += B[j,i] # + B[i,j]
            end
            #@inbounds ∇[i,i,:]./=2
        end
    end
    return ∇
end

#Inferior:
function vect_derv_MPS(params::parameters, sample::Vector{Bool}, L_set::Vector{Matrix{Float64}}, R_set::Vector{Matrix{Float64}})
    ∇::Array{Float64,3}=zeros(Float64, params.χ, params.χ, 2)
    #L_set = L_MPS_strings(params, sample, A)
    #R_set = R_MPS_strings(params, sample, A)
    for m::UInt8 in 1:params.N
        B = R_set[params.N+1-m]*L_set[m]
        #for i::UInt8 in 1:params.χ
        #    for j::UInt8 in 1:params.χ
        @inbounds for (i::UInt8,j::UInt8) in zip(1:params.χ,1:params.χ)
            #∇[i,j,dINDEX2[sample[m]]] += B[i,j] + B[j,i]
            println((i,j))
            ∇[i,j,2-sample[m]] += B[i,j] + B[j,i]
        end
        @inbounds for i in 1:params.χ
            ∇[i,i,:]./=2
        end
    end
    return ∇
end


function normalize_MPS(params::parameters, A::Array{Float64})
    #MPS=(A[:,:,dINDEX2[1]]^2+A[:,:,dINDEX2[0]]^2)^params.N
    MPS=(A[:,:,dINDEX2[1]]*adjoint(A[:,:,dINDEX2[1]]) + A[:,:,dINDEX2[0]]*adjoint(A[:,:,dINDEX2[0]]))^params.N
    return tr(MPS)^(1/params.N)#::ComplexF64
end

export tensor_normalize_MPS

function tensor_normalize_MPS(params::parameters, A::Array{Float64})
    B=rand(Float64,params.χ,params.χ,params.χ,params.χ)
    @tensor B[a,b,u,v] = A[a,b,e]*A[u,v,e]#conj(A[a,b,e,f])*A[u,v,e,f]
    C=deepcopy(B)
    for _ in 1:params.N-1
        @tensor C[a,b,u,v] = C[a,c,u,d]*B[c,b,d,v]
        #B=C
    end
    norm = @tensor C[a,a,u,u]
    return norm^(1/(2*params.N))
end









#ComplexF64 version:

function MPS(params::parameters, sample::Vector{Bool}, A::Array{ComplexF64})
    MPS=Matrix{ComplexF64}(I, params.χ, params.χ)
    for i::UInt16 in 1:params.N
        MPS*=A[:,:,2-sample[i]]
    end
    return tr(MPS)::ComplexF64
end

#Left strings of MPSs:
function L_MPS_strings(params::parameters, sample::Vector{Bool}, A::Array{ComplexF64})
    MPS=Matrix{ComplexF64}(I, params.χ, params.χ)
    L = [ Matrix{ComplexF64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    L[1] = copy(MPS)
    for i::UInt16 in 1:params.N
        MPS *= A[:,:,2-sample[i]]
        L[i+1] = copy(MPS)
    end
    return L
end

#Right strings of MPSs:
function R_MPS_strings(params::parameters, sample::Vector{Bool}, A::Array{ComplexF64})
    MPS=Matrix{ComplexF64}(I, params.χ, params.χ)
    R = [ Matrix{ComplexF64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    R[1] = copy(MPS)
    for i::UInt16 in params.N:-1:1
        MPS = A[:,:,2-sample[i]]*MPS
        #R[i+1] = MPS  MIGHT BE WRONG!!
        R[params.N+2-i] = copy(MPS)
    end
    return R
end

#Claculates the matrix of all derivatives of all elements of the tensor : 
function derv_MPS(params::parameters, sample::Vector{Bool}, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})
    ∇::Array{ComplexF64,3}=zeros(ComplexF64, params.χ, params.χ, 2)
    for m::UInt16 in 1:params.N
        B = R_set[params.N+1-m]*L_set[m]
        for i::UInt8 in 1:params.χ
            for j::UInt8 in 1:params.χ 
                @inbounds ∇[i,j,2-sample[m]] += B[j,i] # + B[i,j]
            end
            #@inbounds ∇[i,i,:]./=2
        end
    end
    return ∇
end

function normalize_MPS(params::parameters, A::Array{ComplexF64})
    MPS=(A[:,:,dINDEX2[1]]*adjoint(A[:,:,dINDEX2[1]]) + A[:,:,dINDEX2[0]]*adjoint(A[:,:,dINDEX2[0]]))^params.N
    return tr(MPS)^(1/params.N)#::ComplexF64
end

function tensor_normalize_MPS(params::parameters, A::Array{ComplexF64})
    B=rand(ComplexF64,params.χ,params.χ,params.χ,params.χ)
    @tensor B[a,b,u,v] = A[a,b,e]*conj(A[u,v,e])#conj(A[a,b,e,f])*A[u,v,e,f]
    C=deepcopy(B)
    for _ in 1:params.N-1
        @tensor C[a,b,u,v] = C[a,c,u,d]*B[c,b,d,v]
        #B=C
    end
    norm = @tensor C[a,a,u,u]
    return norm^(1/(2*params.N))
end














export open_MPS, open_L_MPS_strings, open_R_MPS_strings, open_derv_MPS

function open_MPS(params::parameters, sample::Vector{Bool}, A::Array{Float64}, V::Array{Float64})
    MPS = transpose( V[:,2-sample[1]] ) #left boundary
    for i::UInt16 in 2:params.N-1 #bulk
        MPS*=A[:,:,2-sample[i]]
    end
    MPS*= V[:,2-sample[N]] #right boundary
    return MPS::Float64
end

#Left strings of MPSs:
function open_L_MPS_strings(params::parameters, sample::Vector{Bool}, A::Array{Float64}, V::Array{Float64})
    MPS = transpose( V[:,2-sample[1]] ) #left boundary
    L = [ transpose(Vector{Float64}(undef,params.χ)) for _ in 1:params.N-1 ]
    L[1] = copy(MPS)
    for i::UInt16 in 2:params.N-1
        MPS *= A[:,:,2-sample[i]]
        L[i] = copy(MPS)
    end
    #MPS *= V[:,2-sample[params.N]]
    #L[params.N] = copy(MPS)
    return L
end

#Right strings of MPSs:
function open_R_MPS_strings(params::parameters, sample::Vector{Bool}, A::Array{Float64}, V::Array{Float64})
    MPS = V[:,2-sample[params.N]] #left boundary
    R = [ Vector{Float64}(undef,params.χ) for _ in 1:params.N-1 ]
    R[1] = copy(MPS)
    for i::UInt16 in params.N-1:-1:2
        MPS = A[:,:,2-sample[i]]*MPS
        #R[i+1] = MPS  MIGHT BE WRONG!!
        R[params.N+1-i] = copy(MPS)
    end
    #MPS = transpose(V[:,2-sample[1]])*MPS
    #R[params.N] = copy(MPS)
    return R
end

#Claculates the matrix of all derivatives of all elements of the tensor : 
#function open_derv_MPS(params::parameters, sample::Vector{Bool}, L_set::Vector{Matrix{Float64}}, R_set::Vector{Matrix{Float64}})
function open_derv_MPS(params::parameters, sample::Vector{Bool}, L_set, R_set::Vector{Matrix{Float64}})
    ∇_bulk::Array{Float64,3}=zeros(Float64, params.χ, params.χ, 2)
    ∇_boundary::Array{Float64,2}=zeros(Float64, params.χ, 2)
    for m::UInt16 in 2:params.N-1
        #B = L_set[m]*R_set[params.N+1-m]
        for i::UInt8 in 1:params.χ
            for j::UInt8 in 1:params.χ 
                @inbounds ∇_bulk[i,j,2-sample[m]] += L_set[m-1][i]*R_set[params.N-m][j] #B[j,i] # + B[i,j]
            end
            #@inbounds ∇[i,i,:]./=2
        end
    end
    for i::UInt8 in 1:params.χ
        ∇_boundary[i,2-sample[1]] += R_set[params.N-1][i]
        ∇_boundary[i,2-sample[params.N]] += L_set[params.N-1][i]
    end
    return ∇_bulk, ∇_boundary
end








#ComplexF64 version:

function open_MPS(params::parameters, sample::Vector{Bool}, A::Array{ComplexF64}, V::Array{ComplexF64})
    MPS = transpose( V[:,2-sample[1]] ) #left boundary
    for i::UInt16 in 2:params.N-1 #bulk
        MPS*=A[:,:,2-sample[i]]
    end
    MPS*= V[:,2-sample[N]] #right boundary
    return MPS::ComplexF64
end

#Left strings of MPSs:
function open_L_MPS_strings(params::parameters, sample::Vector{Bool}, A::Array{ComplexF64}, V::Array{ComplexF64})
    MPS = transpose( V[:,2-sample[1]] ) #left boundary
    L = [ transpose(Vector{ComplexF64}(undef,params.χ)) for _ in 1:params.N-1 ]
    L[1] = copy(MPS)
    for i::UInt16 in 2:params.N-1
        MPS *= A[:,:,2-sample[i]]
        L[i] = copy(MPS)
    end
    #MPS *= V[:,2-sample[params.N]]
    #L[params.N] = copy(MPS)
    return L
end

#Right strings of MPSs:
function open_R_MPS_strings(params::parameters, sample::Vector{Bool}, A::Array{ComplexF64}, V::Array{ComplexF64})
    MPS = V[:,2-sample[params.N]] #left boundary
    R = [ Vector{ComplexF64}(undef,params.χ) for _ in 1:params.N-1 ]
    R[1] = copy(MPS)
    for i::UInt16 in params.N-1:-1:2
        MPS = A[:,:,2-sample[i]]*MPS
        #R[i+1] = MPS  MIGHT BE WRONG!!
        R[params.N+1-i] = copy(MPS)
    end
    #MPS = transpose(V[:,2-sample[1]])*MPS
    #R[params.N] = copy(MPS)
    return R
end

#Claculates the matrix of all derivatives of all elements of the tensor : 
#function open_derv_MPS(params::parameters, sample::Vector{Bool}, L_set::Vector{Matrix{Float64}}, R_set::Vector{Matrix{Float64}})
#function open_derv_MPS(params::parameters, sample::Vector{Bool}, L_set, R_set::Vector{Vector{ComplexF64}})
function open_derv_MPS(params::parameters, sample::Vector{Bool}, L_set, R_set)
    ∇_bulk::Array{ComplexF64,3}=zeros(ComplexF64, params.χ, params.χ, 2)
    ∇_boundary::Array{ComplexF64,2}=zeros(ComplexF64, params.χ, 2)
    for m::UInt16 in 2:params.N-1
        #B = L_set[m]*R_set[params.N+1-m]
        for i::UInt8 in 1:params.χ
            for j::UInt8 in 1:params.χ 
                @inbounds ∇_bulk[i,j,2-sample[m]] += L_set[m-1][i]*R_set[params.N-m][j] #B[j,i] # + B[i,j]
            end
            #@inbounds ∇[i,i,:]./=2
        end
    end
    for i::UInt8 in 1:params.χ
        ∇_boundary[i,2-sample[1]] += R_set[params.N-1][i]
        ∇_boundary[i,2-sample[params.N]] += L_set[params.N-1][i]
    end
    return ∇_bulk, ∇_boundary
end