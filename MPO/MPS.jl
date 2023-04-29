export MPS, L_MPS_strings, R_MPS_strings, ∂MPS, normalize_MPS

#Contracts MPS over all indices:
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
    #display(MPS)
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
function ∂MPS(params::parameters, sample::Vector{Bool}, L_set::Vector{Matrix{Float64}}, R_set::Vector{Matrix{Float64}})
    ∂::Array{Float64,3}=zeros(Float64, params.χ, params.χ, 2)
    for m::UInt16 in 1:params.N
        B = R_set[params.N+1-m]*L_set[m]
        for i::UInt8 in 1:params.χ
            for j::UInt8 in 1:params.χ 
                @inbounds ∂[i,j,2-sample[m]] += B[j,i] # + B[i,j]
            end
            #@inbounds ∂[i,i,:]./=2
        end
    end
    return ∂
end

function normalize_MPS(params::parameters, A::Array{Float64})
    B=rand(Float64,params.χ,params.χ,params.χ,params.χ)
    @tensor B[a,b,u,v] = A[a,b,e]*A[u,v,e]#conj(A[a,b,e,f])*A[u,v,e,f]
    C=deepcopy(B)
    for _ in 1:params.N-1
        @tensor C[a,b,u,v] = C[a,c,u,d]*B[c,b,d,v]
        #B=C
    end
    norm = @tensor C[a,a,u,u]
    return A./norm^(1/(2*params.N))
end

function norm_MPS(params::parameters, A::Array{Float64})
    B=rand(Float64,params.χ,params.χ,params.χ,params.χ)
    @tensor B[a,b,u,v] = A[a,b,e]*A[u,v,e]#conj(A[a,b,e,f])*A[u,v,e,f]
    C=deepcopy(B)
    for _ in 1:params.N-1
        @tensor C[a,b,u,v] = C[a,c,u,d]*B[c,b,d,v]
    end
    norm = @tensor C[a,a,u,u]
    return norm
end

export op_exp_val_MPS
function op_exp_val_MPS(op, j, params::parameters, A::Array{Float64})
    B=zeros(Float64,params.χ,params.χ,params.χ,params.χ)
    @tensor B[a,b,u,v] = A[a,b,e]*op[e,f]*conj(A[u,v,f])#conj(A[a,b,e,f])*A[u,v,e,f]
    C=deepcopy(B)
    for _ in 1:params.N-1
        @tensor C[a,b,u,v] = C[a,c,u,d]*B[c,b,d,v]
        #B=C
    end
    exp_val = @tensor C[a,a,u,u]
    return exp_val/norm_MPS(params,A)
end

export make_wavefunction_MPS
function make_wavefunction_MPS(params::parameters, A::Array{Float64}, basis)
    Ψ=zeros(length(basis))
    for (i, state) in enumerate(basis)
        Ψ[i] = MPS(params,state,A)
    end
    return Ψ
end





"""
ComplexF64 version:
"""

function MPS(params::parameters, sample::Vector{Bool}, A::Array{ComplexF64})
    MPS=Matrix{ComplexF64}(I, params.χ, params.χ)
    for i::UInt16 in 1:params.N
        MPS*=A[:,:,2-sample[i]]
    end
    return tr(MPS)::ComplexF64
end
"""
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
"""

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
function ∂MPS(params::parameters, sample::Vector{Bool}, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})
    ∂::Array{ComplexF64,3}=zeros(ComplexF64, params.χ, params.χ, 2)
    for m::UInt16 in 1:params.N
        B = R_set[params.N+1-m]*L_set[m]
        for i::UInt8 in 1:params.χ
            for j::UInt8 in 1:params.χ 
                @inbounds ∂[i,j,2-sample[m]] += B[j,i] # + B[i,j]
            end
            #@inbounds ∂[i,i,:]./=2
        end
    end
    return ∂
end

function normalize_MPS(params::parameters, A::Array{ComplexF64})
    B=rand(ComplexF64,params.χ,params.χ,params.χ,params.χ)
    @tensor B[a,b,u,v] = A[a,b,e]*conj(A[u,v,e])#conj(A[a,b,e,f])*A[u,v,e,f]
    C=deepcopy(B)
    for _ in 1:params.N-1
        @tensor C[a,b,u,v] = C[a,c,u,d]*B[c,b,d,v]
        #B=C
    end
    norm = @tensor C[a,a,u,u]
    return A./norm^(1/(2*params.N))
end