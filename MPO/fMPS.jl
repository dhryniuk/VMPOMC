export fMPS, L_fMPS_strings, R_fMPS_strings, ∂fMPS

function fMPS(params::parameters, sample::Vector{Bool}, A::Array{Float64}, V::Array{Float64})
    MPS = transpose( V[:,2-sample[1]] ) #left boundary
    for i::UInt16 in 2:params.N-1 #bulk
        MPS*=A[:,:,2-sample[i]]
    end
    MPS*= V[:,2-sample[N]] #right boundary
    return MPS::Float64
end

#Left strings of MPSs:
function L_fMPS_strings(params::parameters, sample::Vector{Bool}, A::Array{Float64}, V::Array{Float64})
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
function R_fMPS_strings(params::parameters, sample::Vector{Bool}, A::Array{Float64}, V::Array{Float64})
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
function ∂fMPS(params::parameters, sample::Vector{Bool}, L_set, R_set::Vector{Matrix{Float64}})
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




"""
ComplexF64 version:
"""

function fMPS(params::parameters, sample::Vector{Bool}, A::Array{ComplexF64}, V::Array{ComplexF64})
    MPS = transpose( V[:,2-sample[1]] ) #left boundary
    for i::UInt16 in 2:params.N-1 #bulk
        MPS*=A[:,:,2-sample[i]]
    end
    MPS*= V[:,2-sample[N]] #right boundary
    return MPS::ComplexF64
end

#Left strings of MPSs:
function L_fMPS_strings(params::parameters, sample::Vector{Bool}, A::Array{ComplexF64}, V::Array{ComplexF64})
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
function R_fMPS_strings(params::parameters, sample::Vector{Bool}, A::Array{ComplexF64}, V::Array{ComplexF64})
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
function ∂fMPS(params::parameters, sample::Vector{Bool}, L_set, R_set)
    ∂_bulk::Array{ComplexF64,3}=zeros(ComplexF64, params.χ, params.χ, 2)
    ∂_boundary::Array{ComplexF64,2}=zeros(ComplexF64, params.χ, 2)
    for m::UInt16 in 2:params.N-1
        #B = L_set[m]*R_set[params.N+1-m]
        for i::UInt8 in 1:params.χ
            for j::UInt8 in 1:params.χ 
                @inbounds ∂_bulk[i,j,2-sample[m]] += L_set[m-1][i]*R_set[params.N-m][j] #B[j,i] # + B[i,j]
            end
            #@inbounds ∂[i,i,:]./=2
        end
    end
    for i::UInt8 in 1:params.χ
        ∂_boundary[i,2-sample[1]] += R_set[params.N-1][i]
        ∂_boundary[i,2-sample[params.N]] += L_set[params.N-1][i]
    end
    return ∂_bulk, ∂_boundary
end