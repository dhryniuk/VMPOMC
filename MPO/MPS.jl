export normalize_MPS

function MPS(params::parameters, sample::Vector{Bool}, A::Array{Float64})
    MPS=Matrix{Float64}(I, params.χ, params.χ)
    for i::UInt8 in 1:params.N
        MPS*=A[:,:,dINDEX2[sample[i]]]
    end
    return tr(MPS)::Float64
end

#Left strings of MPSs:
function L_MPS_strings(params::parameters, sample::Vector{Bool}, A::Array{Float64})
    MPS=Matrix{Float64}(I, params.χ, params.χ)
    #L = Vector{Matrix{Float64}}()
    #push!(L,copy(MPS))
    L = [ Matrix{Float64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    L[1] = MPS
    for i::UInt8 in 1:params.N
        MPS*=A[:,:,dINDEX2[sample[i]]]
        L[i+1] = MPS
        #push!(L,copy(MPS))
    end
    return L
end

#Right strings of MPSs:
function R_MPS_strings(params::parameters, sample::Vector{Bool}, A::Array{Float64})
    #R = Vector{Matrix{Float64}}()
    MPS=Matrix{Float64}(I, params.χ, params.χ)
    #push!(R,copy(MPS))
    R = [ Matrix{Float64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    R[1] = MPS
    for i::UInt8 in params.N:-1:1
        MPS=A[:,:,dINDEX2[sample[i]]]*MPS
        R[i+1] = MPS
        #push!(R,copy(MPS))
    end
    return R
end

function derv_MPS(params::parameters, sample::Vector{Bool}, L_set::Vector{Matrix{Float64}}, R_set::Vector{Matrix{Float64}})
    ∇::Array{Float64,3}=zeros(Float64, params.χ, params.χ, 2)
    #L_set = L_MPS_strings(params, sample, A)
    #R_set = R_MPS_strings(params, sample, A)
    for m::UInt8 in 1:params.N
        B = R_set[params.N+1-m]*L_set[m]
        #for i::UInt8 in 1:params.χ
        #    for j::UInt8 in 1:params.χ
        @inbounds for (i::UInt8,j::UInt8) in zip(1:params.χ,1:params.χ)
            #∇[i,j,dINDEX2[sample[m]]] += B[i,j] + B[j,i]
            ∇[i,j,2-sample[m]] += B[i,j] + B[j,i]
        end
        for i in 1:params.χ
            ∇[i,i,:]./=2
        end
    end
    return ∇
end

function normalize_MPS(params::parameters, A::Array{Float64})
    MPS=(A[:,:,dINDEX2[1]]+A[:,:,dINDEX2[0]])^params.N
    return tr(MPS)^(1/params.N)#::ComplexF64
end