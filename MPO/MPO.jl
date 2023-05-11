export normalize_MPO


mutable struct density_matrix#{Coeff<:Int64, Vec<:Vector{Float64}}
    coeff::ComplexF64
    ket::Vector{Bool}
    bra::Vector{Bool}
    #ket::Vector{Int8}
    #bra::Vector{Int8}
    #density_matrix(coeff,ket,bra) = new(coeff,ket,bra)
end

Base.:*(x::density_matrix, y::density_matrix) = density_matrix(x.coeff * y.coeff, vcat(x.ket, y.ket), vcat(x.bra, y.bra))


mutable struct projector
    ket::Vector{Bool}
    bra::Vector{Bool}
end

projector(p::projector) = projector(copy(p.ket), copy(p.bra))

idx(sample::projector,i::UInt8) = 1+2*sample.ket[i]+sample.bra[i]


function MPO(params::parameters, sample::density_matrix, A::Array{ComplexF64})
    MPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    for i in 1:params.N
        MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
    end
    return tr(MPO)::ComplexF64
end

#Left strings of MPOs:
function L_MPO_strings(L_set, sample::projector, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, AUX::workspace)
    L_set[1] = AUX.ID
    for i::UInt8=1:params.N
        mul!(L_set[i+1], L_set[i], @view(A[:,:,idx(sample,i)]))
    end
    return L_set
end

#Right strings of MPOs:
function R_MPO_strings(R_set, sample::projector, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, AUX::workspace)
    R_set[1] = AUX.ID
    for i::UInt8=params.N:-1:1
        mul!(R_set[params.N+2-i], @view(A[:,:,idx(sample,i)]), R_set[params.N+1-i])
    end
    return R_set
end

function normalize_MPO(params::parameters, A::Array{<:Complex{<:AbstractFloat},3})
    #MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])^params.N
    MPO=(A[:,:,1]+A[:,:,4])^params.N
    return A./=tr(MPO)^(1/params.N)
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

#Computes the tensor of derivatives of variational parameters: 
function ∂MPO(sample::projector, L_set::Vector{<:Matrix{<:Complex{<:AbstractFloat}}}, 
    R_set::Vector{<:Matrix{<:Complex{<:AbstractFloat}}}, params::parameters, AUX::workspace)
    ∂::Array{eltype(L_set[1]),3} = zeros(eltype(L_set[1]), params.χ, params.χ, 4)
    for m::UInt8 in 1:params.N
        mul!(AUX.B,R_set[params.N+1-m],L_set[m])
        for i::UInt8=1:params.χ, j::UInt8=1:params.χ
            @inbounds ∂[i,j,idx(sample,m)] += AUX.B[j,i]
        end
    end
    return ∂
end