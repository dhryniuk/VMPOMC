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
function L_MPO_strings(sample::projector, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, AUX::workspace)
    AUX.micro_L_set[1] = AUX.ID
    for i::UInt8 in 1:params.N
        mul!(AUX.micro_L_set[i+1], AUX.micro_L_set[i], @view(A[:,:,1+2*sample.ket[i]+sample.bra[i]]))
    end
    return AUX.micro_L_set#::Vector{Matrix{eltype(A),3}}
end

#Right strings of MPOs:
function R_MPO_strings(sample::projector, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, AUX::workspace)
    AUX.micro_R_set[1] = AUX.ID
    for i::UInt8 in params.N:-1:1
        mul!(AUX.micro_R_set[params.N+2-i], @view(A[:,:,1+2*sample.ket[i]+sample.bra[i]]), AUX.micro_R_set[params.N+1-i])
    end
    return AUX.micro_R_set
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

function B_list(m, sample::density_matrix, A::Array{ComplexF64}) #FIX m ORDERING
    B_list=Matrix{ComplexF64}[Matrix{Int}(I, χ, χ)]
    for j::UInt8 in 1:params.N-1
        push!(B_list,A[:,:,dINDEX[(sample.ket[mod(m+j-1,N)+1],sample.bra[mod(m+j-1,N)+1])]])
    end
    return B_list
end

function ∂MPO(sample::projector, L_set::Vector{<:Matrix{<:Complex{<:AbstractFloat}}}, 
    R_set::Vector{<:Matrix{<:Complex{<:AbstractFloat}}}, params::parameters, AUX::workspace)
    ∂::Array{eltype(L_set[1]),3} = zeros(eltype(L_set[1]), params.χ, params.χ, 4)
    for m::UInt8 in 1:params.N
        mul!(AUX.B,R_set[params.N+1-m],L_set[m])
        for i::UInt8=1:params.χ, j::UInt8=1:params.χ
            @inbounds ∂[i,j,1+2*sample.ket[m]+sample.bra[m]] += AUX.B[j,i]
        end
    end
    return ∂
end