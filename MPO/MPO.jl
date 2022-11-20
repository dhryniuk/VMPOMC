export MPO, MPO_string, L_MPO_strings, R_MPO_strings, normalize_MPO, B_list, derv_MPO, OLDderv_MPO, Δ_MPO

function MPO(sample, A)
    MPO=Matrix{ComplexF64}(I, χ, χ)
    for i::UInt8 in 1:N
        MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
    end
    return tr(MPO)::ComplexF64
end

#MPO string beginning as site l and ending at site r:
function MPO_string(sample,A,l,r)
    MPO=Matrix{ComplexF64}(I, χ, χ)
    for i in l:r
        MPO*=A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
    end
    return MPO
end

#Left strings of MPOs:
function L_MPO_strings(sample, A)
    L = Vector{Matrix{ComplexF64}}()
    MPO=Matrix{ComplexF64}(I, χ, χ)
    push!(L,copy(MPO))
    for i::UInt8 in 1:N
        MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        #MPO*=A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
        push!(L,copy(MPO))
    end
    return L
end

#Right strings of MPOs:
function R_MPO_strings(sample, A)
    R = Vector{Matrix{ComplexF64}}()
    MPO=Matrix{ComplexF64}(I, χ, χ)
    push!(R,copy(MPO))
    for i::UInt8 in N:-1:1
        MPO=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]*MPO

        # MATRIX MULTIPLICATION IS NOT COMMUTATIVE, IDIOT

        #MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        #MPO*=A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
        push!(R,copy(MPO))
    end
    return R
end

function normalize_MPO(A)
    MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])^N
    return tr(MPO)^(1/N)#::ComplexF64
end

function B_list(m, sample, A) #FIX m ORDERING
    B_list=Matrix{ComplexF64}[Matrix{Int}(I, χ, χ)]
    for j::UInt8 in 1:N-1
        push!(B_list,A[:,:,dINDEX[(sample.ket[mod(m+j-1,N)+1],sample.bra[mod(m+j-1,N)+1])]])
    end
    return B_list
end

function derv_MPO(sample, L_set, R_set)
    ∇=zeros(ComplexF64, χ,χ,4)
    #L_set = L_MPO_strings(sample, A)
    #R_set = R_MPO_strings(sample, A)
    for m::UInt8 in 1:N
        B = R_set[N+1-m]*L_set[m]
        for i in 1:χ
            for j in 1:χ
                ∇[i,j,dINDEX[(sample.ket[m],sample.bra[m])]] += B[i,j] + B[j,i]
            end
            ∇[i,i,:]./=2
        end
    end
    return ∇
end

function double_bond_dimension(A)
    global χ*=2
    B = zeros(ComplexF64, χ,χ,4)
    for i in 1:4
        B[:,:,i] = kron(A[:,:,i],[1 1; 1 1])
    end
    return B
end