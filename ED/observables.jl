export magnetization, purity, spin_spin_correlation, occupation, g2, sssf


function magnetization(op,ρ,params)
    first_term_ops = fill(id, params.N)
    first_term_ops[1] = op

    m::ComplexF64=0
    for _ in 1:params.N
        m += tr(ρ*foldl(⊗, first_term_ops))
        first_term_ops = circshift(first_term_ops,1)
    end

    return m/params.N
end

function purity(ρ)
    return tr(ρ*adjoint(ρ))
end

function spin_spin_correlation(op,ρ,params)
    corr = 0
    for j in 2:params.N÷2+1
        ops = fill(id, params.N)
        ops[1] = op
        ops[2] = op
        corr = real(tr(ρ*foldl(⊗, ops)))
    end
    return corr
end

"""
function spin_spin_correlation(op,ρ,params)
    corr = zeros(Float64,params.N÷2)
    for j in 2:params.N÷2+1
        ops = fill(id, params.N)
        ops[1] = op
        ops[j] = op
        corr[j-1] = real(tr(ρ*foldl(⊗, ops)))
    end
    return corr
end
"""

function sssf(op,ρ,params)
    N = params.N
    corr = 0
    corr += sum(spin_spin_correlation(op,ρ,params))
    return corr/(N*(N-1))
end

function occupation(ρ,params)
    first_term_ops = fill(id, params.N)
    first_term_ops[1] = sp_sp*sp_sm

    return tr(ρ*foldl(⊗, first_term_ops))
end

function g2(ρ,params)
    denom = fill(id, params.N)
    denom[1] = sp_sp*sp_sm
    D = real(tr(ρ*foldl(⊗, denom)))

    #println(D)

    g2_d = zeros(Float64,params.N-1)
    for d in 2:params.N
        ops = fill(id, params.N)
        ops[1] = sp_sp*sp_sm
        ops[d] = sp_sp*sp_sm
        g2_d[d-1] = real(tr(ρ*foldl(⊗, ops)))
    end
    return g2_d./D^2
end

export generate_bit_basis, generate_bit_basis_reversed

#Ising bit-basis:
function generate_bit_basis(N)#(N::UInt8)
    set::Vector{Vector{Bool}} = [[true], [false]]
    @simd for i in 1:N-1
        new_set::Vector{Vector{Bool}} = []
        @simd for state in set
            state2::Vector{Bool} = copy(state)
            state = vcat(state, true)
            state2 = vcat(state2, false)
            push!(new_set, state)
            push!(new_set, state2)
        end
        set = new_set
    end
    return Vector{Vector{Bool}}(set)
end

#Ising bit-basis:
function generate_bit_basis_reversed(N)#(N::UInt8)
    set::Vector{Vector{Bool}} = [[false], [true]]
    @simd for i in 1:N-1
        new_set::Vector{Vector{Bool}} = []
        @simd for state in set
            state2::Vector{Bool} = copy(state)
            state = vcat(state, false)
            state2 = vcat(state2, true)
            push!(new_set, state)
            push!(new_set, state2)
        end
        set = new_set
    end
    return Vector{Vector{Bool}}(set)
end

export reduced

function reduced(ρ)
    ρ_red = zeros(ComplexF64, 2,2)
    l = size(ρ)[1]
    for i in 1:2, j in 1:2
        for k in 1:l÷2
            ρ_red[i,j]+=ρ[(i-1)*l÷2+k,(j-1)*l÷2+k]
        end
    end
    return ρ_red
end

using BlockArrays

function partial_transpose(rho, d1, d2)
    idx = [d2 for i = 1:d1]
    blkm = BlockArray(rho, idx, idx)
    for i = 1:d1
        for j = 1:d1
            bfm = blkm[Block(i, j)]
            trm = transpose(bfm)
            blkm[Block(i, j)] = trm
        end
    end
    Array(blkm)
end

export negativity

function negativity(ρ)
    ρ_PT = partial_transpose(ρ,2,2)
    evals, evecs = eigen(ρ_PT)
    display(evals)
    ρ_red = reduced(ρ)
    display(ρ_red)
    println(-real(tr(ρ_PT*log(ρ_PT))))
    println(tr(sqrt(adjoint(ρ_PT)*ρ_PT)))
    return (tr(sqrt(adjoint(ρ_PT)*ρ_PT))-1)/2
end