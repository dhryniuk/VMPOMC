export magnetization, spin_current, spin_spin_correlation


function magnetization(op,ρ,params)
    ops = fill(id, params.N)
    ops[1] = op

    m::ComplexF64=0
    for _ in 1:params.N
        m += tr(ρ*foldl(⊗, ops))
        ops = circshift(ops,1)
    end

    return m/params.N
end

function spin_spin_correlation(i,j,op,ρ,params)
    @assert i!=j
    ops = fill(id, params.N)
    ops[i] = op
    ops[j] = op
    return tr(ρ*foldl(⊗, ops))
end

function spin_spin_correlation(op,ρ,params)
    corr = zeros(Float64,params.N-1)
    for j in 2:params.N
        ops = fill(id, params.N)
        ops[1] = op
        ops[j] = op
        corr[j-1] = real(tr(ρ*foldl(⊗, ops)))
    end
    return corr
end

function spin_current(ρ,params,j,k)
    ops1 = fill(id, params.N)
    ops1[j] = sp
    ops1[k] = sm

    ops2 = fill(id, params.N)
    ops2[j] = sm
    ops2[k] = sp

    I = tr(ρ*1im*(foldl(⊗, ops1)-foldl(⊗, ops2)))
    return I
end

