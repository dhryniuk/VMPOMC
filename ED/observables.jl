export magnetization, spin_current, spin_spin_correlation, steady_state_structure_factor


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

function steady_state_structure_factor(ρ,params)
    sssf = 0
    for j in 1:params.N
        for l in 1:params.N
            if l!=j
                sssf+= spin_spin_correlation(j,l,sx,ρ,params)
            end
        end
    end
    return sssf/(params.N*(params.N-1))
end
