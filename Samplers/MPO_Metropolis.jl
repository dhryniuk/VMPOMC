export Mono_Metropolis_sweep_left, MPO_Metropolis_burn_in, reweighted_Mono_Metropolis_sweep_left #, Mono_Metropolis_sweep_right

function draw_excluded(u::Int8)
    v::Int8 = rand(1:3)
    if v>=u
        v+=1
    end
    return v
end

export MetropolisSampler

struct MetropolisSampler
    N_MC::UInt64
    burn::UInt64
end

#Sweeps lattice from right to left
function Mono_Metropolis_sweep_left(sample::projector, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, cache::workspace)

    acc=0
    cache.R_set[1] = Matrix{eltype(A)}(I, params.χ, params.χ)
    cache.C_mat = cache.L_set[params.N+1]
    C = tr(cache.C_mat) #current probability amplitude

    for i::UInt8 in params.N:-1:1
        sample_p = projector(sample)
        draw = draw_excluded(dINDEX[(sample.ket[i],sample.bra[i])])
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[draw]
        mul!(cache.Metro_1,cache.L_set[i],@view(A[:,:,draw]))
        mul!(cache.Metro_2,cache.Metro_1,cache.R_set[params.N+1-i])
        P=tr(cache.Metro_2) #proposal probability amplitude
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = projector(sample_p)
            acc+=1
        end
        mul!(cache.R_set[params.N+2-i], @view(A[:,:,1+2*sample.ket[i]+sample.bra[i]]), cache.R_set[params.N+1-i])
        mul!(cache.C_mat, cache.L_set[i], cache.R_set[params.N+2-i])
        C = tr(cache.C_mat) #update current probability amplitude
    end
    return sample, acc
end

#Sweeps lattice from left to right
function Mono_Metropolis_sweep_right(sample::projector, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, cache::workspace)

    acc=0
    cache.L_set[1] = Matrix{eltype(A)}(I, params.χ, params.χ)
    cache.C_mat = cache.R_set[params.N+1]
    C = tr(cache.C_mat) #current probability amplitude

    for i::UInt8 in 1:params.N
        sample_p = projector(sample)
        draw = draw_excluded(dINDEX[(sample.ket[i],sample.bra[i])])
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[draw]
        mul!(cache.Metro_1,cache.L_set[i],@view(A[:,:,draw]))
        mul!(cache.Metro_2,cache.Metro_1,cache.R_set[params.N+1-i])
        P=tr(cache.Metro_2) #proposal probability amplitude
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = projector(sample_p)
            acc+=1
        end
        mul!(cache.L_set[i+1], cache.L_set[i], @view(A[:,:,1+2*sample.ket[i]+sample.bra[i]]))
        mul!(cache.C_mat, cache.L_set[i+1], cache.R_set[params.N+1-i])
        C = tr(cache.C_mat) #update current probability amplitude
    end
    return sample, acc
end

function reweighted_Mono_Metropolis_sweep_left(β::Float64, sample::projector, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, cache::workspace)

    acc=0

    #cache.R_set::Vector{Matrix{eltype(A)}} = [ Matrix{eltype(A)}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
    cache.R_set[1] = Matrix{eltype(A)}(I, params.χ, params.χ)
    cache.C_mat = cache.L_set[params.N+1]
    C = tr(cache.C_mat)

    for i::UInt8 in params.N:-1:1
        sample_p = projector(sample)
        draw = draw_excluded(dINDEX[(sample.ket[i],sample.bra[i])])
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[draw]
        mul!(cache.Metro_1,cache.L_set[i],@view(A[:,:,draw]))
        mul!(cache.Metro_2,cache.Metro_1,cache.R_set[params.N+1-i])
        P=tr(cache.Metro_2)
        metropolis_prob = real((P*conj(P)))^(β)/real((C*conj(C)))^(β)
        if rand() <= metropolis_prob
            sample = projector(sample_p) #replace with =sample_p?
            acc+=1
        end
        mul!(cache.R_set[params.N+2-i], @view(A[:,:,1+2*sample.ket[i]+sample.bra[i]]), cache.R_set[params.N+1-i])
        mul!(cache.C_mat, cache.L_set[i], cache.R_set[params.N+2-i])
        C = tr(cache.C_mat)
    end
    return sample, acc
end

"""
function Mono_Metropolis_sweep_right(params::parameters, sample::density_matrix, A::Array{ComplexF64}, R_set::Vector{Matrix{ComplexF64}})

    function draw_excluded(u)
        v = rand(1:3)
        if v>=u
            v+=1
        end
        return v
    end

    acc = 0

    L_set = Vector{Matrix{ComplexF64}}(undef,0)
    L = Matrix{ComplexF64}(I, params.χ, params.χ)
    push!(L_set, copy(L))
    C = tr(R_set[params.N+1]) #Current MPO  ---> move into loop
    for i in 1:params.N

        sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra)) #deepcopy necessary?
        u = dINDEX[(sample.ket[i],sample.bra[i])]
        v = draw_excluded(u)
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[v]
        P = tr(L_set[i]*A[:,:,v]*R_set[params.N+1-i])
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            #sample = sample_p
            sample.ket = deepcopy(sample_p.ket)
            sample.bra = deepcopy(sample_p.bra)
            acc+=1
        end

        L*= A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        push!(L_set, copy(L))
        C = tr(L*R_set[params.N+1-i])

    end
    return sample, L_set::Vector{Matrix{ComplexF64}}, acc
end
"""


function MPO_Metropolis_burn_in(A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, cache::workspace)
    
    # Initialize random sample and calculate L_set for that sample:
    sample::projector = projector(rand(Bool, params.N),rand(Bool, params.N))
    cache.L_set = L_MPO_strings!(cache.L_set, sample, A, params, cache)
    
    #acce1=0
    #acce2=0

    # Perform burn_in:
    for _ in 1:params.burn_in
        sample,_ = Mono_Metropolis_sweep_left(sample,A,params,cache)
        #acce1+=acc1
        sample,_ = Mono_Metropolis_sweep_right(sample,A,params,cache)
        #acce2+=acc2
    end

    #println(acce1)
    #println(acce2)
    #error()

    return sample#, cache.L_set
end