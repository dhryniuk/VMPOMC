export MetropolisSampler, Mono_Metropolis_sweep_left, MPO_Metropolis_burn_in

function draw_excluded(u::Int8)
    v::Int8 = rand(1:3)
    if v>=u
        v+=1
    end
    return v
end

struct MetropolisSampler
    N_MC::UInt64
    burn::UInt64
end

Base.display(sampler::MetropolisSampler) = begin
    println("\nSampler:")
    println("N_MC\t\t", sampler.N_MC)
    println("burn\t\t", sampler.burn)
end

#Sweeps lattice from right to left
function Mono_Metropolis_sweep_left(sample::Projector, optimizer::Optimizer)

    A = optimizer.A
    params = optimizer.params
    cache=optimizer.workspace

    acc=0
    cache.R_set[1] = Matrix{eltype(A)}(I, params.χ, params.χ)
    cache.C_mat = cache.L_set[params.N+1]
    C = tr(cache.C_mat) #current probability amplitude

    for i::UInt8 in params.N:-1:1
        sample_p = Projector(sample)
        draw = draw_excluded(dINDEX[(sample.ket[i],sample.bra[i])])
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[draw]
        mul!(cache.Metro_1,cache.L_set[i],@view(A[:,:,draw]))
        mul!(cache.Metro_2,cache.Metro_1,cache.R_set[params.N+1-i])
        P=tr(cache.Metro_2) #proposal probability amplitude
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = Projector(sample_p)
            acc+=1
        end
        mul!(cache.R_set[params.N+2-i], @view(A[:,:,1+2*sample.ket[i]+sample.bra[i]]), cache.R_set[params.N+1-i])
        mul!(cache.C_mat, cache.L_set[i], cache.R_set[params.N+2-i])
        C = tr(cache.C_mat) #update current probability amplitude
    end
    return sample, acc
end

#Sweeps lattice from left to right
function Mono_Metropolis_sweep_right(sample::Projector, optimizer::Optimizer)

    A = optimizer.A
    params = optimizer.params
    cache=optimizer.workspace

    acc=0
    cache.L_set[1] = Matrix{eltype(A)}(I, params.χ, params.χ)
    cache.C_mat = cache.R_set[params.N+1]
    C = tr(cache.C_mat) #current probability amplitude

    for i::UInt8 in 1:params.N
        sample_p = Projector(sample)
        draw = draw_excluded(dINDEX[(sample.ket[i],sample.bra[i])])
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[draw]
        mul!(cache.Metro_1,cache.L_set[i],@view(A[:,:,draw]))
        mul!(cache.Metro_2,cache.Metro_1,cache.R_set[params.N+1-i])
        P=tr(cache.Metro_2) #proposal probability amplitude
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = Projector(sample_p)
            acc+=1
        end
        mul!(cache.L_set[i+1], cache.L_set[i], @view(A[:,:,1+2*sample.ket[i]+sample.bra[i]]))
        mul!(cache.C_mat, cache.L_set[i+1], cache.R_set[params.N+1-i])
        C = tr(cache.C_mat) #update current probability amplitude
    end
    return sample, acc
end

function MPO_Metropolis_burn_in(optimizer::Optimizer)

    A=optimizer.A
    params=optimizer.params
    cache=optimizer.workspace
    
    # Initialize random sample and calculate L_set for that sample:
    sample::Projector = Projector(rand(Bool, params.N),rand(Bool, params.N))
    cache.L_set = L_MPO_strings!(cache.L_set, sample, A, params, cache)

    # Perform burn_in:
    for _ in 1:optimizer.sampler.burn
        sample,_ = Mono_Metropolis_sweep_left(sample,optimizer)
        sample,_ = Mono_Metropolis_sweep_right(sample,optimizer)
    end
    return sample
end