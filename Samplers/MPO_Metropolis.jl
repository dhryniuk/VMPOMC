export MetropolisSampler#, Metropolis_sweep_left!, Metropolis_burn_in!

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
function Metropolis_sweep_left!(sample::Projector, optimizer::Optimizer)

    A = optimizer.A
    params = optimizer.params
    ws=optimizer.workspace

    acc=0
    ws.R_set[1] = Matrix{eltype(A)}(I, params.χ, params.χ)
    ws.C_mat = ws.L_set[params.N+1]
    C = tr(ws.C_mat) #current probability amplitude

    for i::UInt8 in params.N:-1:1
        sample_p = Projector(sample)
        draw = draw_excluded(dINDEX[(sample.ket[i], sample.bra[i])])
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[draw]
        mul!(ws.Metro_1, ws.L_set[i], @view(A[:,:,draw]))
        mul!(ws.Metro_2, ws.Metro_1, ws.R_set[params.N+1-i])
        P=tr(ws.Metro_2) #proposal probability amplitude
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = Projector(sample_p)
            acc+=1
        end
        mul!(ws.R_set[params.N+2-i], @view(A[:,:,1+2*sample.ket[i]+sample.bra[i]]), ws.R_set[params.N+1-i])
        mul!(ws.C_mat, ws.L_set[i], ws.R_set[params.N+2-i])
        C = tr(ws.C_mat) #update current probability amplitude
    end
    return sample, acc
end

#Sweeps lattice from left to right
function Metropolis_sweep_right!(sample::Projector, optimizer::Optimizer)

    A = optimizer.A
    params = optimizer.params
    ws=optimizer.workspace

    acc=0
    ws.L_set[1] = Matrix{eltype(A)}(I, params.χ, params.χ)
    ws.C_mat = ws.R_set[params.N+1]
    C = tr(ws.C_mat) #current probability amplitude

    for i::UInt8 in 1:params.N
        sample_p = Projector(sample)
        draw = draw_excluded(dINDEX[(sample.ket[i], sample.bra[i])])
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[draw]
        mul!(ws.Metro_1, ws.L_set[i], @view(A[:,:,draw]))
        mul!(ws.Metro_2, ws.Metro_1, ws.R_set[params.N+1-i])
        P=tr(ws.Metro_2) #proposal probability amplitude
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = Projector(sample_p)
            acc+=1
        end
        mul!(ws.L_set[i+1], ws.L_set[i], @view(A[:,:,1+2*sample.ket[i]+sample.bra[i]]))
        mul!(ws.C_mat, ws.L_set[i+1], ws.R_set[params.N+1-i])
        C = tr(ws.C_mat) #update current probability amplitude
    end
    return sample, acc
end

function Metropolis_burn_in!(optimizer::Optimizer)

    A=optimizer.A
    params=optimizer.params
    ws=optimizer.workspace
    
    # Initialize random sample and calculate L_set for that sample:
    sample::Projector = Projector(rand(Bool, params.N),rand(Bool, params.N))
    ws.L_set = L_MPO_products!(ws.L_set, sample, A, params, ws)

    # Perform burn_in:
    for _ in 1:optimizer.sampler.burn
        sample,_ = Metropolis_sweep_left!(sample,optimizer)
        sample,_ = Metropolis_sweep_right!(sample,optimizer)
    end
    return sample
end