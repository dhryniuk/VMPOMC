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

Base.display(sampler::MetropolisSampler) = begin
    println("\nSampler:")
    println("N_MC\t\t", sampler.N_MC)
    println("burn\t\t", sampler.burn)
end

#Sweeps lattice from right to left
#function Mono_Metropolis_sweep_left(sample::Projector, A::Array{<:Complex{<:AbstractFloat},3}, params::Parameters, cache::Workspace)
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
#function Mono_Metropolis_sweep_right(sample::Projector, A::Array{<:Complex{<:AbstractFloat},3}, params::Parameters, cache::Workspace)
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

function reweighted_Mono_Metropolis_sweep_left(β::Float64, sample::Projector, A::Array{<:Complex{<:AbstractFloat},3}, params::Parameters, cache::Workspace)

    acc=0

    #cache.R_set::Vector{Matrix{eltype(A)}} = [ Matrix{eltype(A)}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
    cache.R_set[1] = Matrix{eltype(A)}(I, params.χ, params.χ)
    cache.C_mat = cache.L_set[params.N+1]
    C = tr(cache.C_mat)

    for i::UInt8 in params.N:-1:1
        sample_p = Projector(sample)
        draw = draw_excluded(dINDEX[(sample.ket[i],sample.bra[i])])
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[draw]
        mul!(cache.Metro_1,cache.L_set[i],@view(A[:,:,draw]))
        mul!(cache.Metro_2,cache.Metro_1,cache.R_set[params.N+1-i])
        P=tr(cache.Metro_2)
        metropolis_prob = real((P*conj(P)))^(β)/real((C*conj(C)))^(β)
        if rand() <= metropolis_prob
            sample = Projector(sample_p) #replace with =sample_p?
            acc+=1
        end
        mul!(cache.R_set[params.N+2-i], @view(A[:,:,1+2*sample.ket[i]+sample.bra[i]]), cache.R_set[params.N+1-i])
        mul!(cache.C_mat, cache.L_set[i], cache.R_set[params.N+2-i])
        C = tr(cache.C_mat)
    end
    return sample, acc
end



#function MPO_Metropolis_burn_in(A::Array{<:Complex{<:AbstractFloat},3}, params::Parameters, cache::Workspace)
function MPO_Metropolis_burn_in(optimizer::Optimizer)

    A=optimizer.A
    params=optimizer.params
    cache=optimizer.workspace
    
    # Initialize random sample and calculate L_set for that sample:
    sample::Projector = Projector(rand(Bool, params.N),rand(Bool, params.N))
    cache.L_set = L_MPO_strings!(cache.L_set, sample, A, params, cache)
    
    #acce1=0
    #acce2=0

    # Perform burn_in:
    for _ in 1:optimizer.sampler.burn#params.burn_in
        sample,_ = Mono_Metropolis_sweep_left(sample,optimizer)
        sample,_ = Mono_Metropolis_sweep_right(sample,optimizer)
        #sample,_ = Mono_Metropolis_sweep_left(sample,A,params,cache)
        #acce1+=acc1
        #sample,_ = Mono_Metropolis_sweep_right(sample,A,params,cache)
        #acce2+=acc2
    end

    #println(acce1)
    #println(acce2)
    #error()

    return sample#, cache.L_set
end

export parity_conserving_metropolis_sweep_left

function parity_conserving_metropolis_sweep_left(sample::Projector, optimizer::Optimizer)

    params = optimizer.params

    C = MPO(params, sample, optimizer.A)

    for i::UInt8 in params.N:-1:1
        sample_p = Projector(sample)
        if sample.ket[i]==sample.bra[i]
            sample_p.ket[i]=!sample_p.ket[i]
            sample_p.bra[i]=!sample_p.bra[i]
            println(sample_p)
            P = MPO(params, sample_p, optimizer.A)
            metropolis_prob = real((P*conj(P))/(C*conj(C)))
            if rand() <= metropolis_prob
                sample = Projector(sample_p)
                C = P
            end
        else 
            if rand()<=0.5
                sample_p.ket[mod(i-1-1,params.N)+1] = sample.bra[i]
                sample_p.bra[i] = sample.ket[mod(i-1-1,params.N)+1]
            else
                sample_p.bra[mod(i-1-1,params.N)+1] = sample.ket[i]
                sample_p.ket[i] = sample.bra[mod(i-1-1,params.N)+1]
            end
            println(sample_p)
            P = MPO(params, sample_p, optimizer.A)
            metropolis_prob = real((P*conj(P))/(C*conj(C)))
            if rand() <= metropolis_prob
                sample = Projector(sample_p)
                C = P
            end
        end
    end
    return sample
end

export parity

function parity(sample::Projector)
    N = length(sample.ket)
    P_ket = 1
    P_bra = 1
    for i in 1:N
        if sample.ket[i]==false
            P_ket*=-1
        end
        if sample.bra[i]==false
            P_bra*=-1
        end
    end
    return P_ket*P_bra
end

export mag_diff_conserving_metropolis_sweep_left

function mag_diff_conserving_metropolis_sweep_left(sample::Projector, optimizer::Optimizer)

    params = optimizer.params

    C = MPO(params, sample, optimizer.A)

    for i::UInt8 in params.N:-1:1
        sample_p = Projector(sample)

        #swapping move:
        if rand() <= 0.5
            #println("SWAP")
            if rand()<=0.5
                sample_p.ket[mod(i-1-1,params.N)+1] = sample.ket[i]
                sample_p.ket[i] = sample.ket[mod(i-1-1,params.N)+1]
            else
                sample_p.bra[mod(i-1-1,params.N)+1] = sample.bra[i]
                sample_p.bra[i] = sample.bra[mod(i-1-1,params.N)+1]
            end
        #flipping move:
        else
            #println("FLIP")
            if sample.ket[i]==sample.bra[mod(i-1-1,params.N)+1]
                sample_p.ket[i]=!sample_p.ket[i]
                sample_p.bra[mod(i-1-1,params.N)+1]=!sample_p.bra[mod(i-1-1,params.N)+1]
            end
            if sample.bra[i]==sample.ket[mod(i-1-1,params.N)+1]
                sample_p.bra[i]=!sample_p.bra[i]
                sample_p.ket[mod(i-1-1,params.N)+1]=!sample_p.ket[mod(i-1-1,params.N)+1]
            end
        end
        #print_canonical(sample_p)
        P = MPO(params, sample_p, optimizer.A)
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = Projector(sample_p)
            C = P
            #println("ACCEPTED")
        end
    end
    return sample
end

export mag_diff

function mag_diff(sample::Projector)
    N = length(sample.ket)
    M_ket = 0
    M_bra = 0
    for i in 1:N
        if sample.ket[i]==false
            M_ket-=1
        else
            M_ket+=1
        end
        if sample.bra[i]==false
            M_bra-=1
        else
            M_bra+=1
        end
    end
    return M_ket-M_bra
end