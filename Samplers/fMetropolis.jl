export XMono_Metropolis_sweep_left, Mono_Metropolis_sweep_left

function explicit_Mono_Metropolis_sweep_left(params::parameters, sample::Vector{Bool}, A::Array{Float64,3}, V::Array{Float64})
    acc=0
    for i in params.N:-1:1
        C = fMPS(params,sample,A,V)

        sample_p = deepcopy(sample) #deepcopy necessary?
        sample_p[i] = 1-sample[i]

        P = fMPS(params,sample_p,A,V)
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = deepcopy(sample_p)
            acc+=1
        end
    end
    #display(sample)

    return sample#,acc
end

function Mono_Metropolis_sweep_left(params::parameters, sample::Vector{Bool}, A::Array{Float64,3}, V::Array{Float64}, 
    L_set::Vector{Transpose{Float64, Vector{Float64}}})

    acc::UInt16=0
    R_set = [ Vector{Float64}(undef,params.χ) for _ in 1:params.N-1 ]

    #right boundary only first:
    sample_p = deepcopy(sample) #deepcopy necessary?
    sample_p[params.N] = 1-sample[params.N]

    #display(L_set)
    #display(V)
    #C = L_set[params.N-1]*V[:,dINDEX2[sample[params.N]]]
    C = fMPS(params,sample,A,V)
    P = L_set[params.N-1]*V[:,dINDEX2[sample_p[params.N]]]
    metropolis_prob = real((P*conj(P))/(C*conj(C)))
    if rand() <= metropolis_prob
        #sample = sample_p
        sample = deepcopy(sample_p)
        acc+=1
    end
    R = V[:,dINDEX2[sample[params.N]]]
    R_set[1] = R
    C = L_set[params.N-1]*R

    #bulk:
    for i in params.N-1:-1:2

        sample_p = deepcopy(sample) #deepcopy necessary?
        sample_p[i] = 1-sample[i]

        P = L_set[i-1]*A[:,:,dINDEX2[sample_p[i]]]*R_set[params.N-i]
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            #sample = sample_p
            sample = deepcopy(sample_p)
            acc+=1
        end

        R = A[:,:,dINDEX2[sample[i]]]*R
        R_set[params.N+1-i] = R
        C = L_set[i-1]*R
    end

    #left boundary last:
    sample_p = deepcopy(sample) #deepcopy necessary?
    sample_p[1] = 1-sample[1]

    P = transpose(V[:,dINDEX2[sample_p[1]]])*R_set[params.N-1]
    metropolis_prob = real((P*conj(P))/(C*conj(C)))
    if rand() <= metropolis_prob
        #sample = sample_p
        sample = deepcopy(sample_p)
        acc+=1
    end

    return sample, R_set::Vector{Vector{Float64}}, acc
end
