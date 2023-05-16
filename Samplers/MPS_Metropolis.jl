export Mono_Metropolis_sweep_left, Metropolis_burn_in

function Mono_Metropolis_sweep_left(params::parameters, sample::Vector{Bool}, A::Array{Float64}, L_set::Vector{Matrix{Float64}})
    acc::UInt16=0
    R_set = [ Matrix{Float64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    R = Matrix{Float64}(I, params.χ, params.χ)
    R_set[1] = R
    #display(L_set[params.N+1])
    #error()
    C = tr(L_set[params.N+1]) #Current MPO  ---> move into loop
    for i in params.N:-1:1

        sample_p = deepcopy(sample) #deepcopy necessary?
        sample_p[i] = 1-sample[i]

        #P = tr(L_set[i]*A[:,:,1+sample[i]]*R_set[params.N+1-i])
        P = tr(L_set[i]*A[:,:,dINDEX2[sample_p[i]]]*R_set[params.N+1-i])
        #println(C)
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        #println(metropolis_prob)
        if rand() <= metropolis_prob
            #sample = sample_p
            sample = deepcopy(sample_p)
            acc+=1
        end

        R = A[:,:,2-sample[i]]*R
        R_set[params.N+2-i] = R
        C = tr(L_set[i]*R)
    end
    return sample, R_set::Vector{Matrix{Float64}}, acc
end

function Metropolis_burn_in(p::parameters, A::Array{Float64,3})
    
    # Initialize random sample and calculate L_set for that sample:
    sample = rand(Bool, p.N)
    L_set = L_MPS_strings(p, sample, A)
    
    # Perform burn_in:
    for _ in 1:p.burn_in
        sample, R_set = Mono_Metropolis_sweep_left(p,sample,A,L_set)
        sample, L_set = Mono_Metropolis_sweep_right(p,sample,A,R_set)
    end

    return sample, L_set
end
