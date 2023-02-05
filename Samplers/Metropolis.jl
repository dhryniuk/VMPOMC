export Mono_Metropolis_sweep_left, Mono_Metropolis_sweep_right, local_Lindbladian, calculate_mean_local_Lindbladian,  MC_mean_local_Lindbladian

#Sweep lattice from right to left:
function Dual_Metropolis_sweep(sample::density_matrix, A::Array{ComplexF64}, L_set::Vector{Matrix{ComplexF64}})
    R_set = []
    R = Matrix{ComplexF64}(I, χ, χ)
    push!(R_set, copy(R))
    C = tr(L_set[N+1]) #Current MPO  ---> move into loop
    for i in N:-1:1

        #Update ket:
        sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra)) #deepcopy necessary?
        sample_p.ket[i] = 1-sample.ket[i]
        #P = tr(L_set[i]*A[:,:,dINDEX2[sample_p.ket[i]],dINDEX2[sample.bra[i]]])
        P = tr(L_set[i]*A[:,:,dINDEX[(sample_p.ket[i],sample.bra[i])]]*R_set[N+1-i])
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = sample_p
        end
        #aux_R = R*A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]] #auxiliary R
        aux_R = R*A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        C = tr(L_set[i]*aux_R)

        #Update bra:
        sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
        sample_p.bra[i] = 1-sample.bra[i]
        #P = tr(L_set[i]*A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample_p.bra[i]]])
        P = tr(L_set[i]*A[:,:,dINDEX[(sample.ket[i],sample_p.bra[i])]]*R_set[N+1-i])
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = sample_p
        end
        #R *= A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
        R *= A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        push!(R_set, copy(R))
        C = tr(L_set[i]*R)

    end
    return sample, R_set
end

function Mono_Metropolis_sweep_left(params::parameters, sample::density_matrix, A::Array{ComplexF64}, L_set::Vector{Matrix{ComplexF64}})

    function draw_excluded(u)
        v = rand(1:3)
        if v>=u
            v+=1
        end
        return v
    end

    R_set = Vector{Matrix{ComplexF64}}(undef,0)
    #R_set = []
    R = Matrix{ComplexF64}(I, params.χ, params.χ)
    push!(R_set, copy(R))
    C = tr(L_set[params.N+1]) #Current MPO  ---> move into loop
    for i in params.N:-1:1

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
        end

        R = A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]*R
        push!(R_set, copy(R))
        C = tr(L_set[i]*R)
    
    #    C = tr(L_set[i]*R_set[N+2-i])
    end
    return sample, R_set::Vector{Matrix{ComplexF64}}
end

function Mono_Metropolis_sweep_right(params::parameters, sample::density_matrix, A::Array{ComplexF64}, R_set::Vector{Matrix{ComplexF64}})

    function draw_excluded(u)
        v = rand(1:3)
        if v>=u
            v+=1
        end
        return v
    end

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
        end

        L*= A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        push!(L_set, copy(L))
        C = tr(L*R_set[params.N+1-i])

    end
    return sample, L_set::Vector{Matrix{ComplexF64}}
end


function Mono_Metropolis_sweep_left(params::parameters, sample::Vector{Bool}, A::Array{Float64}, L_set::Vector{Matrix{Float64}})

    #R_set = Vector{Matrix{Float64}}(undef,0)
    R_set = [ Matrix{Float64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    R = Matrix{Float64}(I, params.χ, params.χ)
    #push!(R_set, copy(R))
    R_set[1] = R
    C = tr(L_set[params.N+1]) #Current MPO  ---> move into loop
    for i in params.N:-1:1

        sample_p = deepcopy(sample) #deepcopy necessary?
        sample_p[i] = 1-sample[i]

        #P=MPS(params,sample_p,A)
        P = tr(L_set[i]*A[:,:,1+sample[i]]*R_set[params.N+1-i])
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            #sample = sample_p
            sample = deepcopy(sample_p)
        end

        #R = A[:,:,dINDEX2[sample[i]]]*R
        R = A[:,:,2-sample[i]]*R
        #push!(R_set, copy(R))
        R_set[params.N+2-i] = R
        C = tr(L_set[i]*R)
    end
    return sample, R_set::Vector{Matrix{Float64}}
end






#MOVE TO ANOTHER FILE:

function local_Lindbladian(params, l1,A,sample,L_set,R_set)

    local_L=0
    l_int = 0
    for j in 1:params.N

        #1-local part:
        s = dVEC[(sample.ket[j],sample.bra[j])]
        bra_L = transpose(s)*conj(l1)
        for i in 1:4
            loc = bra_L[i]
            if loc!=0
                state = TPSC[i]
                local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[params.N+1-j])
            end
        end

        #2-local part: #PBC
        l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,params.N)+1]-1)
        l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,params.N)+1]-1)
        l_int += 1.0im*params.J*(l_int_α-l_int_β)

    end

    local_L/=tr(R_set[params.N+1])
    local_L+=l_int

    return local_L
end

function calculate_mean_local_Lindbladian(params::parameters, l1, A, basis)
    mll=0
    Z=0
    for k in 1:params.dim
        for l in 1:params.dim
            sample = density_matrix(1,basis[k],basis[l]) 
            ρ_sample = MPO(params,sample,A)
            p_sample = ρ_sample*conj(ρ_sample)
            Z+=p_sample

            L_set = L_MPO_strings(params,sample,A)
            R_set = R_MPO_strings(params,sample,A)

            local_L = local_Lindbladian(params,l1,A,sample,L_set,R_set)

            mll+=p_sample*local_L*conj(local_L)
        end
    end

    return real(mll/Z)
end

function MC_mean_local_Lindbladian(J,A,N_MC)
    mll=0
    sample = density_matrix(1,basis[1],basis[1]) 
    L_set = L_MPO_strings(sample,A)
    R_set = R_MPO_strings(sample,A)
    for k in 1:N_MC
        sample, R_set = Mono_Metropolis_sweep_left(sample, A, L_set)
        #sample, L2_set = Mono_Metropolis_sweep_right(sample, A, R_set)
        #sample = Single_Metropolis(sample, A, rand(1:N))
        #for i in 1:N
        #    sample = Single_Metropolis(sample, A, i)
        #end
        L_set = L_MPO_strings(sample,A)
        #R_set = R_MPO_strings(sample,A)
        #println(R2_set==R_set)

        local_L = local_Lindbladian(J,A,sample,L_set,R_set)

        mll+=local_L*conj(local_L)
    end

    return mll/N_MC
end