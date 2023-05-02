export Mono_Metropolis_sweep_left, Mono_Metropolis_sweep_right, local_Lindbladian, calculate_mean_local_Lindbladian,  MC_mean_local_Lindbladian

#Sweeps lattice from right to left
function Mono_Metropolis_sweep_left(sample::projector, A::Array{<:Complex{<:AbstractFloat},3}, 
    L_set::Vector{<:Matrix{<:Complex{<:AbstractFloat}}}, params::parameters, AUX::workspace)

    function draw_excluded(u)
        v::Int8 = rand(1:3)
        if v>=u
            v+=1
        end
        return v
    end

    acc=0

    R_set::Vector{Matrix{eltype(A)}} = [ Matrix{eltype(A)}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
    #R_set = [ Matrix{<:Complex{<:AbstractFloat}}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]

    R_set[1] = Matrix{eltype(A)}(I, params.χ, params.χ)
    AUX.C_mat = L_set[params.N+1]
    C = tr(AUX.C_mat)

    for i::UInt8 in params.N:-1:1
        #sample_p = density_matrix(1,copy(sample.ket),copy(sample.bra)) 
        #sample_p = projector(copy(sample.ket),copy(sample.bra)) 
        sample_p = projector(sample)
        draw = draw_excluded(dINDEX[(sample.ket[i],sample.bra[i])])
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[draw]
        mul!(AUX.Metro_1,L_set[i],@view(A[:,:,draw]))
        mul!(AUX.Metro_2,AUX.Metro_1,R_set[params.N+1-i])
        P=tr(AUX.Metro_2)
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            #sample.ket = copy(sample_p.ket)
            #sample.bra = copy(sample_p.bra)
            sample = projector(sample_p)
            acc+=1
        end
        mul!(R_set[params.N+2-i], @view(A[:,:,1+2*sample.ket[i]+sample.bra[i]]), R_set[params.N+1-i])
        mul!(AUX.C_mat, L_set[i], R_set[params.N+2-i])
        C = tr(AUX.C_mat)
    end
    return sample, R_set, acc
end

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

export Metropolis_burn_in

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

function Metropolis_burn_in(p::parameters, A::Array{ComplexF64,3})
    
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


export MPO_Metropolis_burn_in

function MPO_Metropolis_burn_in(A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, AUX::workspace)
    
    # Initialize random sample and calculate L_set for that sample:
    sample::projector = projector(rand(Bool, params.N),rand(Bool, params.N))
    L_set = L_MPO_strings(sample, A, params, AUX)
    
    # Perform burn_in:
    for _ in 1:params.burn_in
        sample, R_set = Mono_Metropolis_sweep_left(params,sample,A,L_set)
        sample, L_set = Mono_Metropolis_sweep_right(params,sample,A,R_set)
    end

    return sample, L_set
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




function reweighted_Mono_Metropolis_sweep_left(β, params::parameters, sample::density_matrix, A::Array{ComplexF64}, L_set::Vector{Matrix{ComplexF64}})

    function draw_excluded(u)
        v = rand(1:3)
        if v>=u
            v+=1
        end
        return v
    end

    acc=0

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
        metropolis_prob = real((P*conj(P)))^(β)/real((C*conj(C)))^(β)
        if rand() <= metropolis_prob
            #sample = sample_p
            sample.ket = deepcopy(sample_p.ket)
            sample.bra = deepcopy(sample_p.bra)
            acc+=1
        end

        #R = A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]*R
        R = A[:,:,1+2*sample.ket[i]+sample.bra[i]]*R
        push!(R_set, copy(R))
        C = tr(L_set[i]*R)
    
    #    C = tr(L_set[i]*R_set[N+2-i])
    end
    return sample, R_set::Vector{Matrix{ComplexF64}}, acc
end


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