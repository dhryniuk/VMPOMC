using LinearAlgebra

#Sweep lattice from right to left:
function Dual_Metropolis_sweep(sample, A, L_set)
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

function Mono_Metropolis_sweep(sample, A, L_set)

    function draw_excluded(u)
        v = rand(1:3)
        if v>=u
            v+=1
        end
        return v
    end


    R_set = []
    R = Matrix{ComplexF64}(I, χ, χ)
    push!(R_set, copy(R))
    C = tr(L_set[N+1]) #Current MPO  ---> move into loop
    for i in N:-1:1

        sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra)) #deepcopy necessary?
        u = dINDEX[(sample.ket[i],sample.bra[i])]
        v = draw_excluded(u)
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[v]
        P = tr(L_set[i]*A[:,:,v]*R_set[N+1-i])
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = sample_p
        end

        R*= A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        push!(R_set, copy(R))
        C = tr(L_set[i]*R)

    end
    return sample, R_set
end

function Random_Metropolis(sample, A)

    function draw_excluded(u)
        v = rand(1:3)
        if v>=u
            v+=1
        end
        return v
    end

    C = MPO(sample, A)
    i=rand(1:N)

    sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra)) #deepcopy necessary?
    u = dINDEX[(sample.ket[i],sample.bra[i])]
    v = draw_excluded(u)
    (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[v]
    P = MPO(sample_p, A)
    metropolis_prob = real((P*conj(P))/(C*conj(C)))
    if rand() <= metropolis_prob
        sample = sample_p
    end

    return sample
end



function check_MC_dist(N_MC)
    #B = zeros(dim,dim)
    B = zeros(4,4,4)
    sample = density_matrix(1,basis[1],basis[1])
    L_set = L_MPO_strings(sample,A)
    for i in 1:N_MC
        sample, R_set = Mono_Metropolis_sweep(sample, A, L_set)
        #sample = Random_Metropolis_sweep(sample, A)
        #display(sample)
        L_set = L_MPO_strings(sample,A)
        a = dINDEX[(sample.ket[1],sample.bra[1])]
        b = dINDEX[(sample.ket[2],sample.bra[2])]
        c = dINDEX[(sample.ket[3],sample.bra[3])]
        B[a,b,c]+=1
    end
    display(B./N_MC)
end

function check_MPO_dist()
    #B = zeros(dim,dim)
    B = zeros(4,4,4)
    Z=0
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) 
            ρ_sample = MPO(sample,A)
            p_sample = ρ_sample*conj(ρ_sample)
            Z+=p_sample
            #B[k,l] = p_sample
            a = dINDEX[(sample.ket[1],sample.bra[1])]
            b = dINDEX[(sample.ket[2],sample.bra[2])]
            c = dINDEX[(sample.ket[3],sample.bra[3])]
            B[a,b,c]=p_sample
        end
    end

    display(real(B./Z))
end


#check_MC_dist(1000000)
#check_MPO_dist()
#error()


function calculate_mean_local_Lindbladian(J,A)
    mll=0
    Z=0
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) 
            ρ_sample = MPO(sample,A)
            p_sample = ρ_sample*conj(ρ_sample)
            Z+=p_sample

            L_set = L_MPO_strings(sample,A)
            R_set = R_MPO_strings(sample,A)

            local_L = local_Lindbladian(J,A,sample,L_set,R_set)

            mll+=p_sample*local_L*conj(local_L)
        end
    end

    return mll/Z
end

function MC_mean_local_Lindbladian(J,A,N_MC)
    mll=0
    sample = density_matrix(1,basis[1],basis[1]) 
    L_set = L_MPO_strings(sample,A)
    for k in 1:N_MC
        #sample, R_set = Mono_Metropolis_sweep(sample, A, L_set)
        sample = Random_Metropolis(sample, A)
        L_set = L_MPO_strings(sample,A)
        R_set = R_MPO_strings(sample,A)

        local_L = local_Lindbladian(J,A,sample,L_set,R_set)

        mll+=local_L*conj(local_L)
    end

    return mll/N_MC
end


#ex = calculate_mean_local_Lindbladian(J,A)
#mc = MC_mean_local_Lindbladian(J,A,100000)

#display(ex)
#display(mc)

#check_MC_dist(100000)
#check_MPO_dist()
#error()
