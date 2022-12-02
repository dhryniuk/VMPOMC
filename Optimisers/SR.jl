function SR_calculate_gradient(J,A)
    L∇L=zeros(ComplexF64,χ,χ,4)
    ΔLL=zeros(ComplexF64,χ,χ,4)
    Z=0

    mean_local_Lindbladian = 0

    # Metric tensor auxiliary arrays:
    S = zeros(ComplexF64,4*χ^2,4*χ^2)
    G = zeros(ComplexF64,χ,χ,4)
    Left = zeros(ComplexF64,χ,χ,4)
    Right = zeros(ComplexF64,χ,χ,4)
    function flatten_index(i,j,s)
        return i+χ*(j-1)+χ^2*(s-1)
    end

    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l])
            L_set = L_MPO_strings(sample, A)
            R_set = R_MPO_strings(sample, A)
            ρ_sample = tr(L_set[N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z+=p_sample

            local_L=0
            local_∇L=zeros(ComplexF64,χ,χ,4)
            l_int = 0

            L_set = Vector{Matrix{ComplexF64}}()
            L=Matrix{ComplexF64}(I, χ, χ)
            push!(L_set,copy(L))

            #L∇L*:
            for j in 1:N

                #1-local part:
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*conj(l1)
                for i in 1:4
                    loc = bra_L[i]
                    if loc!=0
                        state = TPSC[i]
                        local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[N+1-j])
                        micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                        micro_sample.ket[j] = state[1]
                        micro_sample.bra[j] = state[2]
                        
                        micro_L_set = L_MPO_strings(micro_sample, A)
                        micro_R_set = R_MPO_strings(micro_sample, A)
                        local_∇L+= loc*derv_MPO(micro_sample,micro_L_set,micro_R_set)
                    end
                end

                #2-local part:
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,N)+1]-1)
                l_int += 1.0im*J*(l_int_α-l_int_β)

                #Update L_set:
                L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
                push!(L_set,copy(L))
            end

            local_L /=ρ_sample
            local_∇L/=ρ_sample
    
            Δ_MPO_sample = derv_MPO(sample,L_set,R_set)/ρ_sample
    
            #Add in interaction terms:
            local_L +=l_int
            local_∇L+=l_int*Δ_MPO_sample
    
            L∇L+=p_sample*local_L*conj(local_∇L)
    
            #ΔLL:
            local_Δ=p_sample*conj(Δ_MPO_sample)
            ΔLL+=local_Δ
    
            #Mean local Lindbladian:
            mean_local_Lindbladian += p_sample*local_L*conj(local_L)

            #Metric tensor:
            G = Δ_MPO_sample
            Left += p_sample*conj(G)
            Right+= p_sample*G
            for s in 1:4, j in 1:χ, i in 1:χ, ss in 1:4, jj in 1:χ, ii in 1:χ
                S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] += p_sample*conj(G[i,j,s])*G[ii,jj,ss]
            end
        end
    end
    mean_local_Lindbladian/=Z
    ΔLL*=mean_local_Lindbladian

    #Metric tensor:
    S./=Z
    Left./=Z
    Right./=Z
    for s in 1:4, j in 1:χ, i in 1:χ, ss in 1:4, jj in 1:χ, ii in 1:χ
        S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    S+=+0.01*Matrix{Int}(I, χ*χ*4, χ*χ*4)

    grad = (L∇L-ΔLL)/Z
    flat_grad = reshape(grad,4*χ^2)
    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad,χ,χ,4)

    return grad, real(mean_local_Lindbladian)
end


function SRMC_gradient_full(J,A,N_MC,N_sweeps,step)
    L∇L=zeros(ComplexF64,χ,χ,4)
    ΔLL=zeros(ComplexF64,χ,χ,4)

    mean_local_Lindbladian = 0

    sample = density_matrix(1,ones(N),ones(N))
    L_set = L_MPO_strings(sample, A)

    # Metric tensor auxiliary arrays:
    S = zeros(ComplexF64,4*χ^2,4*χ^2)
    G = zeros(ComplexF64,χ,χ,4)
    Left = zeros(ComplexF64,χ,χ,4)
    Right = zeros(ComplexF64,χ,χ,4)
    function flatten_index(i,j,s)
        return i+χ*(j-1)+χ^2*(s-1)
    end

    for k in 1:N_MC

        sample, R_set = Mono_Metropolis_sweep_left(sample, A, L_set)
        for n in N_sweeps
            sample, L_set = Mono_Metropolis_sweep_right(sample, A, R_set)
            sample, R_set = Mono_Metropolis_sweep_left(sample, A, L_set)
        end
        ρ_sample = tr(R_set[N+1])
        L_set = Vector{Matrix{ComplexF64}}()
        L=Matrix{ComplexF64}(I, χ, χ)
        push!(L_set,copy(L))

        local_L=0
        local_∇L=zeros(ComplexF64,χ,χ,4)
        l_int = 0

        #L∇L*:
        for j in 1:N

            #1-local part:
            s = dVEC[(sample.ket[j],sample.bra[j])]
            bra_L = transpose(s)*l1
            for i in 1:4
                loc = bra_L[i]
                if loc!=0
                    state = TPSC[i]
                    local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[N+1-j])
                    
                    micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                    micro_sample.ket[j] = state[1]
                    micro_sample.bra[j] = state[2]

                    micro_L_set = L_MPO_strings(micro_sample, A)
                    micro_R_set = R_MPO_strings(micro_sample, A)
                    local_∇L+= loc*derv_MPO(micro_sample,micro_L_set,micro_R_set)
                end
            end

            #2-local part:
            l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,N)+1]-1)
            l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,N)+1]-1)
            l_int += -1.0im*J*(l_int_α-l_int_β)

            #Update L_set:
            L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
            push!(L_set,copy(L))
        end

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        Δ_MPO_sample = derv_MPO(sample,L_set,R_set)/ρ_sample

        #Add in interaction terms:
        local_L +=l_int
        local_∇L+=l_int*Δ_MPO_sample

        L∇L+=local_L*conj(local_∇L)

        #ΔLL:
        local_Δ=conj(Δ_MPO_sample)
        ΔLL+=local_Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

        #Metric tensor:
        G = Δ_MPO_sample
        Left += G #change order of conjugation, but it shouldn't matter
        Right+= conj(G)
        for s in 1:4, j in 1:χ, i in 1:χ, ss in 1:4, jj in 1:χ, ii in 1:χ
            S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] += conj(G[i,j,s])*G[ii,jj,ss]
        end
    end
    mean_local_Lindbladian/=N_MC
    ΔLL*=mean_local_Lindbladian

    #Metric tensor:
    S./=N_MC
    Left./=N_MC
    Right./=N_MC
    for s in 1:4, j in 1:χ, i in 1:χ, ss in 1:4, jj in 1:χ, ii in 1:χ
        S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    #S+=max(0.001,1*0.95^step)*Matrix{Int}(I, χ*χ*4, χ*χ*4)
    S+=0.1*Matrix{Int}(I, χ*χ*4, χ*χ*4)

    grad = (L∇L-ΔLL)/N_MC
    flat_grad = reshape(grad,4*χ^2)
    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad,χ,χ,4)

    return grad, real(mean_local_Lindbladian)
end
