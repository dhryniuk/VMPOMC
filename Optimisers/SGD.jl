export OLDcalculate_gradient, calculate_gradient, calculate_MC_gradient_full, SRMC_gradient_full, SR_calculate_gradient

function OLDcalculate_gradient(J,A)
    L∇L=zeros(ComplexF64,χ,χ,4)
    ΔLL=zeros(ComplexF64,χ,χ,4)
    Z=0

    #GRADIENT = zeros(ComplexF64,χ,χ,4)

    #mean_local_Lindbladian = local_Lindbladian(J,h,γ,A)
    #mean_local_Lindbladian = MC_local_Lindbladian(J,h,γ,A)
    mean_local_Lindbladian = 0

    #1-local part:
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) #replace by Monte Carlo
            L_set = L_MPO_strings(sample, A)
            R_set = R_MPO_strings(sample, A)
            ρ_sample = tr(L_set[N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z+=p_sample

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
                        local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[N+1-j]) #add if condition for loc=0
                        micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                        micro_sample.ket[j] = state[1]
                        micro_sample.bra[j] = state[2]
                        local_∇L+= loc*OLDderv_MPO(micro_sample,A)
                    end
                end

                #2-local part:
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,N)+1]-1)
                l_int += -1.0im*J*(l_int_α-l_int_β)
            end

            local_L /=ρ_sample
            local_∇L/=ρ_sample

            #Add in interaction terms:
            local_L +=l_int#*MPO(sample, A)
            local_∇L+=l_int*Δ_MPO(sample,A)

            L∇L+=p_sample*local_L*conj(local_∇L)

            #ΔLL:
            local_Δ=p_sample*conj(Δ_MPO(sample,A))
            ΔLL+=local_Δ

            #Mean local Lindbladian:
            mean_local_Lindbladian += p_sample*local_L*conj(local_L)
        end
    end
    #display(mean_local_Lindbladian/Z)
    #display(calculate_mean_local_Lindbladian(J,A))
    ΔLL*=mean_local_Lindbladian
    return (L∇L-ΔLL)/Z, mean_local_Lindbladian/Z
end

function calculate_gradient(J,A)
    L∇L=zeros(ComplexF64,χ,χ,4)
    ΔLL=zeros(ComplexF64,χ,χ,4)
    Z=0

    mean_local_Lindbladian = 0

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
                        
                        local_∇L+= loc*OLDderv_MPO(micro_sample,A)
                    end
                end

                #2-local part:
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,N)+1]-1)
                l_int += -1.0im*J*(l_int_α-l_int_β)
            end

            local_L /=ρ_sample
            local_∇L/=ρ_sample
    
            Δ_MPO_sample = OLDderv_MPO(sample,A)/ρ_sample
    
            #Add in interaction terms:
            local_L +=l_int
            local_∇L+=l_int*Δ_MPO_sample
    
            L∇L+=p_sample*local_L*conj(local_∇L)
    
            #ΔLL:
            local_Δ=p_sample*conj(Δ_MPO_sample)
            ΔLL+=local_Δ
    
            #Mean local Lindbladian:
            mean_local_Lindbladian += p_sample*local_L*conj(local_L)
        end
    end
    mean_local_Lindbladian/=Z
    ΔLL*=mean_local_Lindbladian
    return (L∇L-ΔLL)/Z, real(mean_local_Lindbladian)
end

function calculate_MC_gradient_partial(J,A,N_MC)
    #L∇L=zeros(ComplexF64,χ,χ,4)
    #ΔLL=zeros(ComplexF64,χ,χ,4)
    L∇L=Array{ComplexF64}(undef,χ,χ,4)
    ΔLL=Array{ComplexF64}(undef,χ,χ,4)

    mean_local_Lindbladian = 0

    sample = density_matrix(1,ones(N),ones(N))
    L_set = L_MPO_strings(sample, A)

    for k in 1:N_MC
        #for l in 1:10
        #    sample = Random_Metropolis(sample, A)
        #end

        sample, R_set = Mono_Metropolis_sweep(sample, A, L_set)
        ρ_sample = tr(R_set[N+1])
        L_set = Vector{Matrix{ComplexF64}}()
        L=Matrix{ComplexF64}(I, χ, χ)
        push!(L_set,copy(L))

        local_L=0
        local_∇L=zeros(ComplexF64,χ,χ,4)
        l_int = 0

        #L∇L*:
        for j in 1:N
            current_loc = 0.1
            current_micro_sample = sample
            r1,r2=draw2(4)

            #1-local part:
            s = dVEC[(sample.ket[j],sample.bra[j])]
            bra_L = transpose(s)*l1
            for i in 1:4
                loc = bra_L[i]
                if loc!=0
                    state = TPSC[i]
                    local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[N+1-j])
                    
                    metropolis_prob = real( (loc*conj(loc))/(current_loc*conj(current_loc)) )
                    if rand() <= metropolis_prob
                        current_micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                        current_micro_sample.ket[j] = state[1]
                        current_micro_sample.bra[j] = state[2]
                        current_loc = loc
                    end

                    #local_∇L+= loc*derv_MPO(micro_sample,A)
                end
                if i==r1 || i==r2 #|| i==r3
                    local_∇L+= current_loc*derv_MPO(current_micro_sample,A)
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

        Δ_MPO_sample = derv_MPO(sample,A)/ρ_sample

        #Add in interaction terms:
        local_L +=l_int#*MPO(sample, A)
        local_∇L+=l_int*Δ_MPO_sample

        L∇L+=local_L*conj(local_∇L)

        #ΔLL:
        local_Δ=conj(Δ_MPO_sample)
        ΔLL+=local_Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

    end
    ΔLL*=mean_local_Lindbladian/N_MC
    return (L∇L-ΔLL)/N_MC, mean_local_Lindbladian/N_MC
end

function calculate_MC_gradient_full(J,A,N_MC)
    L∇L=zeros(ComplexF64,χ,χ,4)
    ΔLL=zeros(ComplexF64,χ,χ,4)

    mean_local_Lindbladian = 0

    sample = density_matrix(1,ones(N),ones(N))
    L_set = L_MPO_strings(sample, A)

    for k in 1:N_MC
        #for l in 1:10
        #    sample = Random_Metropolis(sample, A)
        #end

        sample, R_set = Mono_Metropolis_sweep_left(sample, A, L_set)
        sample, L_set = Mono_Metropolis_sweep_right(sample, A, R_set)
        sample, R_set = Mono_Metropolis_sweep_left(sample, A, L_set)
        #sample, L_set = Mono_Metropolis_sweep_right(sample, A, R_set)
        #sample, R_set = Mono_Metropolis_sweep_left(sample, A, L_set)
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
                    local_∇L+= loc*derv_MPO(micro_sample,A,micro_L_set,micro_R_set)
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

        Δ_MPO_sample = derv_MPO(sample,A,L_set,R_set)/ρ_sample

        #Add in interaction terms:
        local_L +=l_int#*MPO(sample, A)
        local_∇L+=l_int*Δ_MPO_sample

        L∇L+=local_L*conj(local_∇L)

        #ΔLL:
        local_Δ=conj(Δ_MPO_sample)
        ΔLL+=local_Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

    end
    mean_local_Lindbladian/=N_MC
    #display(mean_local_Lindbladian)
    ΔLL*=mean_local_Lindbladian
    return (L∇L-ΔLL)/N_MC, real(mean_local_Lindbladian)
end
