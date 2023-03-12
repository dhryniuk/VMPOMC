export SR_calculate_gradient, SR_calculate_MC_gradient_full, SR_LdagL_gradient, MT_SR_MC_grad, multi_threaded_SR_calculate_MC_gradient_full

#experimental:
export distributed_SR_calculate_MC_gradient_full, MC_SR_calculate_MPS_gradient, distributed_SR_calculate_MC_MPS_gradient

function SR_calculate_gradient(p::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64},ϵ,basis)
    #χ=size(A[:,:,1])[1]

    L∇L=zeros(ComplexF64,p.χ,p.χ,4)
    ΔLL=zeros(ComplexF64,p.χ,p.χ,4)
    Z=0

    mean_local_Lindbladian = 0

    # Metric tensor auxiliary arrays:
    S = zeros(ComplexF64,4*p.χ^2,4*p.χ^2)
    G = zeros(ComplexF64,p.χ,p.χ,4)
    Left = zeros(ComplexF64,p.χ,p.χ,4)
    Right = zeros(ComplexF64,p.χ,p.χ,4)
    function flatten_index(i,j,s)
        return i+p.χ*(j-1)+p.χ^2*(s-1)
    end

    for k in 1:p.dim
        for l in 1:p.dim
            sample = density_matrix(1,basis[k],basis[l])
            L_set = L_MPO_strings(p, sample, A)
            R_set = R_MPO_strings(p, sample, A)
            ρ_sample = tr(L_set[p.N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z+=p_sample

            local_L=0
            local_∇L=zeros(ComplexF64,p.χ,p.χ,4)
            l_int = 0

            L_set = Vector{Matrix{ComplexF64}}()
            L=Matrix{ComplexF64}(I, p.χ, p.χ)
            push!(L_set,copy(L))

            #L∇L*:
            for j in 1:p.N

                #1-local part:
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*conj(l1)
                for i in 1:4
                    loc = bra_L[i]
                    if loc!=0
                        state = TPSC[i]
                        local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[p.N+1-j])
                        micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                        micro_sample.ket[j] = state[1]
                        micro_sample.bra[j] = state[2]
                        
                        micro_L_set = L_MPO_strings(p,micro_sample, A)
                        micro_R_set = R_MPO_strings(p, micro_sample, A)
                        local_∇L+= loc*derv_MPO(p, micro_sample,micro_L_set,micro_R_set)
                    end
                end

                #2-local part:
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,p.N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,p.N)+1]-1)
                l_int += 1.0im*p.J*(l_int_α-l_int_β)

                #Update L_set:
                L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
                push!(L_set,copy(L))
            end

            local_L /=ρ_sample
            local_∇L/=ρ_sample
    
            Δ_MPO_sample = derv_MPO(p, sample, L_set, R_set)/ρ_sample
    
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
            for s in 1:4, j in 1:p.χ, i in 1:p.χ, ss in 1:4, jj in 1:p.χ, ii in 1:p.χ
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
    for s in 1:4, j in 1:p.χ, i in 1:p.χ, ss in 1:4, jj in 1:p.χ, ii in 1:p.χ
        S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    S+=ϵ*Matrix{Int}(I, p.χ*p.χ*4, p.χ*p.χ*4)

    grad = (L∇L-ΔLL)/Z
    flat_grad = reshape(grad,4*p.χ^2)
    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad,p.χ,p.χ,4)

    return grad, real(mean_local_Lindbladian)
end


function SR_calculate_MC_gradient_full(p::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, N_MC::Int64, N_sweeps::Int64, ϵ::Float64)
    L∇L=zeros(ComplexF64,p.χ,p.χ,4)
    ΔLL=zeros(ComplexF64,p.χ,p.χ,4)

    mean_local_Lindbladian = 0

    sample = density_matrix(1,rand(0:1,p.N),rand(0:1,p.N))
    L_set = L_MPO_strings(p, sample, A)

    # Metric tensor auxiliary arrays:
    S = zeros(ComplexF64,4*p.χ^2,4*p.χ^2)
    G = zeros(ComplexF64,p.χ,p.χ,4)
    Left = zeros(ComplexF64,p.χ,p.χ,4)
    Right = zeros(ComplexF64,p.χ,p.χ,4)
    function flatten_index(i,j,s)
        return i+p.χ*(j-1)+p.χ^2*(s-1)
    end

    for k in 1:N_MC

        sample, R_set = Mono_Metropolis_sweep_left(p, sample, A, L_set)
        for n in N_sweeps
            sample, L_set = Mono_Metropolis_sweep_right(p, sample, A, R_set)
            sample, R_set = Mono_Metropolis_sweep_left(p, sample, A, L_set)
        end
        ρ_sample = tr(R_set[p.N+1])
        L_set = Vector{Matrix{ComplexF64}}()
        L=Matrix{ComplexF64}(I, p.χ, p.χ)
        push!(L_set,copy(L))

        local_L=0
        local_∇L=zeros(ComplexF64,p.χ,p.χ,4)
        l_int = 0

        #L∇L*:
        for j in 1:p.N

            #1-local part:
            s = dVEC[(sample.ket[j],sample.bra[j])]
            bra_L = transpose(s)*conj(l1)
            for i in 1:4
                loc = bra_L[i]
                if loc!=0
                    state = TPSC[i]
                    local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[p.N+1-j])
                    
                    micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                    micro_sample.ket[j] = state[1]
                    micro_sample.bra[j] = state[2]

                    micro_L_set = L_MPO_strings(p, micro_sample, A)
                    micro_R_set = R_MPO_strings(p, micro_sample, A)
                    local_∇L+= loc*derv_MPO(p, micro_sample, micro_L_set, micro_R_set)
                end
            end

            #2-local part:
            l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,p.N)+1]-1)
            l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,p.N)+1]-1)
            l_int += 1.0im*p.J*(l_int_α-l_int_β)

            #Update L_set:
            L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
            push!(L_set,copy(L))
        end

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        Δ_MPO_sample = derv_MPO(p, sample, L_set, R_set)/ρ_sample

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
        for s in 1:4, j in 1:p.χ, i in 1:p.χ, ss in 1:4, jj in 1:p.χ, ii in 1:p.χ
            S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] += conj(G[i,j,s])*G[ii,jj,ss]
        end
    end
    mean_local_Lindbladian/=N_MC
    ΔLL*=mean_local_Lindbladian

    #Metric tensor:
    S./=N_MC
    Left./=N_MC
    Right./=N_MC
    for s in 1:4, j in 1:p.χ, i in 1:p.χ, ss in 1:4, jj in 1:p.χ, ii in 1:p.χ
        S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    #S+=max(0.001,1*0.95^step)*Matrix{Int}(I, χ*χ*4, χ*χ*4)
    S+=ϵ*Matrix{Int}(I, p.χ*p.χ*4, p.χ*p.χ*4)

    grad = (L∇L-ΔLL)/N_MC
    flat_grad = reshape(grad,4*p.χ^2)
    """
    SHOULD S REALLY BE CONJUGATED?
    """
    flat_grad = inv(S)*flat_grad
    #flat_grad = inv(conj.(S))*flat_grad
    grad = reshape(flat_grad,p.χ,p.χ,4)

    return grad, real(mean_local_Lindbladian)
end


function sample_with_SR(p::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, N_MC::Int64, N_sweeps::Int64)
    function flatten_index(i,j,s)
        return i+p.χ*(j-1)+p.χ^2*(s-1)
    end


    #Initialize variables:
    L∇L=zeros(ComplexF64,p.χ,p.χ,4)
    ΔLL=zeros(ComplexF64,p.χ,p.χ,4)
    mean_local_Lindbladian=0.0+0.0im
    S=zeros(ComplexF64,4*p.χ^2,4*p.χ^2)
    Left=zeros(ComplexF64,p.χ,p.χ,4)
    Right=zeros(ComplexF64,p.χ,p.χ,4)


    sample = density_matrix(1,rand(0:1,p.N),rand(0:1,p.N))
    L_set = L_MPO_strings(p, sample, A)
    for k in 1:N_MC

        sample, R_set = Mono_Metropolis_sweep_left(p, sample, A, L_set)
        for n in N_sweeps
            sample, L_set = Mono_Metropolis_sweep_right(p, sample, A, R_set)
            sample, R_set = Mono_Metropolis_sweep_left(p, sample, A, L_set)
        end
        ρ_sample = tr(R_set[p.N+1])
        L_set = Vector{Matrix{ComplexF64}}()
        L=Matrix{ComplexF64}(I, p.χ, p.χ)
        push!(L_set,copy(L))

        local_L=0
        local_∇L=zeros(ComplexF64,p.χ,p.χ,4)
        l_int = 0

        #L∇L*:
        for j in 1:p.N

            #1-local part:
            s = dVEC[(sample.ket[j],sample.bra[j])]
            bra_L = transpose(s)*conj(l1)
            for i in 1:4
                loc = bra_L[i]
                if loc!=0
                    state = TPSC[i]
                    local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[p.N+1-j])
                    
                    micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                    micro_sample.ket[j] = state[1]
                    micro_sample.bra[j] = state[2]

                    micro_L_set = L_MPO_strings(p, micro_sample, A)
                    micro_R_set = R_MPO_strings(p, micro_sample, A)
                    local_∇L+= loc*derv_MPO(p, micro_sample, micro_L_set, micro_R_set)
                end
            end

            #2-local part:
            l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,p.N)+1]-1)
            l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,p.N)+1]-1)
            l_int += 1.0im*p.J*(l_int_α-l_int_β)

            #Update L_set:
            L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
            push!(L_set,copy(L))
        end

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        Δ_MPO_sample = derv_MPO(p, sample, L_set, R_set)/ρ_sample

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
        for s in 1:4, j in 1:p.χ, i in 1:p.χ, ss in 1:4, jj in 1:p.χ, ii in 1:p.χ
            S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] += conj(G[i,j,s])*G[ii,jj,ss]
        end
    end
    return [L∇L, ΔLL, mean_local_Lindbladian, S, Left, Right]
end


function sum_long_range_interactions(p::parameters, sample)
    l_int_ket = 0.0
    l_int_bra = 0.0
    l_int = 0.0
    for i in 1:p.N-1
        for j in i+1:p.N
            l_int_ket = (2*sample.ket[i]-1)*(2*sample.ket[j]-1)
            l_int_bra = (2*sample.bra[i]-1)*(2*sample.bra[j]-1)
            dist = min(abs(i-j), abs(p.N+i-j))^p.α
            l_int += (l_int_ket-l_int_bra)/dist
        end
    end
    return 1.0im*p.J*l_int
end



function sample_with_SR_long_range(p::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, N_MC::Int64, N_sweeps::Int64)
    function flatten_index(i,j,s)
        return i+p.χ*(j-1)+p.χ^2*(s-1)
    end


    #Initialize variables:
    L∇L=zeros(ComplexF64,p.χ,p.χ,4)
    ΔLL=zeros(ComplexF64,p.χ,p.χ,4)
    mean_local_Lindbladian=0.0+0.0im
    S=zeros(ComplexF64,4*p.χ^2,4*p.χ^2)
    Left=zeros(ComplexF64,p.χ,p.χ,4)
    Right=zeros(ComplexF64,p.χ,p.χ,4)


    sample = density_matrix(1,rand(0:1,p.N),rand(0:1,p.N))
    L_set = L_MPO_strings(p, sample, A)
    for k in 1:N_MC

        sample, R_set = Mono_Metropolis_sweep_left(p, sample, A, L_set)
        for n in N_sweeps
            sample, L_set = Mono_Metropolis_sweep_right(p, sample, A, R_set)
            sample, R_set = Mono_Metropolis_sweep_left(p, sample, A, L_set)
        end
        ρ_sample = tr(R_set[p.N+1])
        L_set = Vector{Matrix{ComplexF64}}()
        L=Matrix{ComplexF64}(I, p.χ, p.χ)
        push!(L_set,copy(L))

        local_L=0
        local_∇L=zeros(ComplexF64,p.χ,p.χ,4)

        #L∇L*:
        for j in 1:p.N

            #1-local part:
            s = dVEC[(sample.ket[j],sample.bra[j])]
            bra_L = transpose(s)*conj(l1)
            for i in 1:4
                loc = bra_L[i]
                if loc!=0
                    state = TPSC[i]
                    local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[p.N+1-j])
                    
                    micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                    micro_sample.ket[j] = state[1]
                    micro_sample.bra[j] = state[2]

                    micro_L_set = L_MPO_strings(p, micro_sample, A)
                    micro_R_set = R_MPO_strings(p, micro_sample, A)
                    local_∇L+= loc*derv_MPO(p, micro_sample, micro_L_set, micro_R_set)
                end
            end

            #Update L_set:
            L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
            push!(L_set,copy(L))
        end

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        Δ_MPO_sample = derv_MPO(p, sample, L_set, R_set)/ρ_sample

        #Add in interaction terms:
        l_int = sum_long_range_interactions(p, sample)
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
        for s in 1:4, j in 1:p.χ, i in 1:p.χ, ss in 1:4, jj in 1:p.χ, ii in 1:p.χ
            S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] += conj(G[i,j,s])*G[ii,jj,ss]
        end
    end
    return [L∇L, ΔLL, mean_local_Lindbladian, S, Left, Right]
end


function distributed_SR_calculate_MC_gradient_full(p::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, N_MC::Int64, N_sweeps::Int64, ϵ::Float64)
    #output = [L∇L, ΔLL, mean_local_Lindbladian, S, Left, Right]

    #perform reduction:
    output = @distributed (+) for i=1:nworkers()
        #sample_with_SR_long_range(p, A, l1, N_MC, N_sweeps)
        sample_with_SR(p, A, l1, N_MC, N_sweeps)
    end

    L∇L=output[1]
    ΔLL=output[2]
    mean_local_Lindbladian=output[3]
    S=output[4]
    Left=output[5]
    Right=output[6]

    mean_local_Lindbladian/=(N_MC*nworkers())
    ΔLL*=mean_local_Lindbladian

    #Metric tensor:
    S/=(N_MC*nworkers())
    Left/=(N_MC*nworkers())
    Right/=(N_MC*nworkers())

    function flatten_index(i,j,s)
        return i+p.χ*(j-1)+p.χ^2*(s-1)
    end

    for s in 1:4, j in 1:p.χ, i in 1:p.χ, ss in 1:4, jj in 1:p.χ, ii in 1:p.χ
        S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    #S+=max(0.001,1*0.95^step)*Matrix{Int}(I, χ*χ*4, χ*χ*4)
    S+=ϵ*Matrix{Int}(I, p.χ*p.χ*4, p.χ*p.χ*4)

    grad = (L∇L-ΔLL)/(N_MC*nworkers())
    flat_grad = reshape(grad,4*p.χ^2)
    #flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad,p.χ,p.χ,4)

    return grad, real(mean_local_Lindbladian)
end

function multi_threaded_SR_calculate_MC_gradient_full(p::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, N_MC::Int64, N_sweeps::Int64, ϵ::Float64)
    tL∇L = [zeros(ComplexF64,p.χ,p.χ,4) for _ in 1:Threads.nthreads()]
    tΔLL = [zeros(ComplexF64,p.χ,p.χ,4) for _ in 1:Threads.nthreads()]

    tmean_local_Lindbladian = [0.0+0.0im for _ in 1:Threads.nthreads()]

    # Metric tensor auxiliary arrays for each individual thread:
    tS = [zeros(ComplexF64,4*p.χ^2,4*p.χ^2) for _ in 1:Threads.nthreads()]
    #G = [zeros(ComplexF64,p.χ,p.χ,4) for _ in 1:2*Threads.nthreads()]
    tLeft  = [zeros(ComplexF64,p.χ,p.χ,4) for _ in 1:Threads.nthreads()]
    tRight = [zeros(ComplexF64,p.χ,p.χ,4) for _ in 1:Threads.nthreads()]
    function flatten_index(i,j,s)
        return i+p.χ*(j-1)+p.χ^2*(s-1)
    end

    Threads.@threads for t in 1:(Threads.nthreads())
        sample = density_matrix(1,rand(0:1,p.N),rand(0:1,p.N))
        L_set = L_MPO_strings(p, sample, A)
        for k in 1:N_MC

            sample, R_set = Mono_Metropolis_sweep_left(p, sample, A, L_set)
            for n in N_sweeps
                sample, L_set = Mono_Metropolis_sweep_right(p, sample, A, R_set)
                sample, R_set = Mono_Metropolis_sweep_left(p, sample, A, L_set)
            end
            ρ_sample = tr(R_set[p.N+1])
            L_set = Vector{Matrix{ComplexF64}}()
            L=Matrix{ComplexF64}(I, p.χ, p.χ)
            push!(L_set,copy(L))

            local_L=0
            local_∇L=zeros(ComplexF64,p.χ,p.χ,4)
            l_int = 0

            #L∇L*:
            for j in 1:p.N

                #1-local part:
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*conj(l1)
                for i in 1:4
                    loc = bra_L[i]
                    if loc!=0
                        state = TPSC[i]
                        local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[p.N+1-j])
                        
                        micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                        micro_sample.ket[j] = state[1]
                        micro_sample.bra[j] = state[2]

                        micro_L_set = L_MPO_strings(p, micro_sample, A)
                        micro_R_set = R_MPO_strings(p, micro_sample, A)
                        local_∇L+= loc*derv_MPO(p, micro_sample, micro_L_set, micro_R_set)
                    end
                end

                #2-local part:
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,p.N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,p.N)+1]-1)
                l_int += 1.0im*p.J*(l_int_α-l_int_β)

                #Update L_set:
                L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
                push!(L_set,copy(L))
            end

            local_L /=ρ_sample
            local_∇L/=ρ_sample

            Δ_MPO_sample = derv_MPO(p, sample, L_set, R_set)/ρ_sample

            #Add in interaction terms:
            local_L +=l_int
            local_∇L+=l_int*Δ_MPO_sample

            tL∇L[t]+=local_L*conj(local_∇L)

            #ΔLL:
            local_Δ=conj(Δ_MPO_sample)
            tΔLL[t]+=local_Δ

            #Mean local Lindbladian:
            tmean_local_Lindbladian[t] += local_L*conj(local_L)

            #Metric tensor:
            G = Δ_MPO_sample
            tLeft[t] += G #change order of conjugation, but it shouldn't matter
            tRight[t]+= conj(G)
            for s in 1:4, j in 1:p.χ, i in 1:p.χ, ss in 1:4, jj in 1:p.χ, ii in 1:p.χ
                tS[t][flatten_index(i,j,s),flatten_index(ii,jj,ss)] += conj(G[i,j,s])*G[ii,jj,ss]
            end
        end
    end
    #mean_local_Lindbladian/=N_MC
    #ΔLL*=mean_local_Lindbladian
    mean_local_Lindbladian = sum(tmean_local_Lindbladian)/(Threads.nthreads()*N_MC)
    ΔLL = sum(tΔLL)/(Threads.nthreads())
    ΔLL*=mean_local_Lindbladian
    L∇L = sum(tL∇L)/(Threads.nthreads())

    #Metric tensor:
    S = sum(tS)/(Threads.nthreads()*N_MC)
    Left = sum(tLeft)/(Threads.nthreads()*N_MC)
    Right= sum(tRight)/(Threads.nthreads()*N_MC)

    #S./=N_MC
    #Left./=N_MC
    #Right./=N_MC
    for s in 1:4, j in 1:p.χ, i in 1:p.χ, ss in 1:4, jj in 1:p.χ, ii in 1:p.χ
        S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    #S+=max(0.001,1*0.95^step)*Matrix{Int}(I, χ*χ*4, χ*χ*4)
    S+=ϵ*Matrix{Int}(I, p.χ*p.χ*4, p.χ*p.χ*4)

    grad = (L∇L-ΔLL)/N_MC
    flat_grad = reshape(grad,4*p.χ^2)
    flat_grad = inv(conj.(S))*flat_grad
    grad = reshape(flat_grad,p.χ,p.χ,4)

    return grad, real(mean_local_Lindbladian)
end

function SR_LdagL_gradient(p,A,l1,ϵ,basis)
    ΔLL=zeros(ComplexF64,p.χ,p.χ,4)
    Z=0

    mean_local_Lindbladian = 0
    mean_Δ = zeros(ComplexF64,p.χ,p.χ,4)

    # Metric tensor auxiliary arrays:
    S = zeros(ComplexF64,4*p.χ^2,4*p.χ^2)
    G = zeros(ComplexF64,p.χ,p.χ,4)
    Left = zeros(ComplexF64,p.χ,p.χ,4)
    Right = zeros(ComplexF64,p.χ,p.χ,4)
    function flatten_index(i,j,s)
        return i+p.χ*(j-1)+p.χ^2*(s-1)
    end    

    for k in 1:p.dim
        for l in 1:p.dim
            sample = density_matrix(1,basis[k],basis[l])
            L_set = L_MPO_strings(p, sample, A)
            R_set = R_MPO_strings(p, sample, A)
            ρ_sample = tr(L_set[p.N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z+=p_sample

            local_L=0
            l_int = 0

            L_set = Vector{Matrix{ComplexF64}}()
            L=Matrix{ComplexF64}(I, p.χ, p.χ)
            push!(L_set,copy(L))

            #L∇L*:
            for j in 1:p.N

                #1-local part:
                s = dVEC[(sample.ket[j],sample.bra[j])]
                #bra_L = transpose(s)*l1
                bra_L = transpose(s)*conj(transpose(l1))
                for i in 1:4
                    loc = bra_L[i]
                    if loc!=0
                        state = TPSC[i]
                        y_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                        y_sample.ket[j] = state[1]
                        y_sample.bra[j] = state[2]

                        for yj in 1:p.N
                            y_s = dVEC[(y_sample.ket[yj],y_sample.bra[yj])]
                            y_bra_L = transpose(y_s)*l1
                            for yi in 1:4
                                y_loc = y_bra_L[yi]
                                if y_loc!=0
                                    y_state = TPSC[yi]
                                    z_sample = density_matrix(1,deepcopy(y_sample.ket),deepcopy(y_sample.bra))
                                    z_sample.ket[yj] = y_state[1]
                                    z_sample.bra[yj] = y_state[2]
                                    local_L += loc*y_loc*MPO(p,z_sample,A)
                                end
                            end
                        end
                    end
                end

                #2-local part:
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,p.N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,p.N)+1]-1)
                #l_int += -1.0im*J*(l_int_α-l_int_β)
                l_int += 1.0im*p.J*(l_int_α-l_int_β)
                l_int*=conj(l_int)

                #Update L_set:
                L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
                push!(L_set,copy(L))
            end

            local_L /= ρ_sample
            #Add in interaction terms:
            local_L += l_int    #/=ρ_sample?
    
            Δ_MPO_sample = derv_MPO(p, sample, L_set, R_set)/ρ_sample
    
            #ΔLL:
            mean_Δ+=p_sample*conj(Δ_MPO_sample)
            ΔLL+=p_sample*conj(Δ_MPO_sample)*local_L
    
            #Mean local Lindbladian:
            mean_local_Lindbladian += p_sample*local_L

            #Metric tensor:
            G = Δ_MPO_sample
            Left += p_sample*conj(G)
            Right+= p_sample*G
            for s in 1:4, j in 1:p.χ, i in 1:p.χ, ss in 1:4, jj in 1:p.χ, ii in 1:p.χ
                S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] += p_sample*conj(G[i,j,s])*G[ii,jj,ss]
            end
        end
    end
    mean_local_Lindbladian/=Z
    mean_Δ*=mean_local_Lindbladian

    #Metric tensor:
    S./=Z
    Left./=Z
    Right./=Z
    for s in 1:4, j in 1:p.χ, i in 1:p.χ, ss in 1:4, jj in 1:p.χ, ii in 1:p.χ
        S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    S+=ϵ*Matrix{Int}(I, p.χ*p.χ*4, p.χ*p.χ*4)

    grad = (ΔLL-mean_Δ)/Z
    flat_grad = reshape(grad,4*p.χ^2)
    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad,p.χ,p.χ,4)

    return grad, real(mean_local_Lindbladian)
end








function MC_SR_calculate_MPS_gradient(p::parameters, A::Array{Float64}, N_MC::Int64, ϵ::Float64)

    # Initialize products:
    L∇L::Array{Float64,3} = zeros(Float64,p.χ,p.χ,2) #coupled product
    ΔLL::Array{Float64,3} = zeros(Float64,p.χ,p.χ,2) #uncoupled product

    # Initialize metric tensor auxiliary arrays:
    S = zeros(Float64, 2*p.χ^2, 2*p.χ^2)
    G = zeros(Float64, p.χ, p.χ, 2)
    Left = zeros(Float64, p.χ, p.χ, 2)
    Right = zeros(Float64, p.χ, p.χ, 2)

    mean_local_Hamiltonian::Float64 = 0

    # Define 1-local Hamiltonian:
    h1::Matrix{ComplexF64} = p.h*sx

    # Initialize sample and L_set for that sample:
    sample, L_set = Metropolis_burn_in(p, A)

    acceptance::UInt64=0

    for _ in 1:N_MC
        sample, R_set, acc = Mono_Metropolis_sweep_left(p, sample, A, L_set)
        acceptance+=acc
        ρ_sample = tr(R_set[p.N+1])

        # Prepare new L_set of left MPS strings:
        L_set = [ Matrix{Float64}(undef, p.χ, p.χ) for _ in 1:p.N+1 ]
        L = Matrix{Float64}(I, p.χ, p.χ)
        L_set[1] = L

        e_field::Float64 = 0
        e_int::Float64 = 0

        #L∇L*:
        for j::UInt16 in 1:p.N

            #1-local part (field):
            bra_L::Transpose{ComplexF64, Vector{ComplexF64}} = transpose(dVEC2[sample[j]])*h1
            for i::UInt8 in 1:2
                loc::ComplexF64 = bra_L[i]
                if loc!=0
                    state::Bool = TPSC2[i]
                    @inbounds e_field -= loc*tr(L_set[j]*A[:,:,2-state]*R_set[p.N+1-j])
                end
            end

            #Interaction term:
            @inbounds e_int -= p.J * (2*sample[j]-1) * (2*sample[mod(j,p.N)+1]-1)

            #Update L_set:
            L*=A[:,:,2-sample[j]]
            L_set[j+1] = L
        end

        e_field/=ρ_sample
        # e_int/=ρ_sample ???

        Δ_MPO_sample = derv_MPS(p, sample, L_set, R_set)/ρ_sample

        #Add energies:
        local_E::Float64 = real(e_int+e_field)
        local_∇E::Array{Float64,3} = real(e_int+e_field)*Δ_MPO_sample
        L∇L += local_∇E

        #ΔLL:
        local_Δ = Δ_MPO_sample
        ΔLL += local_Δ

        #Mean local Lindbladian:
        mean_local_Hamiltonian += real(local_E)

        #Metric tensor:
        G = Δ_MPO_sample
        Left += G #change order of conjugation, but it shouldn't matter
        Right+= conj(G)
        #@inbounds for (s, j, i, ss, jj, ii) in zip(1:2, 1:p.χ, 1:p.χ, 1:2, 1:p.χ, 1:p.χ)
        for s in 1:2, j in 1:p.χ, i in 1:p.χ, ss in 1:2, jj in 1:p.χ, ii in 1:p.χ
        #for (s,j,i,ss,jj,ii) in Iterators.product(1:2, 1:p.χ, 1:p.χ, 1:2, 1:p.χ, 1:p.χ)
            #println(s, " ", j, " ", i, " ", ss, " ", jj, " ", ii)
            @inbounds S[flatten_index(i,j,s,p),flatten_index(ii,jj,ss,p)] += conj(G[i,j,s])*G[ii,jj,ss]
        end
    end
    mean_local_Hamiltonian/=N_MC
    ΔLL*=mean_local_Hamiltonian

    #Metric tensor:
    S./=N_MC
    Left./=N_MC
    Right./=N_MC
    #@inbounds for (s, j, i, ss, jj, ii) in zip(1:2, 1:p.χ, 1:p.χ, 1:2, 1:p.χ, 1:p.χ)
    for s in 1:2, j in 1:p.χ, i in 1:p.χ, ss in 1:2, jj in 1:p.χ, ii in 1:p.χ 
        @inbounds S[flatten_index(i,j,s,p),flatten_index(ii,jj,ss,p)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    S+=ϵ*Matrix{Int}(I, 2*p.χ^2, 2*p.χ^2)

    grad = (L∇L-ΔLL)/N_MC
    flat_grad = reshape(grad, 2*p.χ^2)

    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad, p.χ, p.χ,2 )

    return grad, mean_local_Hamiltonian, acceptance/(p.N*N_MC)
end



function sample_with_SR_MPS(p::parameters, A::Array{Float64}, h1::Matrix{ComplexF64}, N_MC::Int64)

    #Initialize variables:
    L∇L::Array{Float64,3} = zeros(Float64, p.χ, p.χ, 2)
    ΔLL::Array{Float64,3} = zeros(Float64, p.χ, p.χ, 2)
    S=zeros(Float64, 2*p.χ^2, 2*p.χ^2)
    Left=zeros(Float64, p.χ, p.χ, 2)
    Right=zeros(Float64, p.χ, p.χ, 2)

    mean_local_Hamiltonian::Float64 = 0

    sample = rand(Bool, p.N)
    L_set = L_MPS_strings(p, sample, A)
    for _ in 1:N_MC
        sample, R_set = Mono_Metropolis_sweep_left(p, sample, A, L_set)
        ρ_sample = tr(R_set[p.N+1])

        # Prepare new L_set of left MPS strings:
        L_set = [ Matrix{Float64}(undef, p.χ, p.χ) for _ in 1:p.N+1 ]
        L = Matrix{Float64}(I, p.χ, p.χ)
        L_set[1] = L

        e_field::Float64 = 0
        e_int::Float64 = 0

        #L∇L*:
        for j::UInt8 in 1:p.N

            #1-local part (field):
            bra_L::Transpose{ComplexF64, Vector{ComplexF64}} = transpose(dVEC2[sample[j]])*h1
            for i::UInt8 in 1:2
                loc::ComplexF64 = bra_L[i]
                if loc!=0
                    state::Bool = TPSC2[i]
                    @inbounds e_field -= loc*tr(L_set[j]*A[:,:,2-state]*R_set[p.N+1-j])
                end
            end

            #Interaction term:
            @inbounds e_int -= p.J * (2*sample[j]-1) * (2*sample[mod(j,p.N)+1]-1)

            #Update L_set:
            L*=A[:,:,2-sample[j]]
            L_set[j+1] = L
        end

        e_field/=ρ_sample
        # e_int/=ρ_sample ???

        Δ_MPO_sample = derv_MPS(p, sample, L_set, R_set)/ρ_sample

        #Add energies:
        local_E::Float64 = real(e_int+e_field)
        local_∇E::Array{Float64,3} = real(e_int+e_field)*Δ_MPO_sample
        L∇L += local_∇E

        #ΔLL:
        local_Δ = Δ_MPO_sample
        ΔLL += local_Δ

        #Mean local Lindbladian:
        mean_local_Hamiltonian += real(local_E)

        #Metric tensor:
        G = Δ_MPO_sample
        Left += G #change order of conjugation, but it shouldn't matter
        Right+= conj(G)
        #@inbounds for (s, j, i, ss, jj, ii) in zip(1:2, 1:p.χ, 1:p.χ, 1:2, 1:p.χ, 1:p.χ)
        for s in 1:2, j in 1:p.χ, i in 1:p.χ, ss in 1:2, jj in 1:p.χ, ii in 1:p.χ
        #for (s,j,i,ss,jj,ii) in Iterators.product(1:2, 1:p.χ, 1:p.χ, 1:2, 1:p.χ, 1:p.χ)
            #println(s, " ", j, " ", i, " ", ss, " ", jj, " ", ii)
            @inbounds S[flatten_index(i,j,s,p),flatten_index(ii,jj,ss,p)] += conj(G[i,j,s])*G[ii,jj,ss]
        end
    end
    #mean_local_Hamiltonian/=N_MC
    #ΔLL*=mean_local_Hamiltonian
    return dist_output(L∇L, ΔLL, mean_local_Hamiltonian, S, Left, Right)
    #return [L∇L::Array{Float64,3}, ΔLL::Array{Float64,3}, mean_local_Hamiltonian::Float64, S::Array{Float64, 2}, Left::Array{Float64, 3}, Right::Array{Float64, 3}]
end

# The only purpose of this struct is to ensure type-stability during reduction with @distributed
# not sure if there is a better way to do this
mutable struct dist_output
    L∇L::Array{Float64, 3}
    ΔLL::Array{Float64, 3}
    mean_local_Hamiltonian::Float64
    S::Array{Float64, 2}
    Left::Array{Float64, 3}
    Right::Array{Float64, 3}
end
Base.:+(x::dist_output, y::dist_output) = dist_output(x.L∇L + y.L∇L, x.ΔLL + y.ΔLL, x.mean_local_Hamiltonian + y.mean_local_Hamiltonian, x.S + y.S, x.Left + y.Left, x.Right + y.Right)


function distributed_SR_calculate_MC_MPS_gradient(p::parameters, A::Array{Float64}, N_MC::Int64, ϵ::Float64)

    # Define 1-local Hamiltonian:
    h1::Matrix{ComplexF64} = p.h*sx

    #perform reduction:
    output::dist_output = @distributed (+) for i=1:nworkers()
        #sample_with_SR_long_range_MPS(p, A, l1, N_MC, N_sweeps)
        sample_with_SR_MPS(p, A, h1, N_MC)
    end

    #L∇L=output[1]
    #ΔLL=output[2]
    #mean_local_Hamiltonian=output[3]
    #S=output[4]
    #Left=output[5]
    #Right=output[6]

    L∇L=output.L∇L
    ΔLL=output.ΔLL
    mean_local_Hamiltonian=output.mean_local_Hamiltonian
    S=output.S
    Left=output.Left
    Right=output.Right

    #mean_local_Hamiltonian/=(nworkers())
    #ΔLL/=(nworkers())
    mean_local_Hamiltonian/=(N_MC*nworkers())
    ΔLL*=mean_local_Hamiltonian

    #Metric tensor:
    S/=(N_MC*nworkers())
    Left/=(N_MC*nworkers())
    Right/=(N_MC*nworkers())

    for s in 1:2, j in 1:p.χ, i in 1:p.χ, ss in 1:2, jj in 1:p.χ, ii in 1:p.χ
        @inbounds S[flatten_index(i,j,s,p),flatten_index(ii,jj,ss,p)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    #S+=max(0.001,1*0.95^step)*Matrix{Int}(I, χ*χ*4, χ*χ*4)
    S+=ϵ*Matrix{Int}(I, p.χ*p.χ*2, p.χ*p.χ*2)

    grad = (L∇L-ΔLL)/(N_MC*nworkers())
    flat_grad = reshape(grad, 2*p.χ^2)
    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad, p.χ, p.χ, 2)

    return grad, real(mean_local_Hamiltonian)
end

