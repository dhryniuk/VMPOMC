export SR_calculate_gradient, SR_calculate_MC_gradient_full, SR_LdagL_gradient, MT_SR_MC_grad, multi_threaded_SR_calculate_MC_gradient_full

function SR_calculate_gradient(params::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64},ϵ,basis)
    #χ=size(A[:,:,1])[1]

    L∇L=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL=zeros(ComplexF64,params.χ,params.χ,4)
    Z=0

    mean_local_Lindbladian = 0

    # Metric tensor auxiliary arrays:
    S = zeros(ComplexF64,4*params.χ^2,4*params.χ^2)
    G = zeros(ComplexF64,params.χ,params.χ,4)
    Left = zeros(ComplexF64,params.χ,params.χ,4)
    Right = zeros(ComplexF64,params.χ,params.χ,4)
    function flatten_index(i,j,s)
        return i+params.χ*(j-1)+params.χ^2*(s-1)
    end

    for k in 1:params.dim
        for l in 1:params.dim
            sample = density_matrix(1,basis[k],basis[l])
            L_set = L_MPO_strings(params, sample, A)
            R_set = R_MPO_strings(params, sample, A)
            ρ_sample = tr(L_set[params.N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z+=p_sample

            local_L=0
            local_∇L=zeros(ComplexF64,params.χ,params.χ,4)
            l_int = 0

            L_set = Vector{Matrix{ComplexF64}}()
            L=Matrix{ComplexF64}(I, params.χ, params.χ)
            push!(L_set,copy(L))

            #L∇L*:
            for j in 1:params.N

                #1-local part:
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*conj(l1)
                for i in 1:4
                    loc = bra_L[i]
                    if loc!=0
                        state = TPSC[i]
                        local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[params.N+1-j])
                        micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                        micro_sample.ket[j] = state[1]
                        micro_sample.bra[j] = state[2]
                        
                        micro_L_set = L_MPO_strings(params,micro_sample, A)
                        micro_R_set = R_MPO_strings(params, micro_sample, A)
                        local_∇L+= loc*derv_MPO(params, micro_sample,micro_L_set,micro_R_set)
                    end
                end

                #2-local part:
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,params.N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,params.N)+1]-1)
                l_int += 1.0im*params.J*(l_int_α-l_int_β)

                #Update L_set:
                L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
                push!(L_set,copy(L))
            end

            local_L /=ρ_sample
            local_∇L/=ρ_sample
    
            Δ_MPO_sample = derv_MPO(params, sample, L_set, R_set)/ρ_sample
    
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
            for s in 1:4, j in 1:params.χ, i in 1:params.χ, ss in 1:4, jj in 1:params.χ, ii in 1:params.χ
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
    for s in 1:4, j in 1:params.χ, i in 1:params.χ, ss in 1:4, jj in 1:params.χ, ii in 1:params.χ
        S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    S+=ϵ*Matrix{Int}(I, params.χ*params.χ*4, params.χ*params.χ*4)

    grad = (L∇L-ΔLL)/Z
    flat_grad = reshape(grad,4*params.χ^2)
    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad,params.χ,params.χ,4)

    return grad, real(mean_local_Lindbladian)
end


function SR_calculate_MC_gradient_full(params::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, N_MC::Int64, N_sweeps::Int64, ϵ::Float64)
    L∇L=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL=zeros(ComplexF64,params.χ,params.χ,4)

    mean_local_Lindbladian = 0

    sample = density_matrix(1,rand(0:1,params.N),rand(0:1,params.N))
    L_set = L_MPO_strings(params, sample, A)

    # Metric tensor auxiliary arrays:
    S = zeros(ComplexF64,4*params.χ^2,4*params.χ^2)
    G = zeros(ComplexF64,params.χ,params.χ,4)
    Left = zeros(ComplexF64,params.χ,params.χ,4)
    Right = zeros(ComplexF64,params.χ,params.χ,4)
    function flatten_index(i,j,s)
        return i+params.χ*(j-1)+params.χ^2*(s-1)
    end

    for k in 1:N_MC

        sample, R_set = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        for n in N_sweeps
            sample, L_set = Mono_Metropolis_sweep_right(params, sample, A, R_set)
            sample, R_set = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        end
        ρ_sample = tr(R_set[params.N+1])
        L_set = Vector{Matrix{ComplexF64}}()
        L=Matrix{ComplexF64}(I, params.χ, params.χ)
        push!(L_set,copy(L))

        local_L=0
        local_∇L=zeros(ComplexF64,params.χ,params.χ,4)
        l_int = 0

        #L∇L*:
        for j in 1:params.N

            #1-local part:
            s = dVEC[(sample.ket[j],sample.bra[j])]
            bra_L = transpose(s)*conj(l1)
            for i in 1:4
                loc = bra_L[i]
                if loc!=0
                    state = TPSC[i]
                    local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[params.N+1-j])
                    
                    micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                    micro_sample.ket[j] = state[1]
                    micro_sample.bra[j] = state[2]

                    micro_L_set = L_MPO_strings(params, micro_sample, A)
                    micro_R_set = R_MPO_strings(params, micro_sample, A)
                    local_∇L+= loc*derv_MPO(params, micro_sample, micro_L_set, micro_R_set)
                end
            end

            #2-local part:
            l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,params.N)+1]-1)
            l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,params.N)+1]-1)
            l_int += 1.0im*params.J*(l_int_α-l_int_β)

            #Update L_set:
            L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
            push!(L_set,copy(L))
        end

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        Δ_MPO_sample = derv_MPO(params, sample, L_set, R_set)/ρ_sample

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
        for s in 1:4, j in 1:params.χ, i in 1:params.χ, ss in 1:4, jj in 1:params.χ, ii in 1:params.χ
            S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] += conj(G[i,j,s])*G[ii,jj,ss]
        end
    end
    mean_local_Lindbladian/=N_MC
    ΔLL*=mean_local_Lindbladian

    #Metric tensor:
    S./=N_MC
    Left./=N_MC
    Right./=N_MC
    for s in 1:4, j in 1:params.χ, i in 1:params.χ, ss in 1:4, jj in 1:params.χ, ii in 1:params.χ
        S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    #S+=max(0.001,1*0.95^step)*Matrix{Int}(I, χ*χ*4, χ*χ*4)
    S+=ϵ*Matrix{Int}(I, params.χ*params.χ*4, params.χ*params.χ*4)

    grad = (L∇L-ΔLL)/N_MC
    flat_grad = reshape(grad,4*params.χ^2)
    flat_grad = inv(conj.(S))*flat_grad
    grad = reshape(flat_grad,params.χ,params.χ,4)

    return grad, real(mean_local_Lindbladian)
end

function multi_threaded_SR_calculate_MC_gradient_full(params::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, N_MC::Int64, N_sweeps::Int64, ϵ::Float64)
    tL∇L = [zeros(ComplexF64,params.χ,params.χ,4) for _ in 1:2*Threads.nthreads()]
    tΔLL = [zeros(ComplexF64,params.χ,params.χ,4) for _ in 1:2*Threads.nthreads()]

    tmean_local_Lindbladian = [0.0+0.0im for _ in 1:2*Threads.nthreads()]

    # Metric tensor auxiliary arrays for each individual thread:
    tS = [zeros(ComplexF64,4*params.χ^2,4*params.χ^2) for _ in 1:2*Threads.nthreads()]
    #G = [zeros(ComplexF64,params.χ,params.χ,4) for _ in 1:2*Threads.nthreads()]
    tLeft  = [zeros(ComplexF64,params.χ,params.χ,4) for _ in 1:2*Threads.nthreads()]
    tRight = [zeros(ComplexF64,params.χ,params.χ,4) for _ in 1:2*Threads.nthreads()]
    function flatten_index(i,j,s)
        return i+params.χ*(j-1)+params.χ^2*(s-1)
    end

    Threads.@threads for t in 1:(2*Threads.nthreads())
        sample = density_matrix(1,rand(0:1,params.N),rand(0:1,params.N))
        L_set = L_MPO_strings(params, sample, A)
        for k in 1:N_MC

            sample, R_set = Mono_Metropolis_sweep_left(params, sample, A, L_set)
            for n in N_sweeps
                sample, L_set = Mono_Metropolis_sweep_right(params, sample, A, R_set)
                sample, R_set = Mono_Metropolis_sweep_left(params, sample, A, L_set)
            end
            ρ_sample = tr(R_set[params.N+1])
            L_set = Vector{Matrix{ComplexF64}}()
            L=Matrix{ComplexF64}(I, params.χ, params.χ)
            push!(L_set,copy(L))

            local_L=0
            local_∇L=zeros(ComplexF64,params.χ,params.χ,4)
            l_int = 0

            #L∇L*:
            for j in 1:params.N

                #1-local part:
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*conj(l1)
                for i in 1:4
                    loc = bra_L[i]
                    if loc!=0
                        state = TPSC[i]
                        local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[params.N+1-j])
                        
                        micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                        micro_sample.ket[j] = state[1]
                        micro_sample.bra[j] = state[2]

                        micro_L_set = L_MPO_strings(params, micro_sample, A)
                        micro_R_set = R_MPO_strings(params, micro_sample, A)
                        local_∇L+= loc*derv_MPO(params, micro_sample, micro_L_set, micro_R_set)
                    end
                end

                #2-local part:
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,params.N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,params.N)+1]-1)
                l_int += 1.0im*params.J*(l_int_α-l_int_β)

                #Update L_set:
                L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
                push!(L_set,copy(L))
            end

            local_L /=ρ_sample
            local_∇L/=ρ_sample

            Δ_MPO_sample = derv_MPO(params, sample, L_set, R_set)/ρ_sample

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
            for s in 1:4, j in 1:params.χ, i in 1:params.χ, ss in 1:4, jj in 1:params.χ, ii in 1:params.χ
                tS[t][flatten_index(i,j,s),flatten_index(ii,jj,ss)] += conj(G[i,j,s])*G[ii,jj,ss]
            end
        end
    end
    #mean_local_Lindbladian/=N_MC
    #ΔLL*=mean_local_Lindbladian
    mean_local_Lindbladian = sum(tmean_local_Lindbladian)/(2*Threads.nthreads()*N_MC)
    ΔLL = sum(tΔLL)/(2*Threads.nthreads())
    ΔLL*=mean_local_Lindbladian
    L∇L = sum(tL∇L)/(2*Threads.nthreads())

    #Metric tensor:
    S = sum(tS)/(2*Threads.nthreads()*N_MC)
    Left = sum(tLeft)/(2*Threads.nthreads()*N_MC)
    Right= sum(tRight)/(2*Threads.nthreads()*N_MC)

    #S./=N_MC
    #Left./=N_MC
    #Right./=N_MC
    for s in 1:4, j in 1:params.χ, i in 1:params.χ, ss in 1:4, jj in 1:params.χ, ii in 1:params.χ
        S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    #S+=max(0.001,1*0.95^step)*Matrix{Int}(I, χ*χ*4, χ*χ*4)
    S+=ϵ*Matrix{Int}(I, params.χ*params.χ*4, params.χ*params.χ*4)

    grad = (L∇L-ΔLL)/N_MC
    flat_grad = reshape(grad,4*params.χ^2)
    flat_grad = inv(conj.(S))*flat_grad
    grad = reshape(flat_grad,params.χ,params.χ,4)

    return grad, real(mean_local_Lindbladian)
end

# THIS FUNCTION IS WRONG!!!
function MT_SR_MC_grad(params::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, N_MC::Int64, N_sweeps::Int64, ϵ::Float64)
    grad = [Array{ComplexF64}(undef,params.χ,params.χ,4) for _ in 1:2*Threads.nthreads()]
    mean_local_Lindbladian = [0.0 for _ in 1:2*Threads.nthreads()]
    Threads.@threads for t in 1:(2*Threads.nthreads())
        g,m=SR_calculate_MC_gradient_full(params, A, l1, N_MC, N_sweeps, ϵ)
        grad[t]=g
        mean_local_Lindbladian[t]=m
        #grad[t], mean_local_Lindbladian[t] = SR_calculate_MC_gradient_full(params, A, l1, N_MC, N_sweeps, ϵ)
    end
    #display(mean_local_Lindbladian)
    return sum(grad)/(2*Threads.nthreads()), sum(mean_local_Lindbladian)/(2*Threads.nthreads())
end


function SR_LdagL_gradient(params,A,l1,ϵ,basis)
    ΔLL=zeros(ComplexF64,params.χ,params.χ,4)
    Z=0

    mean_local_Lindbladian = 0
    mean_Δ = zeros(ComplexF64,params.χ,params.χ,4)

    # Metric tensor auxiliary arrays:
    S = zeros(ComplexF64,4*params.χ^2,4*params.χ^2)
    G = zeros(ComplexF64,params.χ,params.χ,4)
    Left = zeros(ComplexF64,params.χ,params.χ,4)
    Right = zeros(ComplexF64,params.χ,params.χ,4)
    function flatten_index(i,j,s)
        return i+params.χ*(j-1)+params.χ^2*(s-1)
    end    

    for k in 1:params.dim
        for l in 1:params.dim
            sample = density_matrix(1,basis[k],basis[l])
            L_set = L_MPO_strings(params, sample, A)
            R_set = R_MPO_strings(params, sample, A)
            ρ_sample = tr(L_set[params.N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z+=p_sample

            local_L=0
            l_int = 0

            L_set = Vector{Matrix{ComplexF64}}()
            L=Matrix{ComplexF64}(I, params.χ, params.χ)
            push!(L_set,copy(L))

            #L∇L*:
            for j in 1:params.N

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

                        for yj in 1:params.N
                            y_s = dVEC[(y_sample.ket[yj],y_sample.bra[yj])]
                            y_bra_L = transpose(y_s)*l1
                            for yi in 1:4
                                y_loc = y_bra_L[yi]
                                if y_loc!=0
                                    y_state = TPSC[yi]
                                    z_sample = density_matrix(1,deepcopy(y_sample.ket),deepcopy(y_sample.bra))
                                    z_sample.ket[yj] = y_state[1]
                                    z_sample.bra[yj] = y_state[2]
                                    local_L += loc*y_loc*MPO(params,z_sample,A)
                                end
                            end
                        end
                    end
                end

                #2-local part:
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,params.N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,params.N)+1]-1)
                #l_int += -1.0im*J*(l_int_α-l_int_β)
                l_int += 1.0im*params.J*(l_int_α-l_int_β)
                l_int*=conj(l_int)

                #Update L_set:
                L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
                push!(L_set,copy(L))
            end

            local_L /= ρ_sample
            #Add in interaction terms:
            local_L += l_int    #/=ρ_sample?
    
            Δ_MPO_sample = derv_MPO(params, sample, L_set, R_set)/ρ_sample
    
            #ΔLL:
            mean_Δ+=p_sample*conj(Δ_MPO_sample)
            ΔLL+=p_sample*conj(Δ_MPO_sample)*local_L
    
            #Mean local Lindbladian:
            mean_local_Lindbladian += p_sample*local_L

            #Metric tensor:
            G = Δ_MPO_sample
            Left += p_sample*conj(G)
            Right+= p_sample*G
            for s in 1:4, j in 1:params.χ, i in 1:params.χ, ss in 1:4, jj in 1:params.χ, ii in 1:params.χ
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
    for s in 1:4, j in 1:params.χ, i in 1:params.χ, ss in 1:4, jj in 1:params.χ, ii in 1:params.χ
        S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    S+=ϵ*Matrix{Int}(I, params.χ*params.χ*4, params.χ*params.χ*4)

    grad = (ΔLL-mean_Δ)/Z
    flat_grad = reshape(grad,4*params.χ^2)
    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad,params.χ,params.χ,4)

    return grad, real(mean_local_Lindbladian)
end