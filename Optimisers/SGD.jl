export calculate_gradient, calculate_MC_gradient_full, LdagL_gradient, MT_SGD_MC_grad, calculate_MPS_gradient, SGD_MC_grad_distributed

function B_list(m, sample, A) #FIX m ORDERING
    B_list=Matrix{ComplexF64}[Matrix{Int}(I, χ, χ)]
    for j::UInt8 in 1:N-1
        push!(B_list,A[:,:,dINDEX[(sample.ket[mod(m+j-1,N)+1],sample.bra[mod(m+j-1,N)+1])]])
    end
    return B_list
end

function OLDderv_MPO(sample, A)
    ∇=zeros(ComplexF64, χ,χ,4)
    for m::UInt8 in 1:N
        B = prod(B_list(m, sample, A))
        for i in 1:χ
            for j in 1:χ
                ∇[i,j,dINDEX[(sample.ket[m],sample.bra[m])]] += B[i,j] + B[j,i]
            end
            ∇[i,i,:]./=2
        end
    end
    return ∇
end

function calculate_gradient(params::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, basis)
    L∇L=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL=zeros(ComplexF64,params.χ,params.χ,4)
    Z=0

    mean_local_Lindbladian = 0

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
                #bra_L = transpose(s)*l1
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
                        local_∇L+= loc*derv_MPO(params, micro_sample,micro_L_set,micro_R_set)
                    end
                end

                #2-local part:
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,params.N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,params.N)+1]-1)
                #l_int += -1.0im*J*(l_int_α-l_int_β)
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
            #L∇L+=p_sample*conj(local_L)*local_∇L
    
            #ΔLL:
            local_Δ=p_sample*conj(Δ_MPO_sample)
            #local_Δ=p_sample*Δ_MPO_sample
            ΔLL+=local_Δ
    
            #Mean local Lindbladian:
            mean_local_Lindbladian += p_sample*local_L*conj(local_L)
        end
    end
    mean_local_Lindbladian/=Z
    ΔLL*=mean_local_Lindbladian
    return (L∇L-ΔLL)/Z, real(mean_local_Lindbladian)
end

function calculate_MPS_gradient(params::parameters, A::Array{Float64}, basis)
    L∇L=zeros(Float64,params.χ,params.χ,2) #coupled product
    ΔLL=zeros(Float64,params.χ,params.χ,2) #uncoupled product
    Z=0

    mean_local_Hamiltonian = 0

    l1 = params.h*sx

    for k in 1:params.dim
        sample = basis[k]
        #println(sample)
        L_set = L_MPS_strings(params, sample, A)
        R_set = R_MPS_strings(params, sample, A)
        ρ_sample = tr(L_set[params.N+1])
        #ρ_sample = MPS(params, sample, A)
        p_sample = ρ_sample*conj(ρ_sample)
        Z+=p_sample

        local_E=0
        local_∇L=zeros(Float64, params.χ, params.χ, 2)

        L_set = Vector{Matrix{Float64}}()
        L=Matrix{Float64}(I, params.χ, params.χ)
        push!(L_set,copy(L))

        e_field=0
        e_int=0
        #L∇L*:
        for j in 1:params.N

            #1-local part (field):
            s = dVEC2[sample[j]]
            bra_L = transpose(s)*l1
            for i in 1:2
                loc = bra_L[i]
                if loc!=0
                    state = TPSC2[i]
                    #new_sample = deepcopy(sample)
                    #new_sample[j] = state
                    #e_field -= loc*MPS(params,new_sample,A)
                    e_field -= loc*tr(L_set[j]*A[:,:,dINDEX2[state]]*R_set[params.N+1-j])
                end
            end

            #Interaction term:
            e_int -= params.J * (2*sample[j]-1) * (2*sample[mod(j,params.N)+1]-1)

            #Update L_set:
            L*=A[:,:,dINDEX2[sample[j]]]
            push!(L_set,copy(L))
        end

        e_field/=ρ_sample

        #println("F: ", e_field)
        #println("I: ", e_int)

        #e_int*=params.J

        #local_L /=ρ_sample
        #local_∇L/=ρ_sample

        #Δ_MPO_sample = derv_MPS(params, sample, A)/ρ_sample
        Δ_MPO_sample = derv_MPS(params, sample, L_set, R_set)/ρ_sample

        #Add in interaction terms:
        local_E  = real(e_int+e_field)
        local_∇E = real(e_int+e_field)*Δ_MPO_sample
        L∇L += p_sample*local_∇E # conj?

        #ΔLL:
        local_Δ = p_sample*Δ_MPO_sample
        #local_Δ=p_sample*Δ_MPO_sample
        ΔLL += local_Δ

        #Mean local Lindbladian:
        mean_local_Hamiltonian += real(p_sample*local_E)
    end
    #display(L∇L)
    #display(ΔLL)
    #display(Z)
    #display(mean_local_Hamiltonian)

    mean_local_Hamiltonian/=Z
    ΔLL*=mean_local_Hamiltonian

    #display((L∇L-ΔLL)/Z)

    return (L∇L-ΔLL)/Z, mean_local_Hamiltonian
end

function MC_calculate_MPS_gradient(params::parameters, A::Array{Float64}, N_MC::Int64)
    L∇L=zeros(Float64,params.χ,params.χ,2) #coupled product
    ΔLL=zeros(Float64,params.χ,params.χ,2) #uncoupled product

    mean_local_Hamiltonian = 0

    l1 = params.h*sx

    sample = rand(0:1,params.N)
    L_set = L_MPS_strings(params, sample, A)
    for k in 1:N_MC
        sample, R_set = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        #R_set = R_MPS_strings(params, sample, A)
        ρ_sample = tr(R_set[params.N+1])

        local_E=0
        local_∇L=zeros(Float64, params.χ, params.χ, 2)

        L_set = Vector{Matrix{Float64}}()
        L=Matrix{Float64}(I, params.χ, params.χ)
        push!(L_set,copy(L))

        e_field=0
        e_int=0

        #L∇L*:
        for j in 1:params.N

            #1-local part (field):
            s = dVEC2[sample[j]]
            bra_L = transpose(s)*l1
            for i in 1:2
                loc = bra_L[i]
                if loc!=0
                    state = TPSC2[i]
                    e_field -= loc*tr(L_set[j]*A[:,:,dINDEX2[state]]*R_set[params.N+1-j])
                end
            end

            #Interaction term:
            e_int -= params.J * (2*sample[j]-1) * (2*sample[mod(j,params.N)+1]-1)

            #Update L_set:
            L*=A[:,:,dINDEX2[sample[j]]]
            push!(L_set,copy(L))
        end

        e_field/=ρ_sample
        # e_int/=ρ_sample ???


        Δ_MPO_sample = derv_MPS(params, sample, L_set, R_set)/ρ_sample

        #Add in interaction terms:
        local_E  = real(e_int+e_field)
        local_∇E = real(e_int+e_field)*Δ_MPO_sample
        L∇L += local_∇E

        #ΔLL:
        local_Δ = Δ_MPO_sample
        #local_Δ=p_sample*Δ_MPO_sample
        ΔLL += local_Δ

        #Mean local Lindbladian:
        mean_local_Hamiltonian += real(local_E)
    end

    mean_local_Hamiltonian/=N_MC
    ΔLL*=mean_local_Hamiltonian

    return (L∇L-ΔLL)/N_MC, mean_local_Hamiltonian
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

function calculate_MC_gradient_full(params::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, N_MC::Int64, N_sweeps::Int64)
    L∇L=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL=zeros(ComplexF64,params.χ,params.χ,4)

    mean_local_Lindbladian = 0

    sample = density_matrix(1,ones(params.N),ones(params.N))
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

        L∇L+=local_L*conj(local_∇L)

        #ΔLL:
        local_Δ=conj(Δ_MPO_sample)
        ΔLL+=local_Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

    end
    mean_local_Lindbladian/=N_MC
    ΔLL*=mean_local_Lindbladian
    return [ (L∇L-ΔLL)/N_MC, real(mean_local_Lindbladian) ]
end

#TRY DISTRIBUTED VERSION
function MT_SGD_MC_grad(params::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, N_MC::Int64, N_sweeps::Int64)
    grad = [Array{ComplexF64}(undef,params.χ,params.χ,4) for _ in 1:Threads.nthreads()]
    mean_local_Lindbladian = [0.0 for _ in 1:Threads.nthreads()]
    Threads.@threads for t in 1:(Threads.nthreads())
        g,m=calculate_MC_gradient_full(params, A, l1, N_MC, N_sweeps)
        grad[t]=g
        mean_local_Lindbladian[t]=m
        #grad[t], mean_local_Lindbladian[t] = SR_calculate_MC_gradient_full(params, A, l1, N_MC, N_sweeps, ϵ)
    end
    #display(mean_local_Lindbladian)
    return sum(grad)/(Threads.nthreads()), sum(mean_local_Lindbladian)/(Threads.nthreads())
end

function SGD_MC_grad_distributed(params::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, N_MC::Int64, N_sweeps::Int64)
    output = @distributed (+) for i=1:nworkers()
        calculate_MC_gradient_full(params, A, l1, N_MC, N_sweeps)
    end
    output/=nworkers()
    return output[1], output[2]
    #return grad/nworkers(), mean_local_Lindbladian/nworkers()
end

function LdagL_gradient(params,A,l1,basis) #CHECK IF COMPLEX CONJUGATE AND TRANSPOSES ARE TAKEN CORRECTLY!!!
    ΔLL=zeros(ComplexF64,params.χ,params.χ,4)
    Z=0

    mean_local_Lindbladian = 0
    mean_Δ = zeros(ComplexF64,params.χ,params.χ,4)

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
                #bra_L = transpose(s)*transpose(l1)
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
                            #y_bra_L = transpose(y_s)*conj(l1)
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
        end
    end
    mean_local_Lindbladian/=Z
    mean_Δ*=mean_local_Lindbladian
    return (ΔLL-mean_Δ)/Z, real(mean_local_Lindbladian)
end