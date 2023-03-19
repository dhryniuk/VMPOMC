export SR_MPO_gradient

function MPO_flatten_index(i::UInt8,j::UInt8,s::UInt8,params::parameters)
    return i+params.χ*(j-1)+params.χ^2*(s-1)
end

function sample_update_SR(params::parameters, S::Array{ComplexF64,2}, Left::Array{ComplexF64,3}, Δ_MPO_sample::Array{ComplexF64,3})
    G = Δ_MPO_sample
    conj_G = conj(G)
    Left += G
    for s::UInt8 in 1:4, j::UInt8 in 1:params.χ, i::UInt8 in 1:params.χ, 
        ss::UInt8 in 1:4, jj::UInt8 in 1:params.χ, ii::UInt8 in 1:params.χ
        @inbounds S[MPO_flatten_index(i,j,s,params),MPO_flatten_index(ii,jj,ss,params)] += conj_G[i,j,s]*G[ii,jj,ss]
    end
    return S, Left
end

function apply_SR(params::parameters, S::Array{ComplexF64,2}, Left::Array{ComplexF64,3}, N_MC::Int64, 
    ϵ::Float64, L∇L::Array{ComplexF64,3}, ΔLL::Array{ComplexF64,3})
    #Metric tensor:
    S./=N_MC
    Left./=N_MC
    Right = conj(Left)
    for s::UInt8 in 1:4, j::UInt8 in 1:params.χ, i::UInt8 in 1:params.χ, 
        ss::UInt8 in 1:4, jj::UInt8 in 1:params.χ, ii::UInt8 in 1:params.χ
        @inbounds S[MPO_flatten_index(i,j,s,params),MPO_flatten_index(ii,jj,ss,params)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    #Regularize the metric tensor:
    S+=ϵ*Matrix{Int}(I, params.χ*params.χ*4, params.χ*params.χ*4)

    #Find SR'd gradient:
    grad::Array{ComplexF64,3} = (L∇L-ΔLL)/N_MC
    flat_grad::Vector{ComplexF64} = reshape(grad,4*params.χ^2)
    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad,params.χ,params.χ,4)
    return grad
end

function SR_MPO_gradient(p::parameters, A::Array{ComplexF64}, l1::Matrix{ComplexF64}, N_MC::Int64, ϵ::Float64)
    L∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)

    mean_local_Lindbladian::ComplexF64 = 0

    # Initialize sample and L_set for that sample:
    sample, L_set = MPO_Metropolis_burn_in(params, A)
    acceptance::UInt64=0

    # Metric tensor auxiliary arrays:
    S::Array{ComplexF64,2} = zeros(ComplexF64,4*params.χ^2,4*params.χ^2)
    #G::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
    Left::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
    Right::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)

    for _ in 1:N_MC

        sample, R_set = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        #for n in N_sweeps
        #    sample, L_set = Mono_Metropolis_sweep_right(p, sample, A, R_set)
        #    sample, R_set = Mono_Metropolis_sweep_left(p, sample, A, L_set)
        #end
        ρ_sample::ComplexF64 = tr(R_set[params.N+1])

        L_set::Vector{Matrix{ComplexF64}} = [ Matrix{ComplexF64}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
        L = Matrix{ComplexF64}(I, params.χ, params.χ)
        L_set[1] = L

        local_L::ComplexF64 = 0
        local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
        l_int::ComplexF64 = 0

        #L∇L*:
        for j::UInt16 in 1:params.N

            #1-local part:
            lL, l∇L = one_body_Lindblad_term(params,sample,j,l1,A,L_set,R_set)
            #lL, l∇L = one_body_Lindblad_term(params,sample_ket,sample_bra,j,l1,A,L_set,R_set)
            local_L += lL
            local_∇L += l∇L
            #Update L_set:
            L*=A[:,:,1+2*sample.ket[j]+sample.bra[j]]
            L_set[j+1] = L
        end

        l_int = Lindblad_Ising_interaction_energy(params, sample, "periodic")

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        Δ_MPO_sample = ∂MPO(p, sample, L_set, R_set)/ρ_sample

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
        S, Left = sample_update_SR(params, S, Left, Δ_MPO_sample)
        #G = Δ_MPO_sample
        #Left += G #change order of conjugation, but it shouldn't matter
        #for s in 1:4, j in 1:params.χ, i in 1:params.χ, ss in 1:4, jj in 1:params.χ, ii in 1:params.χ
        #    S[MPO_flatten_index(i,j,s,params),MPO_flatten_index(ii,jj,ss,params)] += conj(G[i,j,s])*G[ii,jj,ss]
        #end
    end
    mean_local_Lindbladian/=N_MC
    ΔLL*=mean_local_Lindbladian

    """
    #Metric tensor:
    S./=N_MC
    Left./=N_MC
    Right = conj(Left)
    #Right./=N_MC
    for s in 1:4, j in 1:params.χ, i in 1:params.χ, ss in 1:4, jj in 1:params.χ, ii in 1:params.χ
        S[MPO_flatten_index(i,j,s,params),MPO_flatten_index(ii,jj,ss,params)] -= Left[i,j,s]*Right[ii,jj,ss]
    end

    #S+=max(0.001,1*0.95^step)*Matrix{Int}(I, χ*χ*4, χ*χ*4)
    S+=ϵ*Matrix{Int}(I, params.χ*params.χ*4, params.χ*params.χ*4)

    grad = (L∇L-ΔLL)/N_MC
    flat_grad = reshape(grad,4*params.χ^2)
    flat_grad = inv(S)*flat_grad
    #flat_grad = inv(conj.(S))*flat_grad
    grad = reshape(flat_grad,params.χ,params.χ,4)
    """

    grad = apply_SR(params,S,Left,N_MC,ϵ,L∇L,ΔLL)

    return grad, real(mean_local_Lindbladian)
end