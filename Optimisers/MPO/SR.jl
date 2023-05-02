export SR_MPO_gradient

function MPO_flatten_index(i::UInt8,j::UInt8,s::UInt8,params::parameters)
    return i+params.χ*(j-1)+params.χ^2*(s-1)
end

function sample_update_SR(S::Array{<:Complex{<:AbstractFloat},2}, avg_G::Array{<:Complex{<:AbstractFloat}}, 
    Δ_MPO_sample::Array{<:Complex{<:AbstractFloat},3}, params::parameters, AUX::workspace)
    
    G = reshape(Δ_MPO_sample,4*params.χ^2)
    conj_G = conj(G)
    avg_G.+= G
    mul!(AUX.plus_S,conj_G,transpose(G))
    S.+=AUX.plus_S
    return S, avg_G
end

function apply_SR(S::Array{<:Complex{<:AbstractFloat},2}, avg_G::Array{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, 
    L∇L::Array{<:Complex{<:AbstractFloat},3}, ΔLL::Array{<:Complex{<:AbstractFloat},3}, params::parameters)

    #Metric tensor:
    S./=N_MC
    avg_G./=N_MC
    conj_avg_G = conj(avg_G)
    S-=avg_G*transpose(conj_avg_G)

    #Regularize the metric tensor:
    S+=ϵ*Matrix{Int}(I, params.χ*params.χ*4, params.χ*params.χ*4)

    #Find SR'd gradient:
    grad::Array{eltype(S),3} = (L∇L-ΔLL)/N_MC
    flat_grad::Vector{eltype(S)} = reshape(grad,4*params.χ^2)
    flat_grad = inv(S)*flat_grad
    grad = reshape(flat_grad,params.χ,params.χ,4)
    return grad
end

function SR_MPO_gradient(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::parameters)
    
    # Define ensemble averages:
    L∇L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    ΔLL::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    mean_local_Lindbladian::eltype(A) = 0

    # Preallocate auxiliary arrays:
    AUX = workspace(
        [ Matrix{eltype(A)}(undef,params.χ,params.χ) for _ in 1:params.N+1 ],
        [ Matrix{eltype(A)}(undef,params.χ,params.χ) for _ in 1:params.N+1 ],
        zeros(eltype(A), 4*params.χ^2,4*params.χ^2),
        zeros(eltype(A), params.χ,params.χ),
        Matrix{eltype(A)}(I, params.χ, params.χ),
        zeros(eltype(A), params.χ,params.χ),
        zeros(eltype(A), params.χ,params.χ),
        zeros(eltype(A), params.χ,params.χ),
        zeros(eltype(A), params.χ,params.χ),
        zeros(eltype(A), params.χ,params.χ),
        zeros(eltype(A), 1, 4),
        zeros(eltype(A), params.χ, params.χ, 4)
    )



    #dVEC_transpose::Dict{Tuple{Bool,Bool},Matrix{eltype(A)}} = Dict((0,0) => [1 0 0 0], (0,1) => [0 1 0 0], (1,0) => [0 0 1 0], (1,1) => [0 0 0 1])

    # Initialize sample and L_set for that sample:
    sample, L_set = MPO_Metropolis_burn_in(A, params, AUX)
    acceptance::UInt64=0

    # Metric tensor auxiliary arrays:
    S::Array{eltype(A),2} = zeros(eltype(A),4*params.χ^2,4*params.χ^2)
    Left::Array{eltype(A)} = zeros(eltype(A),4*params.χ^2)

    for _ in 1:N_MC

        sample, R_set, acc = Mono_Metropolis_sweep_left(sample, A, L_set, params, AUX)
        acceptance+=acc
        #for n in N_sweeps
        #    sample, L_set = Mono_Metropolis_sweep_right(p, sample, A, R_set)
        #    sample, R_set = Mono_Metropolis_sweep_left(p, sample, A, L_set)
        #end
        ρ_sample::eltype(A) = tr(R_set[params.N+1])

        L_set::Vector{Matrix{eltype(A)}} = [ Matrix{eltype(A)}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
        L = Matrix{eltype(A)}(I, params.χ, params.χ)
        L_set[1] = L

        local_L::eltype(A) = 0
        local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
        l_int::eltype(A) = 0

        #L∇L*:
        for j::UInt8 in 1:params.N
            #1-local part:
            lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,L_set,R_set,params,AUX)
            local_L  += lL
            local_∇L.+= l∇L
            #Update L_set:
            mul!(L_set[j+1], L_set[j], @view(A[:,:,1+2*sample.ket[j]+sample.bra[j]]))
        end

        l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)
        #l_int = N4_Lindblad_Ising_interaction_energy_2D(params, sample)
        #println("2d: ", l_int)
        #println("1d: ", Lindblad_Ising_interaction_energy(params, sample, "periodic"))

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        AUX.Δ_MPO_sample = ∂MPO(sample, L_set, R_set, params, AUX)./ρ_sample


        #Add in interaction terms:
        local_L +=l_int
        local_∇L+=l_int*AUX.Δ_MPO_sample

        L∇L+=local_L*conj(local_∇L)

        #ΔLL:
        local_Δ=conj(AUX.Δ_MPO_sample)
        ΔLL+=local_Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

        #Metric tensor:
        S, Left = sample_update_SR(S, Left, AUX.Δ_MPO_sample, params, AUX)
    end
    mean_local_Lindbladian/=N_MC
    ΔLL*=mean_local_Lindbladian

    grad = apply_SR(S,Left,N_MC,ϵ,L∇L,ΔLL,params)

    return grad, real(mean_local_Lindbladian), acceptance/(N_MC*params.N)
end