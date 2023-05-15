#export SR_MPO_gradient_two_body

function SR_MPO_gradient_two_body(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, l2::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::parameters)
    
    # Define ensemble averages:
    L∂L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    ΔLL::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    mean_local_Lindbladian::eltype(A) = 0

    # Preallocate auxiliary arrays:
    AUX = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample::projector = MPO_Metropolis_burn_in(A, params, AUX)
    acceptance::UInt64=0

    # Metric tensor auxiliary arrays:
    S::Array{eltype(A),2} = zeros(eltype(A),4*params.χ^2,4*params.χ^2)
    Left::Array{eltype(A)} = zeros(eltype(A),4*params.χ^2)

    for _ in 1:N_MC

        #Initialize auxiliary arrays:
        local_L::eltype(A) = 0
        local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
        l_int::eltype(A) = 0
        AUX.local_∇L_diagonal_coeff = 0

        #Generate sample:
        sample, acc = Mono_Metropolis_sweep_left(sample, A, params, AUX)
        acceptance+=acc

        ρ_sample::eltype(A) = tr(AUX.R_set[params.N+1])
        AUX.L_set = L_MPO_strings(AUX.L_set, sample,A,params,AUX)
        AUX.Δ = ∂MPO(sample, AUX.L_set, AUX.R_set, params, AUX)./ρ_sample

        #L∂L*:
        for j::UInt8 in 1:params.N
            #1-local part:
            lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,params,AUX)
            local_L  += lL
            local_∇L.+= l∇L
        end
        for j::UInt8 in 1:params.N-1
            lL, l∇L = two_body_Lindblad_term(sample,j,l2,A,params,AUX)
            local_L += lL
            local_∇L += l∇L
        end
        if params.N>2
            lL, l∇L = boundary_two_body_Lindblad_term(sample,l2,A,params,AUX)
            local_L += lL
            local_∇L += l∇L
        end

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        #Add in diagonal part of the local derivative:
        local_∇L.+=AUX.local_∇L_diagonal_coeff.*AUX.Δ

        #Add in interaction terms:
        l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)
        local_L +=l_int
        local_∇L+=l_int*AUX.Δ

        #Update L∂L* ensemble average:
        L∂L.+=local_L*conj(local_∇L)

        #Update ΔLL ensemble average:
        ΔLL.+=AUX.Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

        #Update metric tensor:
        S, Left = sample_update_SR(S, Left, params, AUX)
    end
    mean_local_Lindbladian/=N_MC
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate
    ΔLL*=real(mean_local_Lindbladian)

    #Reconfigure gradient:
    grad = apply_SR(S,Left,N_MC,ϵ,L∂L,ΔLL,params)

    return grad, real(mean_local_Lindbladian), acceptance/(N_MC*params.N)
end


### DISTRIBUTED VERSION:

function one_worker_SR_MPO_gradient_two_body(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, l2::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::parameters)
    
    # Define ensemble averages:
    L∂L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    ΔLL::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)
    mean_local_Lindbladian::eltype(A) = 0

    # Preallocate auxiliary arrays:
    AUX = set_workspace(A,params)

    # Initialize sample and L_set for that sample:
    sample = MPO_Metropolis_burn_in(A, params, AUX)
    acceptance::UInt64=0

    # Metric tensor auxiliary arrays:
    S::Array{eltype(A),2} = zeros(eltype(A),4*params.χ^2,4*params.χ^2)
    Left::Array{eltype(A)} = zeros(eltype(A),4*params.χ^2)

    for _ in 1:N_MC

        #Initialize auxiliary arrays:
        local_L::eltype(A) = 0
        local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
        l_int::eltype(A) = 0
        AUX.local_∇L_diagonal_coeff = 0

        #Generate sample:
        sample, acc = Mono_Metropolis_sweep_left(sample, A, params, AUX)
        acceptance+=acc

        ρ_sample::eltype(A) = tr(AUX.R_set[params.N+1])
        AUX.L_set = L_MPO_strings(AUX.L_set, sample,A,params,AUX)
        AUX.Δ = ∂MPO(sample, AUX.L_set, AUX.R_set, params, AUX)./ρ_sample

        #Calculate L∂L*:
        for j::UInt8 in 1:params.N
            #1-local part:
            lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,params,AUX)
            local_L  += lL
            local_∇L.+= l∇L
        end
        for j::UInt8 in 1:params.N-1
            lL, l∇L = two_body_Lindblad_term(sample,j,l2,A,params,AUX)
            local_L += lL
            local_∇L += l∇L
        end
        if params.N>2
            lL, l∇L = boundary_two_body_Lindblad_term(sample,l2,A,params,AUX)
            local_L += lL
            local_∇L += l∇L
        end

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        #Add in diagonal part of the local derivative:
        local_∇L.+=AUX.local_∇L_diagonal_coeff.*AUX.Δ

        #Add in interaction terms:
        l_int = Lindblad_Ising_interaction_energy(sample, "periodic", A, params)
        local_L +=l_int
        local_∇L+=l_int*AUX.Δ

        L∂L+=local_L*conj(local_∇L)

        #ΔLL:
        ΔLL+=AUX.Δ

        #Mean local Lindbladian:
        mean_local_Lindbladian += local_L*conj(local_L)

        #Update metric tensor:
        S, Left = sample_update_SR(S, Left, params, AUX)
    end
    ΔLL.=conj.(ΔLL) #remember to take the complex conjugate

    return [L∂L, ΔLL, mean_local_Lindbladian, S, Left, acceptance]
end

function distributed_SR_MPO_gradient_two_body(A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, l2::Matrix{<:Complex{<:AbstractFloat}}, N_MC::Int64, ϵ::AbstractFloat, params::parameters)
    #output = [L∇L, ΔLL, mean_local_Lindbladian, S, Left, Right]

    #perform reduction:
    output = @distributed (+) for i=1:nworkers()
        #sample_with_SR_long_range(p, A, l1, N_MC, N_sweeps)
        #one_worker_SR_MPO_gradient(A, l1, convert(Int64,ceil(N_MC/nworkers())), ϵ, params)
        one_worker_SR_MPO_gradient_two_body(A, l1, l2, N_MC, ϵ, params)
    end

    L∂L=output[1]
    ΔLL=output[2]
    mean_local_Lindbladian=output[3]
    S=output[4]
    Left=output[5]
    acc=output[6]

    mean_local_Lindbladian/=(N_MC*nworkers())
    ΔLL*=mean_local_Lindbladian

    #Metric tensor:
    S./=(nworkers())
    Left./=(nworkers())

    grad = apply_SR(S,Left,N_MC,ϵ,L∂L,ΔLL,params)

    acc/=(N_MC*nworkers()*params.N)

    return grad, real(mean_local_Lindbladian), acc
end