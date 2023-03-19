export SGD_MPO_gradient


function SGD_MPO_gradient(params::parameters, A::Array{ComplexF64,3}, l1::Matrix{ComplexF64}, N_MC::Int64)#, N_sweeps::Int64)
    L∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)

    mean_local_Lindbladian::ComplexF64 = 0

    # Initialize sample and L_set for that sample:
    sample, L_set = MPO_Metropolis_burn_in(params, A)
    acceptance::UInt64=0

    for _ in 1:N_MC

        sample, R_set = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        #for n in N_sweeps
        #    sample, L_set = Mono_Metropolis_sweep_right(params, sample, A, R_set)
        #    sample, R_set = Mono_Metropolis_sweep_left(params, sample, A, L_set)
        #end
        ρ_sample = tr(R_set[params.N+1])
        L_set = [ Matrix{ComplexF64}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
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

            #2-local part:
            #l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,params.N)+1]-1)
            #l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,params.N)+1]-1)
            #l_int += -1.0im*J*(l_int_α-l_int_β)
            #l_int += 1.0im*params.J*(l_int_α-l_int_β)

            #Update L_set:
            #L*=A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]
            L*=A[:,:,1+2*sample.ket[j]+sample.bra[j]]
            L_set[j+1] = L
        end

        l_int = Lindblad_Ising_interaction_energy(params, sample, "periodic")

        local_L /=ρ_sample
        local_∇L/=ρ_sample

        Δ_MPO_sample = ∂MPO(params, sample, L_set, R_set)/ρ_sample

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
    return (L∇L-ΔLL)/N_MC, real(mean_local_Lindbladian)
end