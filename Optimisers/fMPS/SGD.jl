export SGD_fMPS_gradient

function SGD_fMPS_gradient(params::parameters, A::Array{Float64}, V::Array{Float64}, h1::Matrix, N_MC::Int64)
    L∇L_bulk::Array{Float64,3}=zeros(Float64,params.χ,params.χ,2) #coupled product
    ΔLL_bulk::Array{Float64,3}=zeros(Float64,params.χ,params.χ,2) #uncoupled product
    L∇L_boundary::Array{Float64}=zeros(Float64,params.χ,2) #coupled product
    ΔLL_boundary::Array{Float64}=zeros(Float64,params.χ,2) #uncoupled product
    Z=0

    mean_local_Hamiltonian::Float64 = 0

    # Initialize sample and L_set for that sample:
    #sample, L_set = Metropolis_burn_in(params, A)
    acceptance::UInt64=0
    sample=rand(Bool,params.N)
    L_set = L_fMPS_strings(params, sample, A, V)

    for _ in 1:N_MC
        sample,R_set,acc=Mono_Metropolis_sweep_left(params,sample,A,V,L_set)
        #L_set = L_fMPS_strings(params, sample, A, V)
        #R_set = R_fMPS_strings(params, sample, A, V)
        #ρ_sample = L_set[params.N-1]*R_set[1]
        ρ_sample = transpose(V[:,dINDEX2[sample[1]]])*R_set[params.N-1]
        #ρ_sample = MPS(params, sample, A)
        #p_sample = ρ_sample*conj(ρ_sample)
        #Z+=p_sample

        local_E=0

        L_set = [ transpose(Vector{Float64}(undef,params.χ)) for _ in 1:params.N-1 ]

        e_field::Float64=0
        #L∇L*:
            #left boundary:
            #1-local part (field):
            e_field -= left_boundary_one_body_Hamiltonian_term(params,sample,h1,V,R_set)

            #Update L_set:
            L=transpose(V[:,dINDEX2[sample[1]]])
            L_set[1] = copy(L)

            #bulk:
            for j::UInt16 in 2:params.N-1

                #1-local part (field):
                e_field-=one_body_Hamiltonian_term(params,sample,j,h1,A,L_set,R_set)

                #Update L_set:
                L*=A[:,:,dINDEX2[sample[j]]]
                L_set[j] = copy(L)
            end

            #right boundary:
            e_field-=right_boundary_one_body_Hamiltonian_term(params,sample,h1,V,L_set)

            #Ising interaction:
            e_int = Ising_interaction_energy(params,sample,"open")

        e_field/=ρ_sample

        #Differentiate the fMPS:
        Δ_bulk_sample, Δ_boundary_sample = ∂fMPS(params, sample, L_set, R_set)
        Δ_bulk_sample./=ρ_sample
        Δ_boundary_sample./=ρ_sample

        #Add in interaction terms:
        local_E  = e_int+e_field
        L∇L_bulk += local_E*Δ_bulk_sample
        L∇L_boundary += local_E*Δ_boundary_sample

        #ΔLL:
        local_Δ_bulk = Δ_bulk_sample
        ΔLL_bulk += local_Δ_bulk
        local_Δ_boundary = Δ_boundary_sample
        ΔLL_boundary += local_Δ_boundary

        #Mean local Lindbladian:
        mean_local_Hamiltonian += local_E
    end

    mean_local_Hamiltonian/=N_MC
    ΔLL_bulk*=mean_local_Hamiltonian
    ΔLL_boundary*=mean_local_Hamiltonian

    return (L∇L_bulk-ΔLL_bulk)/N_MC, (L∇L_boundary-ΔLL_boundary)/N_MC, mean_local_Hamiltonian
end
