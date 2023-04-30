export Exact_MPO_gradient#, one_body_Lindblad_term

function one_body_Lindblad_term(params::parameters, sample::density_matrix, j::UInt16, l1::Matrix, A::Array{ComplexF64,3}, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})
    local_L::ComplexF64 = 0
    local_∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)

    s::Matrix{ComplexF64} = dVEC_transpose[(sample.ket[j],sample.bra[j])]
    #bra_L::Transpose{ComplexF64, Vector{ComplexF64}} = transpose(s)*conj(l1)
    bra_L::Matrix{ComplexF64} = s*conj(l1)
    #bra_L::Matrix{ComplexF64} = s*l1
    #display(typeof(bra_L))
    #error()
    #for i in 1:4
    #left = L_set[j]
    #right = R_set[params.N+1-j]
    @inbounds for (i::UInt16,state) in zip(1:4,TPSC)
        loc::ComplexF64 = bra_L[i]
        if loc!=0
            #state = TPSC[i]
            #local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[params.N+1-j])
            local_L += loc*tr((L_set[j::UInt16]*A[:,:,i]::Matrix{ComplexF64})*R_set[(params.N+1-j)::Int64])
            #local_L += loc*tr(left*A[:,:,i]*right)
            micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
            micro_sample.ket[j] = state[1]
            micro_sample.bra[j] = state[2]
            
            micro_L_set = L_MPO_strings(params, micro_sample, A)
            micro_R_set = R_MPO_strings(params, micro_sample, A)
            local_∇L+= loc*∂MPO(params, micro_sample,micro_L_set,micro_R_set)
        end
    end
    return local_L, local_∇L
end

function one_body_Lindblad_term_without_preallocation(AUX::workspace, params::parameters, sample::density_matrix, j::UInt8, l1::Matrix, A::Array{ComplexF64,3}, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})
    local_L::ComplexF64 = 0
    local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)

    #micro_L_set::Vector{Matrix{ComplexF64}} = [ Matrix{ComplexF64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    #micro_R_set::Vector{Matrix{ComplexF64}} = [ Matrix{ComplexF64}(undef,params.χ,params.χ) for _ in 1:params.N+1 ]
    #try replacing L_array wiht micro_L_set

    #B::Matrix{ComplexF64} = zeros(params.χ,params.χ)

    s::Matrix{ComplexF64} = dVEC_transpose[(sample.ket[j],sample.bra[j])]
    #bra_L::Matrix{ComplexF64} = s*conj(l1)
    mul!(AUX.bra_L, s, conj.(l1))
    @inbounds for (i::UInt16,state::Tuple{Bool,Bool}) in zip(1:4,TPSC)
        loc::ComplexF64 = AUX.bra_L[i]
        if loc!=0
            mul!(AUX.loc_1, L_set[j], @view(A[:,:,i]))
            mul!(AUX.loc_2, AUX.loc_1, R_set[(params.N+1-j)])
            local_L += loc.*tr(AUX.loc_2)
            #local_L += loc.*tr((L_set[j::UInt16]*A[:,:,i]::Matrix{ComplexF64})*R_set[(params.N+1-j)::Int64])
            #micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
            micro_sample = density_matrix(1,copy(sample.ket),copy(sample.bra))
            micro_sample.ket[j] = state[1]
            micro_sample.bra[j] = state[2]
            
            micro_L_set = L_MPO_strings_without_preallocation(AUX, params, micro_sample, A)
            micro_R_set = R_MPO_strings_without_preallocation(AUX, params, micro_sample, A)
            #local_∇L+= loc.*∂MPO(params, micro_sample,micro_L_set,micro_R_set)
            local_∇L.+= loc.*∂MPO_without_preallocation(AUX,params, micro_sample,micro_L_set,micro_R_set)
        end
    end
    return local_L, local_∇L
end

function partial_one_body_Lindblad_term(params::parameters, sample::density_matrix, j::UInt16, l1::Matrix, A::Array{ComplexF64,3}, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})
    local_L::ComplexF64 = 0
    local_∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)

    s::Matrix{ComplexF64} = dVEC_transpose[(sample.ket[j],sample.bra[j])]
    #bra_L::Transpose{ComplexF64, Vector{ComplexF64}} = transpose(s)*conj(l1)
    bra_L::Matrix{ComplexF64} = s*conj(l1)
    #bra_L::Matrix{ComplexF64} = s*l1
    #display(typeof(bra_L))
    #error()
    #for i in 1:4
    #left = L_set[j]
    #right = R_set[params.N+1-j]
    @inbounds for (i::UInt16,state) in zip(1:4,TPSC)
        if rand()<=0.8
            loc::ComplexF64 = bra_L[i]
            if loc!=0
                #state = TPSC[i]
                #local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[params.N+1-j])
                local_L += loc*tr((L_set[j::UInt16]*A[:,:,i]::Matrix{ComplexF64})*R_set[(params.N+1-j)::Int64])
                #local_L += loc*tr(left*A[:,:,i]*right)
                micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                micro_sample.ket[j] = state[1]
                micro_sample.bra[j] = state[2]
                
                micro_L_set = L_MPO_strings(params, micro_sample, A)
                micro_R_set = R_MPO_strings(params, micro_sample, A)
                local_∇L+= loc*∂MPO(params, micro_sample,micro_L_set,micro_R_set)
            end
        end
    end
    return local_L, local_∇L
end

function Lindblad_Ising_interaction_energy(params::parameters, sample::density_matrix, boundary_conditions)
    l_int::ComplexF64=0
    for j::UInt16 in 1:params.N-1
        l_int_ket = (2*sample.ket[j]-1)*(2*sample.ket[j+1]-1)
        l_int_bra = (2*sample.bra[j]-1)*(2*sample.bra[j+1]-1)
        #l_int += -1.0im*J*(l_int_α-l_int_β)
        l_int += l_int_ket-l_int_bra
    end
    if boundary_conditions=="periodic"
        l_int_ket = (2*sample.ket[params.N]-1)*(2*sample.ket[1]-1)
        l_int_bra = (2*sample.bra[params.N]-1)*(2*sample.bra[1]-1)
        #l_int += -1.0im*J*(l_int_α-l_int_β)
        l_int += l_int_ket-l_int_bra
    end
    return 1.0im*params.J*l_int::ComplexF64
end

function Lindblad_Ising_interaction_energy(params::parameters, sample_ket::Array{Bool}, sample_bra::Array{Bool}, boundary_conditions)
    l_int::ComplexF64=0
    for j::UInt16 in 1:params.N-1
        l_int_ket = (2*sample_ket[j]-1)*(2*sample_ket[j+1]-1)
        l_int_bra = (2*sample_bra[j]-1)*(2*sample_bra[j+1]-1)
        #l_int += -1.0im*J*(l_int_α-l_int_β)
        l_int += l_int_ket-l_int_bra
    end
    if boundary_conditions=="periodic"
        l_int_ket = (2*sample_ket[params.N]-1)*(2*sample_ket[1]-1)
        l_int_bra = (2*sample_bra[params.N]-1)*(2*sample_bra[1]-1)
        #l_int += -1.0im*J*(l_int_α-l_int_β)
        l_int += l_int_ket-l_int_bra
    end
    return 1.0im*params.J*l_int::ComplexF64
end

function N4_Lindblad_Ising_interaction_energy_2D(params::parameters, sample::density_matrix)

    if params.N==4
        sqrt_N = convert(Int64,sqrt(params.N)) #==##2!
        l_int::ComplexF64=0
        for k in 0:sqrt_N-1
            l_int_ket = (2*sample.ket[1+k*sqrt_N]-1) * (2*sample.ket[sqrt_N+k*sqrt_N]-1)
            l_int_bra = (2*sample.bra[1+k*sqrt_N]-1) * (2*sample.bra[sqrt_N+k*sqrt_N]-1)
            l_int += l_int_ket-l_int_bra
        end
        for j in 1:sqrt_N
            l_int_ket = (2*sample.ket[j+0]-1) * (2*sample.ket[j+sqrt_N]-1)
            l_int_bra = (2*sample.bra[j+0]-1) * (2*sample.bra[j+sqrt_N]-1)
            l_int += l_int_ket-l_int_bra
        end
        return 1.0im*params.J*l_int::ComplexF64
    end
end

function Lindblad_Ising_interaction_energy_2D(params::parameters, sample::density_matrix)

    if params.N>=4
        sqrt_N = convert(Int64,sqrt(params.N))
        l_int::ComplexF64=0
        for k in 0:sqrt_N-1
            for j in 1:sqrt_N-1
                #println(k, " ; ", j)
                #println(j+k*params.N)
                #println(j+1+k*params.N)
                l_int_ket = (2*sample.ket[j+k*sqrt_N]-1) * (2*sample.ket[j+1+k*sqrt_N]-1)
                l_int_bra = (2*sample.bra[j+k*sqrt_N]-1) * (2*sample.bra[j+1+k*sqrt_N]-1)
                l_int += l_int_ket-l_int_bra
            end
            l_int_ket = (2*sample.ket[sqrt_N+k*sqrt_N]-1) * (2*sample.ket[1+k*sqrt_N]-1)
            l_int_bra = (2*sample.bra[sqrt_N+k*sqrt_N]-1) * (2*sample.bra[1+k*sqrt_N]-1)
            l_int += l_int_ket-l_int_bra
        end
        for j in 1:sqrt_N-1
            for k in 0:sqrt_N-2
                l_int_ket = (2*sample.ket[j+k*sqrt_N]-1) * (2*sample.ket[j+(k+1)*sqrt_N]-1)
                l_int_bra = (2*sample.bra[j+k*sqrt_N]-1) * (2*sample.bra[j+(k+1)*sqrt_N]-1)
                l_int += l_int_ket-l_int_bra
            end
            l_int_ket = (2*sample.ket[j+sqrt_N*(sqrt_N-1)]-1) * (2*sample.ket[j+0]-1)
            l_int_bra = (2*sample.bra[j+sqrt_N*(sqrt_N-1)]-1) * (2*sample.bra[j+0]-1)
            l_int += l_int_ket-l_int_bra
        end
        return 1.0im*params.J*l_int::ComplexF64
    end
end

function two_body_Lindblad_term(params::parameters, sample::density_matrix, k::UInt16, l2::Matrix, A::Array{ComplexF64,3}, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})
    local_L::ComplexF64 = 0
    local_∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)

    s1::Matrix{ComplexF64} = dVEC_transpose[(sample.ket[k],sample.bra[k])]
    s2::Matrix{ComplexF64} = dVEC_transpose[(sample.ket[k+1],sample.bra[k+1])]
    s = kron(s1,s2)
    bra_L::Matrix{ComplexF64} = s*conj(l2)
    #@inbounds for i::UInt16 in 1:4, j::UInt16 in 1:4
    for (i::UInt16,state_i::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})
        for (j::UInt16,state_j::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})

            loc::ComplexF64 = bra_L[j+4*(i-1)]
            if loc!=0
                local_L += loc*tr( (L_set[k::UInt16]*A[:,:,i]::Matrix{ComplexF64})*A[:,:,j]::Matrix{ComplexF64}*R_set[(params.N-k)::Int64])
                micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                micro_sample.ket[k] = state_i[1]
                micro_sample.bra[k] = state_i[2]
                micro_sample.ket[k+1] = state_j[1]
                micro_sample.bra[k+1] = state_j[2]
                
                micro_L_set = L_MPO_strings(params, micro_sample, A)
                micro_R_set = R_MPO_strings(params, micro_sample, A)
                local_∇L+= loc*∂MPO(params, micro_sample,micro_L_set,micro_R_set)
            end
        end
    end
    return local_L, local_∇L
end

function boundary_two_body_Lindblad_term(params::parameters, sample::density_matrix, l2::Matrix, A::Array{ComplexF64,3}, L_set::Vector{Matrix{ComplexF64}}, R_set::Vector{Matrix{ComplexF64}})

    #Need to find middle string, by inverting the first tensor A:
    M = inv(A[:,:,dINDEX[(sample.ket[1],sample.bra[1])]])*L_set[params.N]

    local_L::ComplexF64 = 0
    local_∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)

    s1::Matrix{ComplexF64} = dVEC_transpose[(sample.ket[params.N],sample.bra[params.N])]
    s2::Matrix{ComplexF64} = dVEC_transpose[(sample.ket[1],sample.bra[1])]
    s = kron(s1,s2)
    bra_L::Matrix{ComplexF64} = s*conj(l2)
    #@inbounds for i::UInt16 in 1:4, j::UInt16 in 1:4
    for (i::UInt16,state_i::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})
        for (j::UInt16,state_j::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})

            loc::ComplexF64 = bra_L[j+4*(i-1)]
            if loc!=0
                local_L += loc*tr( M*A[:,:,i]*A[:,:,j] )
                #local_L += loc*MPO(params,sample,A)
                micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                micro_sample.ket[1] = state_i[1]
                micro_sample.bra[1] = state_i[2]
                micro_sample.ket[params.N] = state_j[1]
                micro_sample.bra[params.N] = state_j[2]
                
                micro_L_set = L_MPO_strings(params, micro_sample, A)
                micro_R_set = R_MPO_strings(params, micro_sample, A)
                local_∇L+= loc*∂MPO(params, micro_sample, micro_L_set, micro_R_set)
            end
        end
    end
    return local_L, local_∇L
end


function Exact_MPO_gradient(params::parameters, A::Array{ComplexF64,3}, l1::Matrix{ComplexF64}, basis)
    L∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    Z::ComplexF64 = 0

    mean_local_Lindbladian::ComplexF64 = 0

    for k in 1:params.dim
        #sample_ket = basis[k]
        for l in 1:params.dim
            sample = density_matrix(1,basis[k],basis[l])
            L_set = L_MPO_strings(params, sample, A)
            R_set = R_MPO_strings(params, sample, A)
            #sample_bra = basis[l]
            #L_set = L_MPO_strings(params, sample_ket, sample_bra, A)
            #R_set = R_MPO_strings(params, sample_ket, sample_bra, A)

            ρ_sample = tr(L_set[params.N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z += p_sample

            local_L::ComplexF64 = 0
            local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
            l_int::ComplexF64 = 0

            #L_set = Vector{Matrix{ComplexF64}}()
            #L::Matrix{ComplexF64} = Matrix{ComplexF64}(I, params.χ, params.χ)
            #push!(L_set,copy(L))

            L_set = [ Matrix{ComplexF64}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
            L = Matrix{ComplexF64}(I, params.χ, params.χ)
            L_set[1] = L

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
                #push!(L_set,copy(L))
                L_set[j+1] = L
            end

            l_int = Lindblad_Ising_interaction_energy(params, sample, "periodic")
            #l_int = Lindblad_Ising_interaction_energy(params, sample_ket, sample_bra, "periodic")

            #l_int = N4_Lindblad_Ising_interaction_energy_2D(params, sample)

            #println("2d: ", l_int)
            #println("1d: ", Lindblad_Ising_interaction_energy(params, sample, "periodic"))

            local_L /=ρ_sample
            local_∇L/=ρ_sample
    
            Δ_MPO_sample = ∂MPO(params, sample, L_set, R_set)/ρ_sample
            #Δ_MPO_sample = derv_MPO(params, sample_ket, sample_bra, L_set, R_set)/ρ_sample
    
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

export Two_body_Exact_MPO_gradient

function Two_body_Exact_MPO_gradient(params::parameters, A::Array{ComplexF64}, 
    l1::Matrix{ComplexF64}, l2::Matrix{ComplexF64}, basis)

    L∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    Z::ComplexF64 = 0
    mean_local_Lindbladian::ComplexF64 = 0

    for k in 1:params.dim
        for l in 1:params.dim
            sample = density_matrix(1,basis[k],basis[l])
            L_set = L_MPO_strings(params, sample, A)
            R_set = R_MPO_strings(params, sample, A)

            ρ_sample = tr(L_set[params.N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z += p_sample

            local_L::ComplexF64 = 0
            local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
            l_int::ComplexF64 = 0

            L_set = [ Matrix{ComplexF64}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
            L = Matrix{ComplexF64}(I, params.χ, params.χ)
            L_set[1] = L

            #L∇L*:
            for j::UInt16 in 1:params.N

                #1-local part:
                lL, l∇L = one_body_Lindblad_term(params,sample,j,l1,A,L_set,R_set)
                #lL, l∇L = one_body_Lindblad_term(params,sample_ket,sample_bra,j,l1,A,L_set,R_set)
                local_L += lL
                local_∇L += l∇L

                L*=A[:,:,1+2*sample.ket[j]+sample.bra[j]]
                L_set[j+1] = L
            end
            for j::UInt16 in 1:params.N-1
                lL, l∇L = two_body_Lindblad_term(params,sample,j,l2,A,L_set,R_set)
                local_L += lL
                local_∇L += l∇L
            end
            if params.N>2
                lL, l∇L = boundary_two_body_Lindblad_term(params,sample,l2,A,L_set,R_set)
                local_L += lL
                local_∇L += l∇L
            end

            #l_int = Lindblad_Ising_interaction_energy(params, sample, "periodic")

            local_L /=ρ_sample
            local_∇L/=ρ_sample
    
            Δ_MPO_sample = ∂MPO(params, sample, L_set, R_set)/ρ_sample
            #Δ_MPO_sample = derv_MPO(params, sample_ket, sample_bra, L_set, R_set)/ρ_sample
    
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



export NAGD
function NAGD(δ, params::parameters, A_new::Array{ComplexF64}, A_old::Array{ComplexF64}, l1::Matrix{ComplexF64}, basis)
    display(A_new)
    A = A_new+0.9*(A_new-A_old)

    new_A=zeros(ComplexF64, χ,χ,4)
    ∇,L=Exact_MPO_gradient(params,A,l1,basis)
    ∇./=maximum(abs.(∇))
    new_A = A - δ*∇#.*(1+0.5*rand())

    A_old = A_new
    global A_new = new_A
    global A_new = normalize_MPO(params, A)
    return A_new, A_old
end

#NAGD version:

function NAGD_Exact_MPO_gradient(params::parameters, A_new::Array{ComplexF64}, A_old::Array{ComplexF64}, l1::Matrix{ComplexF64}, basis)
    L∇L::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    ΔLL::Array{ComplexF64,3}=zeros(ComplexF64,params.χ,params.χ,4)
    Z::ComplexF64 = 0

    mean_local_Lindbladian::ComplexF64 = 0

    for k in 1:params.dim
        for l in 1:params.dim
            sample = density_matrix(1,basis[k],basis[l])
            L_set = L_MPO_strings(params, sample, A)
            R_set = R_MPO_strings(params, sample, A)

            ρ_sample = tr(L_set[params.N+1])
            p_sample = ρ_sample*conj(ρ_sample)
            Z += p_sample

            local_L::ComplexF64 = 0
            local_∇L::Array{ComplexF64,3} = zeros(ComplexF64,params.χ,params.χ,4)
            l_int::ComplexF64 = 0

            L_set = [ Matrix{ComplexF64}(undef, params.χ, params.χ) for _ in 1:params.N+1 ]
            L = Matrix{ComplexF64}(I, params.χ, params.χ)
            L_set[1] = L

            #L∇L*:
            for j::UInt16 in 1:params.N

                #1-local part:
                lL, l∇L = one_body_Lindblad_term(params,sample,j,l1,A,L_set,R_set)
                local_L += lL
                local_∇L += l∇L

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