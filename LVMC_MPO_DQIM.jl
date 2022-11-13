using LinearAlgebra
include("ED_Ising.jl")
include("ED_Lindblad.jl")

const J=1.0 #interaction strength
const h=1.0 #transverse field strength
const γ=1.0 #spin decay rate
const N=4
const dim = 2^N

#Make single-body Lindbladian:
const l1 = make_Liouvillian(h*sx,γ*sm)
display(l1)

#Generate complete basis (not necessary when sampling via MCMC):
const basis=generate_bit_basis(N)
display(basis)

#Useful dictionaries:
dREVINDEX = Dict(1 => (1,1), 2 => (1,0), 3 => (0,1), 4 => (0,0))
dINDEX = Dict((1,1) => 1, (1,0) => 2, (0,1) => 3, (0,0) => 4)
dVEC =   Dict((1,1) => [1,0,0,0], (1,0) => [0,1,0,0], (0,1) => [0,0,1,0], (0,0) => [0,0,0,1])
dUNVEC = Dict([1,0,0,0] => (1,1), [0,1,0,0] => (1,0), [0,0,1,0] => (0,1), [0,0,0,1] => (0,0))
TPSC = [(1,1),(1,0),(0,1),(0,0)]


dINDEX2 = Dict(1 => 1, 0 => 2)


mutable struct density_matrix
    coeff::ComplexF64
    ket::Vector{Int8}
    bra::Vector{Int8}
end

function draw2(n)
    a = rand(1:n)
    b = rand(1:n)
    while b==a
        b = rand(1:n)
    end
    return a, b
end
function draw3(n)
    a = rand(1:n)
    b = rand(1:n)
    c = rand(1:n)
    while b==a
        b = rand(1:n)
    end
    while c==a && c==b
        c = rand(1:n)
    end
    return a, b, c
end

function L_MPO_strings(sample, A) # BEWARE OF NUMBERING
    L = Vector{Matrix{ComplexF64}}()
    MPO=Matrix{ComplexF64}(I, χ, χ)
    push!(L,copy(MPO))
    for i::UInt8 in 1:N
        MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        #MPO*=A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
        push!(L,copy(MPO))
    end
    return L
end

function R_MPO_strings(sample, A) # BEWARE OF NUMBERING
    R = Vector{Matrix{ComplexF64}}()
    MPO=Matrix{ComplexF64}(I, χ, χ)
    push!(R,copy(MPO))
    for i::UInt8 in N:-1:1
        MPO=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]*MPO

        # MATRIX MULTIPLICATION IS NOT COMMUTATIVE, IDIOT
        #MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        #MPO*=A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
        push!(R,copy(MPO))
    end
    return R
end

χ=6
A_init=rand(ComplexF64, χ,χ,2,2)
A=copy(A_init)
A=reshape(A,χ,χ,4)


#A=zeros(ComplexF64, χ,χ,4)
#A[:, :, 1] .= 0.97
#A[:, :, 2] .= 0.01
#A[:, :, 3] .= 0.01
#A[:, :, 4] .= 0.01



sample = density_matrix(1,basis[1],basis[1])




function MPO(sample, A)
    MPO=Matrix{ComplexF64}(I, χ, χ)
    for i::UInt8 in 1:N
        MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
    end
    return tr(MPO)::ComplexF64
end

function MPO_string(sampe,A,l,r)
    MPO=Matrix{ComplexF64}(I, χ, χ)
    for i in l:r
        MPO*=A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
    end
    return MPO
end

#Sweep lattice from right to left:
function Dual_Metropolis_sweep(sample, A, L_set)
    R_set = []
    R = Matrix{ComplexF64}(I, χ, χ)
    push!(R_set, copy(R))
    C = tr(L_set[N+1]) #Current MPO  ---> move into loop
    for i in N:-1:1

        #Update ket:
        sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra)) #deepcopy necessary?
        sample_p.ket[i] = 1-sample.ket[i]
        #P = tr(L_set[i]*A[:,:,dINDEX2[sample_p.ket[i]],dINDEX2[sample.bra[i]]])
        P = tr(L_set[i]*A[:,:,dINDEX[(sample_p.ket[i],sample.bra[i])]]*R_set[N+1-i])
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = sample_p
        end
        #aux_R = R*A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]] #auxiliary R
        aux_R = R*A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        C = tr(L_set[i]*aux_R)

        #Update bra:
        sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
        sample_p.bra[i] = 1-sample.bra[i]
        #P = tr(L_set[i]*A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample_p.bra[i]]])
        P = tr(L_set[i]*A[:,:,dINDEX[(sample.ket[i],sample_p.bra[i])]]*R_set[N+1-i])
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = sample_p
        end
        #R *= A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
        R *= A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        push!(R_set, copy(R))
        C = tr(L_set[i]*R)

    end
    return sample, R_set
end

function Mono_Metropolis_sweep_left(sample, A, L_set)

    function draw_excluded(u)
        v = rand(1:3)
        if v>=u
            v+=1
        end
        return v
    end

    R_set = []
    R = Matrix{ComplexF64}(I, χ, χ)
    push!(R_set, copy(R))
    C = tr(L_set[N+1]) #Current MPO  ---> move into loop
    for i in N:-1:1

        sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra)) #deepcopy necessary?
        u = dINDEX[(sample.ket[i],sample.bra[i])]
        v = draw_excluded(u)
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[v]
        P = tr(L_set[i]*A[:,:,v]*R_set[N+1-i])
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            #sample = sample_p
            sample.ket = deepcopy(sample_p.ket)
            sample.bra = deepcopy(sample_p.bra)
        end

        R = A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]*R
        push!(R_set, copy(R))
        C = tr(L_set[i]*R)
    
    #    C = tr(L_set[i]*R_set[N+2-i])
    end
    return sample, R_set
end

function Mono_Metropolis_sweep_right(sample, A, R_set)

    function draw_excluded(u)
        v = rand(1:3)
        if v>=u
            v+=1
        end
        return v
    end

    L_set = []
    L = Matrix{ComplexF64}(I, χ, χ)
    push!(L_set, copy(L))
    C = tr(R_set[N+1]) #Current MPO  ---> move into loop
    for i in 1:N

        sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra)) #deepcopy necessary?
        u = dINDEX[(sample.ket[i],sample.bra[i])]
        v = draw_excluded(u)
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[v]
        P = tr(L_set[i]*A[:,:,v]*R_set[N+1-i])
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = sample_p
        end

        L*= A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        push!(L_set, copy(L))
        C = tr(L*R_set[N+1-i])

    end
    return sample, L_set
end

function Single_Metropolis(sample, A, i)

    function draw_excluded(u)
        v = rand(1:3)
        if v>=u
            v+=1
        end
        return v
    end

    C = MPO(sample, A)
    #i=rand(1:N)

    sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra)) #deepcopy necessary?
    u = dINDEX[(sample.ket[i],sample.bra[i])]
    v = draw_excluded(u)
    (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[v]
    P = MPO(sample_p, A)
    metropolis_prob = real((P*conj(P))/(C*conj(C)))
    if rand() <= metropolis_prob
        sample = sample_p
    end

    return sample
end

function calculate_mean_local_Lindbladian(J,A)
    mll=0
    Z=0
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) 
            ρ_sample = MPO(sample,A)
            p_sample = ρ_sample*conj(ρ_sample)
            Z+=p_sample

            L_set = L_MPO_strings(sample,A)
            R_set = R_MPO_strings(sample,A)

            local_L = local_Lindbladian(J,A,sample,L_set,R_set)

            mll+=p_sample*local_L*conj(local_L)
        end
    end

    return mll/Z
end



function local_Lindbladian(J,A,sample,L_set,R_set)

    local_L=0
    l_int = 0
    for j in 1:N

        #1-local part:
        s = dVEC[(sample.ket[j],sample.bra[j])]
        bra_L = transpose(s)*l1
        for i in 1:4
            loc = bra_L[i]
            if loc!=0
                state = TPSC[i]
                local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[N+1-j]) #add if condition for loc=0
            end
        end

        #2-local part: #PBC
        l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,N)+1]-1)
        l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,N)+1]-1)
        l_int += -1.0im*J*(l_int_α-l_int_β)

    end

    local_L/=tr(R_set[N+1])
    local_L+=l_int#*MPO(sample,A)

    return local_L
end

function MC_mean_local_Lindbladian(J,A,N_MC)
    mll=0
    sample = density_matrix(1,basis[1],basis[1]) 
    L_set = L_MPO_strings(sample,A)
    R_set = R_MPO_strings(sample,A)
    for k in 1:N_MC
        sample, R_set = Mono_Metropolis_sweep_left(sample, A, L_set)
        #sample, L2_set = Mono_Metropolis_sweep_right(sample, A, R_set)
        #sample = Single_Metropolis(sample, A, rand(1:N))
        #for i in 1:N
        #    sample = Single_Metropolis(sample, A, i)
        #end
        L_set = L_MPO_strings(sample,A)
        #R_set = R_MPO_strings(sample,A)
        #println(R2_set==R_set)

        local_L = local_Lindbladian(J,A,sample,L_set,R_set)

        mll+=local_L*conj(local_L)
    end

    return mll/N_MC
end



#ex = calculate_mean_local_Lindbladian(J,A)
#mc = MC_mean_local_Lindbladian(J,A,10000)

#display(ex)
#display(mc)

#error()




#GRADIENT:
function B_list(m, sample, A) #FIX m ORDERING
    B_list=Matrix{ComplexF64}[Matrix{Int}(I, χ, χ)]
    for j::UInt8 in 1:N-1
        push!(B_list,A[:,:,dINDEX[(sample.ket[mod(m+j-1,N)+1],sample.bra[mod(m+j-1,N)+1])]])
    end
    return B_list
end

function derv_MPO(sample, A, L_set, R_set)
    ∇=zeros(ComplexF64, χ,χ,4)
    #L_set = L_MPO_strings(sample, A)
    #R_set = R_MPO_strings(sample, A)
    for m::UInt8 in 1:N
        B = R_set[N+1-m]*L_set[m]
        for i in 1:χ
            for j in 1:χ
                ∇[i,j,dINDEX[(sample.ket[m],sample.bra[m])]] += B[i,j] + B[j,i]
            end
            ∇[i,i,:]./=2
        end
    end
    return ∇
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

sample = density_matrix(1,basis[1],basis[2])


function Δ_MPO(sample, A)
    return derv_MPO(sample, A)/MPO(sample, A)
end

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
    #L∇L=zeros(ComplexF64,χ,χ,4)
    #ΔLL=zeros(ComplexF64,χ,χ,4)
    L∇L=Array{ComplexF64}(undef,χ,χ,4)
    ΔLL=Array{ComplexF64}(undef,χ,χ,4)
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
    
            Δ_MPO_sample = OLDderv_MPO(sample,A)/ρ_sample
    
            #Add in interaction terms:
            local_L +=l_int#*MPO(sample, A)
            local_∇L+=l_int*Δ_MPO_sample
    
            L∇L+=p_sample*local_L*conj(local_∇L)
    
            #ΔLL:
            local_Δ=p_sample*conj(Δ_MPO_sample)
            ΔLL+=local_Δ
    
            #Mean local Lindbladian:
            mean_local_Lindbladian += p_sample*local_L*conj(local_L)
        end
    end
    ΔLL*=mean_local_Lindbladian/Z
    return (L∇L-ΔLL)/Z, mean_local_Lindbladian/Z
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
    #L∇L=Array{ComplexF64}(undef,χ,χ,4)
    #ΔLL=Array{ComplexF64}(undef,χ,χ,4)


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
    return (L∇L-ΔLL)/N_MC, mean_local_Lindbladian
end

function normalize_MPO(A)
    MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])^N
    return tr(MPO)^(1/N)#::ComplexF64
end


#∇,L=calculate_gradient(J,A)
#display(∇)
#∇,L=calculate_MC_gradient_full(J,A,100000)
#display(∇)

#error()

A./=normalize_MPO(A)
N_MC=100

δ = 0.01
Q=0.92
@time begin
    for k in 1:1000
        new_A=zeros(ComplexF64, χ,χ,4)
        #new_A = A - (1+rand())*δ*Q*sign.(calculate_MC_gradient(J,A))
        #∇,L=calculate_gradient(J,A)
        ∇,L=calculate_MC_gradient_full(J,A,N_MC+5*k)
        ∇./=maximum(abs.(∇))
        new_A = A - δ*Q*(sign.(∇).+0.01*rand())
        #new_A = A - δ*Q*∇.*(1+0.01*rand())

        global A = new_A
        global A./=normalize_MPO(A)
        Lex=calculate_mean_local_Lindbladian(J,A)
        #global Q=sqrt(calculate_mean_local_Lindbladian(J,A))
        global Q=sqrt(L)
        println("k=$k: ", real(L), " ; ", real(Lex))
    end
end

converged_A=A


function double_bd(A)
    global χ*=2
    B = zeros(ComplexF64, χ,χ,4)
    for i in 1:4
        B[:,:,i] = kron(A[:,:,i],[1 1; 1 1])
    end
    return B
end