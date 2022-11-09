using LinearAlgebra
include("ED_Ising.jl")
include("ED_Lindblad.jl")

const J=1.0 #interaction strength
const h=1.0 #transverse field strength
const γ=1.0 #spin decay rate
const N=3
const dim = 2^N

#Make single-body Lindbladian:
const l1 = make_Liouvillian(h*sx,γ*sm)
display(l1)

#Generate complete basis (not necessary when sampling via MCMC):
const basis=generate_bit_basis(N)
display(basis)

#Useful dictionaries:
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
        MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        #MPO*=A[:,:,dINDEX2[sample.ket[i]],dINDEX2[sample.bra[i]]]
        push!(R,copy(MPO))
    end
    return R
end

χ=4
A_init=rand(ComplexF64, χ,χ,2,2)
A=copy(A_init)
A=reshape(A,χ,χ,4)

#display(A_init)
#display(A)



sample = density_matrix(1,basis[1],basis[1])

L_set=L_MPO_strings(sample,A)
display(L_set)


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
function Metropolis_sweep(sample, A, L_set)
    R_set = []
    R = Matrix{ComplexF64}(I, χ, χ)
    push!(R_set, copy(R))
    C = tr(L_set[N+1]) #Current MPO  ---> move into loop
    for i in N:-1:1

        #Update ket:
        sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra)) #deepcopy necessary?
        sample_p.ket[i] = 1-sample.ket[i]
        #P = tr(L_set[i]*A[:,:,dINDEX2[sample_p.ket[i]],dINDEX2[sample.bra[i]]])
        P = tr(L_set[i]*A[:,:,dINDEX[(sample_p.ket[i],sample.bra[i])]])
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
        P = tr(L_set[i]*A[:,:,dINDEX[(sample.ket[i],sample_p.bra[i])]])
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

new_sample, R_set = Metropolis_sweep(sample, A, L_set)






function local_Lindbladian(J,A,sample,L_set,R_set)
    #L_set = L_MPO_strings(sample, A)

    local_L=0
    l_int = 0
    for j in 1:N

        #1-local part:
        s = dVEC[(sample.ket[j],sample.bra[j])]
        bra_L = transpose(s)*l1
        for i in 1:4
            loc = bra_L[i]
            state = TPSC[i]
            local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[N+1-j]) #add if condition for loc=0
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











#GRADIENT:
function B_list(m, sample, A) #FIX m ORDERING
    B_list=Matrix{ComplexF64}[Matrix{Int}(I, χ, χ)]
    for j::UInt8 in 1:N-1
        push!(B_list,A[:,:,dINDEX[(sample.ket[mod(m+j-1,N)+1],sample.bra[mod(m+j-1,N)+1])]])
    end
    return B_list
end

function derv_MPO(sample, A)
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

function Δ_MPO(sample, A)
    return derv_MPO(sample, A)/MPO(sample, A)
end

function calculate_gradient(J,A)
    L∇L=zeros(ComplexF64,χ,χ,4)
    ΔLL=zeros(ComplexF64,χ,χ,4)
    Z=0

    GRADIENT = zeros(ComplexF64,χ,χ,4)

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
                    state = TPSC[i]
                    local_L += loc*tr(L_set[j]*A[:,:,dINDEX[(state[1],state[2])]]*R_set[N+1-j]) #add if condition for loc=0
                    micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                    micro_sample.ket[j] = state[1]
                    micro_sample.bra[j] = state[2]

                    local_∇L+= loc*derv_MPO(micro_sample,A)
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
    return (L∇L-ΔLL)/Z
end

function normalize_MPO(A)
    MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])^N
    return tr(MPO)^(1/N)#::ComplexF64
end


A=copy(A_init)
A=reshape(A,χ,χ,4)
δ = 0.01
Q=0.99
@time begin
    for k in 1:500
        new_A=zeros(ComplexF64, χ,χ,4)
        new_A = A - (1+rand())*δ*Q*sign.(calculate_gradient(J,A))
        #for i in 1:χ
        #    for j in 1:χ
        #        for u in TPSC
        #            new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - (1+rand())*δ*Q*sign.(calculate_gradient(J,A,i,j,u))
        #        end
        #    end
        #end
        global A = new_A
        global A./=normalize_MPO(A)
        global Q=calculate_mean_local_Lindbladian(J,A)
        #global Q=sqrt(calculate_mean_local_Lindbladian(J,A))
        println(calculate_mean_local_Lindbladian(J,A))
    end
end

