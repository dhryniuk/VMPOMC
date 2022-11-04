using LinearAlgebra
include("ED_Ising.jl")
include("ED_Lindblad.jl")
using BenchmarkTools


J=1.0
h=1.0
γ=1.0
N=2
dim = 2^N

l1 = make_Liouvillian(h*sx,γ*sm)

display(l1)

basis=generate_bit_basis(N)
display(basis)

dINDEX = Dict((1,1) => 1, (1,0) => 2, (0,1) => 3, (0,0) => 4)
dVEC =   Dict((1,1) => [1,0,0,0], (1,0) => [0,1,0,0], (0,1) => [0,0,1,0], (0,0) => [0,0,0,1])
dUNVEC = Dict([1,0,0,0] => (1,1), [0,1,0,0] => (1,0), [0,0,1,0] => (0,1), [0,0,0,1] => (0,0))

TPSC = [(1,1),(1,0),(0,1),(0,0)]


function bra_L(bra,L)
    return transpose(bra)*L
end

mutable struct density_matrix
    coeff::ComplexF64
    ket::Vector{Int8}
    bra::Vector{Int8}
end
Base.copy(ρ::density_matrix) = density_matrix(ρ.coeff, ρ.ket, ρ.bra)

function MPO(sample, A)
    MPO=Matrix{ComplexF64}(I, χ, χ)
    for i::UInt8 in 1:N
        MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
    end
    #display(MPO)
    return tr(MPO)::ComplexF64
end

function MPO_inserted(sample, A, j, state)
    MPO=Matrix{ComplexF64}(I, χ, χ)
    for i::UInt8 in 1:N
        if i==j
            MPO*=A[:,:,dINDEX[state]]
        else
            MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        end
    end
    return tr(MPO)::ComplexF64
end

function MPO_Z(A)
    Z=0
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) 
            Z+=MPO(sample,A)*conj(MPO(sample,A))
        end
    end
    return Z
end

#   INTERACTIONS CORRECT!
function local_Lindbladian(J,h,γ,A) #should be called mean local lindbladian
    L_LOCAL=0
    Z = MPO_Z(A)

    #1-local part:
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) #replace by Monte Carlo
            ρ_sample = MPO(sample,A)
            p_sample = ρ_sample*conj(ρ_sample)

            local_L=0
            l_int = 0
            for j in 1:N
                l_int_α = 0
                l_int_β = 0

                #1-local part:
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*l1
                for i in 1:4
                    loc = bra_L[i]
                    state = TPSC[i]
                    local_L += loc*MPO_inserted(sample,A,j,state)
                end

                #2-local part: #PBC
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,N)+1]-1)
                l_int += -1.0im*J*(l_int_α-l_int_β)
            end

            local_L/=ρ_sample
            local_L+=l_int#*MPO(sample,A)
            L_LOCAL+=p_sample*local_L*conj(local_L)
        end
    end

    return L_LOCAL/Z
end

χ=3
A_init=rand(ComplexF64, χ,χ,4)
A=copy(A_init)

#println(local_Lindbladian(J,h,γ,A))

#error()

#A=zeros(ComplexF64, χ,χ,4)
#A[:, :, 1] .= 0.7
#A[:, :, 2] .= 0.2im
#A[:, :, 3] .= -0.1im
#A[:, :, 4] .= 0.3

#id = Matrix{Int}(I, χ, χ)
#A=zeros(ComplexF64, χ,χ,4)
#A[:, :, 1] .= id+0.1*rand(χ,χ)
#A[:, :, 2] .= 0.1*rand(χ,χ)
#A[:, :, 3] .= 0.1*rand(χ,χ)
#A[:, :, 4] .= id+0.1*rand(χ,χ)


println(local_Lindbladian(J,h,γ,A))
#error()


#GRADIENT:
function B_list(m, sample, A) #FIX m ORDERING
    B_list=Matrix{ComplexF64}[Matrix{Int}(I, χ, χ)]
    for j::UInt8 in 1:N-1
        push!(B_list,A[:,:,dINDEX[(sample.ket[mod(m+j-1,N)+1],sample.bra[mod(m+j-1,N)+1])]])
    end
    return B_list
end

function derv_MPO(i, j, u, sample, A)
    sum::ComplexF64 = 0
    for m::UInt8 in 1:N
        if u == (sample.ket[m],sample.bra[m])
            B = prod(B_list(m, sample, A))
            sum += B[i,j] + B[j,i]
        end
    end
    if i==j
        sum/=2
    end
    return sum
end

function Δ_MPO(i, j, u, sample, A)
    return derv_MPO(i, j, u, sample, A)/MPO(sample, A)
end

function calculate_gradient(J,h,γ,A,ii,jj,u)
    L∇L=0
    ΔLL=0
    Z = MPO_Z(A)
    mean_local_Lindbladian = local_Lindbladian(J,h,γ,A)

    #1-local part:
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) #replace by Monte Carlo
            ρ_sample = MPO(sample,A)
            p_sample = ρ_sample*conj(ρ_sample)

            local_L=0
            local_∇L=0

            l_int = 0

            #L∇L*:
            for j in 1:N
                l_int_α = 0
                l_int_β = 0

                #1-local part:
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*l1
                for i in 1:4
                    loc = bra_L[i]
                    state = TPSC[i]
                    local_L += loc*MPO_inserted(sample,A,j,state)
                    micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                    micro_sample.ket[j] = state[1]
                    micro_sample.bra[j] = state[2]
                    local_∇L+= loc*derv_MPO(ii,jj,u,micro_sample,A) 
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
            local_∇L+=l_int*Δ_MPO(ii,jj,u,sample,A)

            L∇L+=p_sample*local_L*conj(local_∇L)

            #ΔLL:
            local_Δ=p_sample*conj(Δ_MPO(ii,jj,u,sample,A))
            ΔLL+=local_Δ
        end
    end
    ΔLL*=mean_local_Lindbladian
    return (L∇L-ΔLL)/Z
end


function normalize_MPO(A)
    MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])^N
    #MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(1,0)]]+A[:,:,dINDEX[(0,1)]]+A[:,:,dINDEX[(0,0)]])^N
    return tr(MPO)^(1/N)#::ComplexF64
end


g = calculate_gradient(J,h,γ,A,1,1,(0,0))
display(g)
#error()


δχ = 0.01
Q=0.99
@time begin
    for k in 1:2500
        new_A=zeros(ComplexF64, χ,χ,4)
        for i in 1:χ
            for j in 1:χ
                for u in TPSC
                    #new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - (1+rand())*δχ*Q*sign.(calculate_gradient(J,h,γ,A,i,j,u))
                    new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - (1+rand())*δχ*Q*sign.(calculate_gradient(J,h,γ,A,i,j,u))
                    #new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - rand()*δχ*Q^k*sign.(calculate_gradient(J,h,γ,A,i,j,u))
                    #new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - δχ*Q^k*(calculate_gradient(J,h,γ,A,i,j,u))
                end
            end
        end
        global A = new_A
        global A./=normalize_MPO(A)
        global Q=local_Lindbladian(J,h,γ,A)
        #global Q=sqrt(local_Lindbladian(J,h,γ,A))
        println(local_Lindbladian(J,h,γ,A))
    end
end


function show_density_matrix(A)
    den_mat = zeros(ComplexF64, dim,dim)
    k=0
    for ket in basis
        k+=1
        b=0
        for bra in basis
            b+=1
            sample = density_matrix(1,ket,bra)
            ρ_sample = MPO(sample,A)
            den_mat[k,b] = ρ_sample#^2
        end
    end
    display(den_mat)
    display(eigen(den_mat))
end


show_density_matrix(A)