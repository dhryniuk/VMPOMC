export eigen_sparse, set_parameters, calculate_Kac_norm
export ⊗, id, sx, sy, sz, sp, sm



function eigen_sparse(x)
    decomp, history = partialschur(x, nev=1, which=LR(); restarts=100000); # only solve for the ground state
    vals, vecs = partialeigen(decomp);
    return vals, vecs
end

⊗(x,y) = kron(x,y)

id = [1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im]
sx = [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im]
sy = [0.0+0.0im 0.0-1im; 0.0+1im 0.0+0.0im]
sz = [1.0+0.0im 0.0+0.0im; 0.0+0.0im -1.0+0.0im]
sp = (sx+1im*sy)/2
sm = (sx-1im*sy)/2

sp_id = sparse(id)
sp_sx = sparse(sx)
sp_sy = sparse(sy)
sp_sz = sparse(sz)
sp_sp = sparse(sp)
sp_sm = sparse(sm)

"""
mutable struct parameters
    N::Int
    dim::Int
    J::Float64
    h::Float64
    γ::Float64
    α::Float64
    d_max::Int
    ϵ::Float64
    N_K::Float64
end

function set_parameters(N,J,h,γ,α,d_max,ϵ,N_K)
	params.N = N;
    params.dim = 2^N;
    params.J = J;
    params.h = h;
    params.γ = γ;
    params.α = α;
    params.d_max = d_max;
    params.ϵ = ϵ;
    params.N_K = N_K;
end

function set_parameters_ED(N,J,h,γ,α,d_max,ϵ,N_K)
	params_ED.N = N;
    params_ED.dim = 2^N;
    params_ED.J = J;
    params_ED.h = h;
    params_ED.γ = γ;
    params_ED.α = α;
    params_ED.d_max = d_max;
    params_ED.ϵ = ϵ;
    params_ED.N_K = N_K;
end
"""

function calculate_Kac_norm(d_max, α; offset=0.0) #periodic BCs only!
    N_K = offset
    #for i in 1:convert(Int64,floor(N/2))
    for i in 1:d_max
        N_K+=1/i^α
    end
    return N_K
end