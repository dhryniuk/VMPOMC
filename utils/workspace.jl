""" Cache which stores intermediate results to reduce memory allocations"""
mutable struct Workspace{T<:Complex{<:AbstractFloat}}
    L_set::Vector{Matrix{T}}
    R_set::Vector{Matrix{T}}
    micro_L_set::Vector{Matrix{T}}
    micro_R_set::Vector{Matrix{T}}
    plus_S::Array{T,2}
    B::Matrix{T}
    ID::Matrix{T}
    loc_1::Matrix{T}
    loc_2::Matrix{T}
    loc_3::Matrix{T}
    Metro_1::Matrix{T}
    Metro_2::Matrix{T}
    C_mat::Matrix{T}
    bra_L_l1::Matrix{T}
    bra_L_l2::Matrix{T}
    ∂::Array{T,3}
    Δ::Array{T,3} #tensor of derivatives
    #slw::SweepLindbladWorkspace{T}
    
    sample::Projector
    micro_sample::Projector
    dVEC_transpose::Dict{Tuple{Bool,Bool},Matrix{T}}
    s::Matrix{T}

    local_L::T
    local_∇L::Array{T,3}
    l_int::T
    local_∇L_diagonal_coeff::T

    #temp_local_L::T
    #temp_local_∇L::Array{T,3}

end

function set_workspace(A::Array{T,3}, params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    o = convert(T, 0.0+0.0im)
    i = convert(T, 1.0+0.0im)
    cache = Workspace(
        #zeros(T, params.χ, params.χ, 4),
        [ Matrix{T}(undef,params.χ,params.χ) for _ in 1:params.N+1 ],
        [ Matrix{T}(undef,params.χ,params.χ) for _ in 1:params.N+1 ],
        [ Matrix{T}(undef,params.χ,params.χ) for _ in 1:params.N+1 ],
        [ Matrix{T}(undef,params.χ,params.χ) for _ in 1:params.N+1 ],
        zeros(T, 4*params.χ^2,4*params.χ^2),
        zeros(T, params.χ,params.χ),
        Matrix{T}(I, params.χ, params.χ),
        zeros(T, params.χ,params.χ),
        zeros(T, params.χ,params.χ),
        zeros(T, params.χ,params.χ),
        zeros(T, params.χ,params.χ),
        zeros(T, params.χ,params.χ),
        zeros(T, params.χ,params.χ),
        zeros(T, 1, 4),
        zeros(T, 1, 16),
        zeros(T, params.χ, params.χ, 4),
        zeros(T, params.χ, params.χ, 4),

        Projector([false],[false]),
        Projector([false],[false]),
        Dict((false,false) => [i o o o], (false,true) => [o i o o], (true,false) => [o o i o], (true,true) => [o o o i]),
        zeros(T, 1, 16),

        0.0+0.0im,
        zeros(T,params.χ,params.χ,4),
        0.0+0.0im,
        0.0+0.0im#,

        #0.0+0.0im,
        #zeros(T,params.χ,params.χ,4)
        )
    return cache
end

"""
mutable struct SweepLindbladWorkspace{T}
    local_L::T
    local_∇L::Array{T,3}
    l_int::T
end


function Init!(slw::SweepLindbladWorkspace{T}) where {T<:Complex{<:AbstractFloat}}
    slw.local_L = 0
    slw.local_∇L = zeros(T,params.χ,params.χ,4)
    slw.l_int = 0
end
"""