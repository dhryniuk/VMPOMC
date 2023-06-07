export parameters


mutable struct parameters
    N::Int64
    dim::Int64
    χ::Int64
    Jx::Float32
    Jy::Float32
    J::Float32
    hx::Float32
    hz::Float32
    γ::Float32
    α::Int
    burn_in::Int
end

Base.display(params::parameters) = begin
    println("N\t", params.N)
    println("dim\t", params.dim)
    println("χ\t", params.χ)
    println("Jx\t", params.Jx)
    println("Jy\t", params.Jy)
    println("J\t", params.J)
    println("hx\t", params.hx)
    println("hz\t", params.hz)
    println("γ\t", params.γ)
    println("α\t", params.α)
    println("burn_in\t", params.burn_in)
end

function set_parameters(N,χ,Jx,Jy,J,hx,hz,γ,α,burn_in)
	params.N = N;
    params.dim = 2^N;
    params.χ = χ;
    params.Jx = Jx;
    params.Jy = Jy;
    params.J = J;
    params.hx = hx;
    params.hz = hz;
    params.γ = γ;
    params.α = α;
    params.burn_in = burn_in;
end