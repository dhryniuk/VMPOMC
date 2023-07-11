export Parameters


mutable struct Parameters
    N::Int64
    dim::Int64
    χ::Int64
    Jx::Float32
    Jy::Float32
    J::Float32
    hx::Float32
    hz::Float32
    γ::Float32
    γ_d::Float32
    α::Float64
    #burn_in::Int
end

Base.display(params::Parameters) = begin
    println("\nParameters:")
    println("N\t\t", params.N)
    println("dim\t\t", params.dim)
    println("χ\t\t", params.χ)
    println("Jx\t\t", params.Jx)
    println("Jy\t\t", params.Jy)
    println("Jz\t\t", params.J)
    println("hx\t\t", params.hx)
    println("hz\t\t", params.hz)
    println("γ_l\t\t", params.γ)
    println("γ_d\t\t", params.γ_d)
    println("α\t\t", params.α)
    #println("burn_in\t", params.burn_in)
end

#write a constructor that defaults to 0 whenever some paramter is not specified...

function set_parameters(N,χ,Jx,Jy,J,hx,hz,γ,γ_d,α,burn_in)
	params.N = N;
    params.dim = 2^N;
    params.χ = χ;
    params.Jx = Jx;
    params.Jy = Jy;
    params.J = J;
    params.hx = hx;
    params.hz = hz;
    params.γ = γ;
    params.γ_d = γ_d;
    params.α = α;
    #params.burn_in = burn_in;
end