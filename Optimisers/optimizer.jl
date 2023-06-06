abstract type OptimizerCache end

abstract type StochasticCache <: OptimizerCache end

abstract type Optimizer{T} end

abstract type Exact{T} <: Optimizer{T} end

abstract type Stochastic{T} <: Optimizer{T} end