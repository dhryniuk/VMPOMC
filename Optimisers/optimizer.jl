abstract type OptimizerCache end

abstract type StochasticCache <: OptimizerCache end

abstract type Optimizer end

abstract type Exact{T} <: Optimizer end

abstract type Stochastic{T} <: Optimizer end