module ProxSDP

    using MathOptInterface
    using TimerOutputs
    using Arpack
    using Compat
    using Printf
    using SparseArrays
    using LinearAlgebra

    include("MOI_wrapper.jl")
    include("structs.jl")
    include("util.jl")
    include("printing.jl")    
    include("scaling.jl")
    include("pdhg.jl")
    include("residuals.jl")
    include("prox_operators.jl")

    MOIU.@model _ProxSDPModelData () (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan) (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.PositiveSemidefiniteConeTriangle) () (MOI.SingleVariable,) (MOI.ScalarAffineFunction,) (MOI.VectorOfVariables,) (MOI.VectorAffineFunction,)

    Solver(;args...) = MOIU.CachingOptimizer(_ProxSDPModelData{Float64}(), ProxSDP.Optimizer(args))

end