
module ProxSDP

    using MathOptInterface, TimerOutputs
    using Arpack
    using Compat
    using Printf

    include("MOI_wrapper.jl")
    include("eigsolver.jl")

    MOIU.@model _ProxSDPModelData () (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan) (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.PositiveSemidefiniteConeTriangle) () (MOI.SingleVariable,) (MOI.ScalarAffineFunction,) (MOI.VectorOfVariables,) (MOI.VectorAffineFunction,)

    Solver(;args...) = MOIU.CachingOptimizer(_ProxSDPModelData{Float64}(), ProxSDP.Optimizer(args))

end