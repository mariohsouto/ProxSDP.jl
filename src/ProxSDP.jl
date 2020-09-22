module ProxSDP

    using Arpack
    using KrylovKit
    using MathOptInterface
    using TimerOutputs

    using Printf
    using SparseArrays
    using LinearAlgebra
    import Random

    import LinearAlgebra: BlasInt

    include("structs.jl")
    include("util.jl")
    include("printing.jl")
    include("scaling.jl")
    include("equilibration.jl")
    include("pdhg.jl")
    include("residuals.jl")
    include("eigsolver.jl")
    include("prox_operators.jl")

    include("MOI_wrapper.jl")

    MOIU.@model _ProxSDPModelData () (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan) (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.PositiveSemidefiniteConeTriangle) () (MOI.SingleVariable,) (MOI.ScalarAffineFunction,) (MOI.VectorOfVariables,) (MOI.VectorAffineFunction,)

    Solver(;args...) = MOIU.CachingOptimizer(_ProxSDPModelData{Float64}(), ProxSDP.Optimizer(args))

end