
export ProxSDPSolverInstance

using MathOptInterface
const MOI = MathOptInterface
const CI = MOI.ConstraintIndex
const VI = MOI.VariableIndex

using MathOptInterfaceUtilities
const MOIU = MathOptInterfaceUtilities

MOIU.@instance ProxSDPInstanceData () (EqualTo, GreaterThan, LessThan) (Zeros, Nonnegatives, Nonpositives, PositiveSemidefiniteConeTriangle) () (SingleVariable,) (ScalarAffineFunction,) (VectorOfVariables,) (VectorAffineFunction,)

mutable struct ProxSDPSolverInstance <: MOI.AbstractSolverInstance
    data::ProxSDPInstanceData{Float64}
    varmap::Dict{VI, Int}
    constrmap::Dict{Int64, Int}
    ret_val::Int
    primal::Vector{Float64}
    dual::Vector{Float64}
    slack::Vector{Float64}
    objval::Float64
    function ProxSDPSolverInstance()
        new(ProxSDPInstanceData{Float64}(), Dict{VI, Int}(), Dict{Int64, Int}(), 1, Float64[], Float64[], Float64[], 0.)
    end
end

# Redirect data modification calls to data
include("data.jl")

# Implements optimize! : translate data to SCSData and call SCS_solve
include("solve.jl")

# Implements getter for result value and statuses
include("attributes.jl")
