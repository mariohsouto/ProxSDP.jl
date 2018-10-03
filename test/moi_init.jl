const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities

use_MOI = true

MOIU.@model ProxSDPModelData () (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan) (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.PositiveSemidefiniteConeTriangle) () (MOI.SingleVariable,) (MOI.ScalarAffineFunction,) (MOI.VectorOfVariables,) (MOI.VectorAffineFunction,)