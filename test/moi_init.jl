const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities

use_MOI = true

MOIU.@model ProxSDPModelData () (EqualTo, GreaterThan, LessThan) (Zeros, Nonnegatives, Nonpositives, PositiveSemidefiniteConeTriangle) () (SingleVariable,) (ScalarAffineFunction,) (VectorOfVariables,) (VectorAffineFunction,)