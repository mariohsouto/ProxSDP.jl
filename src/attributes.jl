# SCS returns one of the following integers:
# -7 SCS_INFEASIBLE_INACCURATE
# -6 SCS_UNBOUNDED_INACCURATE
# -5 SCS_SIGINT
# -4 SCS_FAILED
# -3 SCS_INDETERMINATE
# -2 SCS_INFEASIBLE  : primal infeasible, dual unbounded
# -1 SCS_UNBOUNDED   : primal unbounded, dual infeasible
#  0 SCS_UNFINISHED  : never returned, used as placeholder
#  1 SCS_SOLVED
#  2 SCS_SOLVED_INACCURATE
MOI.canget(instance::ProxSDPSolverInstance, ::MOI.TerminationStatus) = true
function MOI.get(instance::ProxSDPSolverInstance, ::MOI.TerminationStatus)
    s = instance.ret_val
    @assert -7 <= s <= 2
    @assert s != 0
    if s in (-7, -6, 2)
        MOI.AlmostSuccess
    elseif s == -5
        MOI.Interrupted
    elseif s == -4
        MOI.NumericalError
    elseif s == -3
        MOI.SlowProgress
    else
        @assert -2 <= s <= 1
        MOI.Success
    end
end

MOI.canget(instance::ProxSDPSolverInstance, ::MOI.ObjectiveValue) = true
MOI.get(instance::ProxSDPSolverInstance, ::MOI.ObjectiveValue) = instance.objval

MOI.canget(instance::ProxSDPSolverInstance, ::MOI.PrimalStatus) = true
function MOI.get(instance::ProxSDPSolverInstance, ::MOI.PrimalStatus)
    s = instance.ret_val
    if s in (-3, 1, 2)
        MOI.FeasiblePoint
    elseif s in (-6, -1)
        MOI.InfeasibilityCertificate
    else
        MOI.InfeasiblePoint
    end
end
function MOI.canget(instance::ProxSDPSolverInstance, ::Union{MOI.VariablePrimal, MOI.ConstraintPrimal}, r::MOI.Index)
    instance.ret_val in (-6, -3, -1, 1, 2)
end
function MOI.canget(instance::ProxSDPSolverInstance, ::Union{MOI.VariablePrimal, MOI.ConstraintPrimal}, r::Vector{<:MOI.Index})
    instance.ret_val in (-6, -3, -1, 1, 2)
end
function MOI.get(instance::ProxSDPSolverInstance, ::MOI.VariablePrimal, vr::VI)
    instance.primal[instance.varmap[vr]]
end
MOI.get(instance::ProxSDPSolverInstance, a::MOI.VariablePrimal, vr::Vector{VI}) = MOI.get.(instance, a, vr)
_unshift(value, s) = value
_unshift(value, s::MOI.EqualTo) = value + s.value
_unshift(value, s::MOI.GreaterThan) = value + s.lower
_unshift(value, s::MOI.LessThan) = value + s.upper
reorderval(val, s) = val
function reorderval(val, s::MOI.PositiveSemidefiniteConeTriangle)
    sympackedLtoU(val, s.dimension)
end
function MOI.get(instance::ProxSDPSolverInstance, ::MOI.ConstraintPrimal, ci::CI)
    offset = instance.constrmap[ci.value]
    s = MOI.get(instance, MOI.ConstraintSet(), ci)
    rows = constrrows(s)
    _unshift(scalecoef(rows, reorderval(instance.slack[offset + rows], s), false, s, true), s)
end

MOI.canget(instance::ProxSDPSolverInstance, ::MOI.DualStatus) = true
function MOI.get(instance::ProxSDPSolverInstance, ::MOI.DualStatus)
    s = instance.ret_val
    if s in (-3, 1, 2)
        MOI.FeasiblePoint
    elseif s in (-7, -2)
        MOI.InfeasibilityCertificate
    else
        MOI.InfeasiblePoint
    end
end
function MOI.canget(instance::ProxSDPSolverInstance, ::MOI.ConstraintDual, ::CI)
    instance.ret_val in (-7, -3, -2, 1, 2)
end
function MOI.get(instance::ProxSDPSolverInstance, ::MOI.ConstraintDual, ci::CI)
    offset = instance.constrmap[ci.value]
    s = MOI.get(instance, MOI.ConstraintSet(), ci)
    rows = constrrows(s)
    scalecoef(rows, reorderval(instance.dual[offset + rows], s), false, s, true)
end

MOI.canget(instance::ProxSDPSolverInstance, ::MOI.ResultCount) = true
MOI.get(instance::ProxSDPSolverInstance, ::MOI.ResultCount) = 1