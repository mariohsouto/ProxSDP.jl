# References
MOI.candelete(instance::ProxSDPSolverInstance, r::MOI.Index) = MOI.candelete(instance.data, r)
MOI.isvalid(instance::ProxSDPSolverInstance, r::MOI.Index) = MOI.isvalid(instance.data, r)
MOI.delete!(instance::ProxSDPSolverInstance, r::MOI.Index) = MOI.delete!(instance.data, r)

# Attributes
for f in (:canget, :canset, :set!, :get, :get!)
    @eval begin
        MOI.$f(instance::ProxSDPSolverInstance, attr::MOI.AnyAttribute) = MOI.$f(instance.data, attr)
        MOI.$f(instance::ProxSDPSolverInstance, attr::MOI.AnyAttribute, ref::MOI.Index) = MOI.$f(instance.data, attr, ref)
        MOI.$f(instance::ProxSDPSolverInstance, attr::MOI.AnyAttribute, refs::Vector{<:MOI.Index}) = MOI.$f(instance.data, attr, refs)
        # Objective function
        MOI.$f(instance::ProxSDPSolverInstance, attr::MOI.AnyAttribute, arg::Union{MOI.OptimizationSense, MOI.AbstractScalarFunction}) = MOI.$f(instance.data, attr, arg)
    end
end

# Constraints
MOI.canaddconstraint(instance::ProxSDPSolverInstance, f::MOI.AbstractFunction, s::MOI.AbstractSet) = MOI.canaddconstraint(instance.data, f, s)
MOI.addconstraint!(instance::ProxSDPSolverInstance, f::MOI.AbstractFunction, s::MOI.AbstractSet) = MOI.addconstraint!(instance.data, f, s)
MOI.canmodifyconstraint(instance::ProxSDPSolverInstance, ci::CI, change) = MOI.canmodifyconstraint(instance.data, ci, change)
MOI.modifyconstraint!(instance::ProxSDPSolverInstance, ci::CI, change) = MOI.modifyconstraint!(instance.data, ci, change)

# Objective
MOI.canmodifyobjective(instance::ProxSDPSolverInstance, change::MOI.AbstractFunctionModification) = MOI.canmodifyobjective(instance.data, change)
MOI.modifyobjective!(instance::ProxSDPSolverInstance, change::MOI.AbstractFunctionModification) = MOI.modifyobjective!(instance.data, change)

# Variables
MOI.addvariable!(instance::ProxSDPSolverInstance) = MOI.addvariable!(instance.data)
MOI.addvariables!(instance::ProxSDPSolverInstance, n) = MOI.addvariables!(instance.data, n)