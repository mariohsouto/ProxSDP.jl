const MOI = MathOptInterface

MOI.Utilities.@product_of_sets(Zeros, MOI.Zeros)

MOI.Utilities.@product_of_sets(Nonpositives, MOI.Nonpositives)

MOI.Utilities.@struct_of_constraints_by_set_types(
    StructCache,
    MOI.Zeros,
    MOI.Nonpositives,
    MOI.SecondOrderCone,
    MOI.PositiveSemidefiniteConeTriangle,
)

const OptimizerCache{T} = MOI.Utilities.GenericModel{
    T,
    MOI.Utilities.ObjectiveContainer{T},
    MOI.Utilities.VariablesContainer{T},
    StructCache{T}{
        MOI.Utilities.MatrixOfConstraints{
            T,
            MOI.Utilities.MutableSparseMatrixCSC{
                T,
                Int64,
                MOI.Utilities.OneBasedIndexing,
            },
            Vector{T},
            Zeros{T},
        },
        MOI.Utilities.MatrixOfConstraints{
            T,
            MOI.Utilities.MutableSparseMatrixCSC{
                T,
                Int64,
                MOI.Utilities.OneBasedIndexing,
            },
            Vector{T},
            Nonpositives{T},
        },
        MOI.Utilities.VectorOfConstraints{
            MOI.VectorOfVariables,
            MOI.SecondOrderCone,
        },
        MOI.Utilities.VectorOfConstraints{
            MOI.VectorOfVariables,
            MOI.PositiveSemidefiniteConeTriangle,
        },
    },
}

Base.@kwdef mutable struct ConeData
    psc::Vector{Vector{Int}} = Vector{Int}[] # semidefinite
    soc::Vector{Vector{Int}} = Vector{Int}[] # second order
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    cones::Union{Nothing, ConeData}
    zeros::Union{Nothing, Zeros{Float64}}
    nonps::Union{Nothing, Nonpositives{Float64}}
    sol::Result
    options::Options
end

function Optimizer(;kwargs...)
    optimizer = Optimizer(
        nothing,
        nothing,
        nothing,
        Result(),
        Options()
        )
    for (key, value) in kwargs
        MOI.set(optimizer, MOI.RawOptimizerAttribute(string(key)), value)
    end
    return optimizer
end

#=
    Basic Attributes
=#

MOI.get(::Optimizer, ::MOI.SolverName) = "ProxSDP"

MOI.get(::Optimizer, ::MOI.SolverVersion) = "1.7.0"

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    fields = fieldnames(Options)
    name = Symbol(param.name)
    if name in fields
        setfield!(optimizer.options, name, value)
    else
        error("No parameter matching $(name)")
    end
    return value
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    fields = fieldnames(Options)
    name = Symbol(param.name)
    if name in fields
        return getfield(optimizer.options, name)
    else
        error("No parameter matching $(name)")
    end
end

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    if value == true
        optimizer.options.timer_verbose = false
    end
    optimizer.options.log_verbose = !value
end

function MOI.get(optimizer::Optimizer, ::MOI.Silent)
    if optimizer.options.log_verbose
        return false
    elseif optimizer.options.timer_verbose
        return false
    else
        return true
    end
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(optimizer::Optimizer, ::MOI.TimeLimitSec, value)
    optimizer.options.time_limit = value
end

function MOI.get(optimizer::Optimizer, ::MOI.TimeLimitSec)
    return optimizer.options.time_limit
end

MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = false

function MOI.is_empty(optimizer::Optimizer)
    return optimizer.cones === nothing &&
    optimizer.zeros === nothing &&
    optimizer.nonps === nothing &&
    optimizer.sol.status == 0
end

function MOI.empty!(optimizer::Optimizer)
    optimizer.cones = nothing
    optimizer.zeros = nothing
    optimizer.nonps = nothing
    optimizer.sol = Result()
    return
end

function _rows(
    optimizer::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{T},MOI.Zeros},
) where T
    return MOI.Utilities.rows(optimizer.zeros, ci)
end

function _rows(
    optimizer::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{T},MOI.Nonpositives},
) where T
    return MOI.Utilities.rows(optimizer.nonps, ci)
end

# MOI.supports

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}},
    },
) where T
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorAffineFunction{T}},
    ::Type{<:Union{MOI.Zeros, MOI.Nonpositives}}
) where T
    return true
end

function MOI.supports_add_constrained_variables(
    ::Optimizer,
    ::Type{<:Union{
        MOI.SecondOrderCone,
        MOI.PositiveSemidefiniteConeTriangle,
        }
    }
)
    return true
end

function cone_data(src::OptimizerCache, ::Type{T}) where T
    cache = MOI.Utilities.constraints(
        src.constraints,
        MOI.VectorOfVariables,
        T,
    )
    indices = MOI.get(
        cache,
        MOI.ListOfConstraintIndices{MOI.VectorOfVariables, T}(),
    )
    funcs = MOI.get.(cache, MOI.ConstraintFunction(), indices)
    return Vector{Int64}[Int64[v.value for v in f.variables] for f in funcs]
end

# Vectorized length for matrix dimension n
sympackedlen(n) = div(n * (n + 1), 2)
# Matrix dimension for vectorized length n
sympackeddim(n) = div(isqrt(1 + 8n) - 1, 2)

matindices(n::Integer) = LinearIndices(trues(n,n))[findall(LinearAlgebra.tril(trues(n,n)))]

function _optimize!(dest::Optimizer, src::OptimizerCache)
    MOI.empty!(dest)
    TimerOutputs.reset_timer!()

    @timeit "pre-processing" begin

    #=
        Affine
    =#
    Ab = MOI.Utilities.constraints(
        src.constraints,
        MOI.VectorAffineFunction{Float64},
        MOI.Zeros,
    )
    A = Ab.coefficients
    b = -Ab.constants
    Gh = MOI.Utilities.constraints(
        src.constraints,
        MOI.VectorAffineFunction{Float64},
        MOI.Nonpositives,
    )
    G = Gh.coefficients
    h = -Gh.constants
    @assert A.n == G.n
    #=
        Objective
    =#
    max_sense = MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    obj =
        MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    objective_constant = MOI.constant(obj)
    c = zeros(A.n)
    obj_sign = ifelse(max_sense, -1.0, 1.0)
    for term in obj.terms
        c[term.variable.value] += obj_sign * term.coefficient
    end
    dest.zeros = deepcopy(Ab.sets) # TODO copy(Ab.sets)
    dest.nonps = deepcopy(Gh.sets) # TODO copy(Gh.sets)
    # TODO: simply this after
    # https://github.com/jump-dev/MathOptInterface.jl/issues/1711
    _A = if A.m == 0
        SparseArrays.sparse(Int64[], Int64[], Float64[], 0, A.n)
    else
        convert(SparseArrays.SparseMatrixCSC{Float64,Int64}, A)
    end
    _G = if G.m == 0
        SparseArrays.sparse(Int64[], Int64[], Float64[], 0, G.n)
    else
        convert(SparseArrays.SparseMatrixCSC{Float64,Int64}, G)
    end
    aff = AffineSets(A.n, A.m, G.m, 0,
        _A, _G, b, h, c)
    #=
        Cones
    =#
    con = ConicSets(
        SDPSet[],
        SOCSet[]
        )
    soc_s = cone_data(src, MOI.SecondOrderCone)
    for soc in soc_s
        push!(con.socone, SOCSet(soc, length(soc)))
    end
    psc_s = cone_data(src, MOI.PositiveSemidefiniteConeTriangle)
    for psc in psc_s
        tri_len = length(psc)
        sq_side = sympackeddim(tri_len)
        mat_inds = matindices(sq_side)
        push!(con.sdpcone, SDPSet(psc, mat_inds, tri_len, sq_side))
    end
    dest.cones = ConeData(
        psc_s,
        soc_s,
    )
    #
    end # timeit

    #= 
        Solve modified problem
    =#

    options = dest.options

    # warm = WarmStart()

    if options.disable_julia_logger
        # disable logger
        global_log = Logging.current_logger()
        Logging.global_logger(Logging.NullLogger())
    end

    sol = @timeit "Main" chambolle_pock(aff, con, options)

    if options.disable_julia_logger
        # re-enable logger
        Logging.global_logger(global_log)
    end

    if options.timer_verbose
        TimerOutputs.print_timer(TimerOutputs.DEFAULT_TIMER)
        print("\n")
        TimerOutputs.print_timer(TimerOutputs.flatten(TimerOutputs.DEFAULT_TIMER))
        print("\n")
    end
    if options.timer_file
        f = open("time.log","w")
        TimerOutputs.print_timer(f,TimerOutputs.DEFAULT_TIMER)
        print(f,"\n")
        TimerOutputs.print_timer(f,TimerOutputs.flatten(TimerOutputs.DEFAULT_TIMER))
        print(f,"\n")
        close(f)
    end

    #= 
        Fix solution
    =#

    sol.objval = obj_sign * sol.objval + objective_constant
    sol.dual_objval = obj_sign * sol.dual_objval + objective_constant

    dest.sol = sol

    return
end

function MOI.optimize!(dest::Optimizer, src::OptimizerCache)
    _optimize!(dest, src)
    return MOI.Utilities.identity_index_map(src), false
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    cache = OptimizerCache{Float64}()
    index_map = MOI.copy_to(cache, src)
    _optimize!(dest, cache)
    return index_map, false
end

#=
    Attributes
=#

MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec) = optimizer.sol.time

"""
    PDHGIterations()

The number of PDHG iterations completed during the solve.
"""
struct PDHGIterations <: MOI.AbstractModelAttribute end

MOI.is_set_by_optimize(::PDHGIterations) = true

function MOI.get(optimizer::Optimizer, ::PDHGIterations)
    return Int64(optimizer.sol.iter)
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return optimizer.sol.status_string
end

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    s = optimizer.sol.status
    @assert 0 <= s <= 6
    if s == 0
        return MOI.OPTIMIZE_NOT_CALLED
    elseif s == 1
        return MOI.OPTIMAL
    elseif s == 2
        return MOI.TIME_LIMIT
    elseif s == 3
        return MOI.ITERATION_LIMIT
    elseif s == 4
        return MOI.INFEASIBLE_OR_UNBOUNDED
    elseif s == 5
        return MOI.DUAL_INFEASIBLE
    elseif s == 6
        return MOI.INFEASIBLE
    end
end

function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    value = optimizer.sol.objval
    return value
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    value = optimizer.sol.dual_objval
    return value
end

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    s = optimizer.sol.status
    if attr.result_index > 1 || s == 0
        return MOI.NO_SOLUTION
    end
    if s == 5 && optimizer.sol.certificate_found
        return MOI.INFEASIBILITY_CERTIFICATE
    end
    if optimizer.sol.primal_feasible_user_tol#s == 1
        return MOI.FEASIBLE_POINT
    else
        return MOI.INFEASIBLE_POINT
    end
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualStatus)
    s = optimizer.sol.status
    if attr.result_index > 1 || s ==0
        return MOI.NO_SOLUTION
    end
    if s == 6 && optimizer.sol.certificate_found
        return MOI.INFEASIBILITY_CERTIFICATE
    end
    if optimizer.sol.dual_feasible_user_tol
        return MOI.FEASIBLE_POINT
    else
        return MOI.INFEASIBLE_POINT
    end
end

function MOI.get(optimizer::Optimizer, ::MOI.ResultCount)
    return optimizer.sol.result_count
end

#=
    Results
=#

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.primal[vi.value]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros},
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.slack_eq[_rows(optimizer, ci)]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonpositives},
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.slack_in[_rows(optimizer, ci)]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.PositiveSemidefiniteConeTriangle}
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.primal[optimizer.cones.psc[ci.value]]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.SecondOrderCone}
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.primal[optimizer.cones.soc[ci.value]]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64},MOI.Zeros},
)
    MOI.check_result_index_bounds(optimizer, attr)
    return -optimizer.sol.dual_eq[_rows(optimizer, ci)]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64},MOI.Nonpositives},
)
    MOI.check_result_index_bounds(optimizer, attr)
    return -optimizer.sol.dual_in[_rows(optimizer, ci)]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.PositiveSemidefiniteConeTriangle}
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.dual_cone[optimizer.cones.psc[ci.value]]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.SecondOrderCone}
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.dual_cone[optimizer.cones.soc[ci.value]]
end
