
const MOI = MathOptInterface
const CI = MOI.ConstraintIndex
const VI = MOI.VariableIndex

const MOIU = MOI.Utilities

Base.@kwdef mutable struct MOISolution
    ret_val::Int = 0
    raw_status::String = "Problem not solved"
    primal::Vector{Float64} = Float64[] # primal of variables
    dual_eq::Vector{Float64} = Float64[] # dual of constraints
    dual_in::Vector{Float64} = Float64[] # dual of constraints
    # dual_sd::Vector{Vector{Float64}}
    # dual_so::Vector{Vector{Float64}}
    slack_eq::Vector{Float64} = Float64[]
    slack_in::Vector{Float64} = Float64[]
    # slack_sd::Vector{Vector{Float64}}
    # slack_so::Vector{Vector{Float64}}
    primal_residual::Float64 = NaN
    dual_residual::Float64 = NaN
    objval::Float64 = NaN
    dual_objval::Float64 = NaN
    # obj_cte::Float64 = NaN # used in rays - see SCS
    gap::Float64 = NaN
    time::Float64 = NaN
    final_rank::Int = -1
end

# MOISolution() = MOISolution(0, # SCS_UNFINISHED
#                       Float64[], Float64[], Float64[], NaN, NaN, NaN, NaN, NaN, NaN, 0)

# Used to build the data with allocate-load during `copy_to`.
# When `optimize!` is called, a the data is passed to SCS
# using `SCS_solve` and the `ModelData` struct is discarded
mutable struct ModelData
    m::Int # Number of rows/constraints
    n::Int # Number of cols/variables

    # equality matrix
    I::Vector{Int} # List of rows
    J::Vector{Int} # List of cols
    V::Vector{Float64} # List of coefficients
    # equality rhs
    b::Vector{Float64} # constants

    # inequality matrix
    Ii::Vector{Int} # List of rows
    Ji::Vector{Int} # List of cols
    Vi::Vector{Float64} # List of coefficients
    # inequality rhs
    h::Vector{Float64} # constants

    objective_constant::Float64 # The objective is min c'x + objective_constant
    c::Vector{Float64}
end

# This is tied to SCS's internal representation
Base.@kwdef mutable struct ConeData

    cone_cols::Int = 0

    # matrix lines
    eq_tot::Int = 0 # number of linear equality constraints
    eq_start::Vector{Int} = Int[]
    eq_len::Vector{Int} = Int[]
    in_tot::Int = 0 # length of LP cone
    in_start::Vector{Int} = Int[]
    in_len::Vector{Int} = Int[]

    # cones
    sdc::Vector{Vector{Int}} = Vector{Int}[] # semidefinite
    soc::Vector{Vector{Int}} = Vector{Int}[] # second order
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    cone::ConeData
    maxsense::Bool
    data::Union{Nothing, ModelData} # only non-Void between MOI.copy_to and MOI.optimize!
    sol::MOISolution
    options::Options
end
function Optimizer(;kwargs...)
    optimizer = Optimizer(ConeData(), false, nothing, MOISolution(), Options())
    for (key, value) in kwargs
        MOI.set(optimizer, MOI.RawParameter(key), value)
    end
    return optimizer
end

MOI.get(::Optimizer, ::MOI.SolverName) = "ProxSDP"

function MOI.set(optimizer::Optimizer, param::MOI.RawParameter, value)
    fields = fieldnames(Options)
    name = param.name
    if name in fields
        setfield!(optimizer.options, name, value)
    else
        error("No parameter matching $(name)")
    end
    return value
end
function MOI.get(optimizer::Optimizer, param::MOI.RawParameter)
    fields = fieldnames(Options)
    name = param.name
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

MOI.supports(::Optimizer, ::MOI.TimeLimitSec ) = true
function MOI.set(optimizer::Optimizer, ::MOI.TimeLimitSec , value)
    optimizer.options.time_limit = value
end
function MOI.get(optimizer::Optimizer, ::MOI.TimeLimitSec)
    return optimizer.options.time_limit
end

function MOI.is_empty(optimizer::Optimizer)
    !optimizer.maxsense && optimizer.data === nothing
end
function MOI.empty!(optimizer::Optimizer)
    optimizer.maxsense = false
    optimizer.data = nothing # It should already be nothing except if an error is thrown inside copy_to
    optimizer.sol.ret_val = 0
end

MOIU.supports_allocate_load(::Optimizer, copy_names::Bool) = !copy_names

function MOI.supports(::Optimizer,
                      ::Union{MOI.ObjectiveSense,
                              MOI.ObjectiveFunction{MOI.SingleVariable},
                              MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}})
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:MOI.VectorAffineFunction{Float64}},
    ::Type{<:Union{MOI.Zeros, MOI.Nonpositives}})
    return true
end
# function MOI.supports_constraint(
#     ::Optimizer,
#     ::Type{<:MOI.ScalarAffineFunction{Float64}},
#     ::Type{<:Union{MOI.EqualTo{Float64}, MOI.GreaterThan{Float64},
#         MOI.LessThan{Float64}}})
#     return true
# end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:MOI.VectorOfVariables},
    ::Type{<:Union{MOI.SecondOrderCone,
                   MOI.PositiveSemidefiniteConeTriangle}})
    return true
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike; kws...)
    return MOIU.automatic_copy_to(dest, src; kws...)
end

const SupportedSets = Union{MOI.PositiveSemidefiniteConeTriangle,
                            MOI.SecondOrderCone}
function get_new_variable_indexes(optimizer::Optimizer, set::SupportedSets)
    len = MOI.dimension(set)
    first = optimizer.cone.cone_cols + 1
    last = optimizer.cone.cone_cols + len
    optimizer.cone.cone_cols += len
    return first, last
end
function get_cone_list(optimizer::Optimizer, set::MOI.PositiveSemidefiniteConeTriangle)
    return optimizer.cone.sdp
end
function get_cone_list(optimizer::Optimizer, set::MOI.SecondOrderCone)
    return optimizer.cone.soc
end

function MOIU.allocate_constrained_variables(optimizer::Optimizer,
    set::SupportedSets)
    first, last = get_new_variable_indexes(optimizer, set)
    list = get_cone_list(optimizer, set)
    push!(list, collect(first:last))
    ci = MOI.ConstraintIndex{MOI.VectorOfVariables, typeof(set)}(length(list))
    return [MOI.VariableIndex(i) for i in first:last], ci
end

function MOIU.load_constrained_variables(
    optimizer::Optimizer, vis::Vector{MOI.VariableIndex},
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables},
    set::SupportedSets)
end

function _allocate_constraint(cone::ConeData, f, s::MOI.Zeros)
    len = MOI.dimension(s)
    push!(cone.eq_start, cone.eq_tot + 1)
    push!(cone.eq_len, len)
    cone.eq_tot += len
    return length(cone.eq_len)
end

function _allocate_constraint(cone::ConeData, f, s::MOI.Nonpositives)
    len = MOI.dimension(s)
    push!(cone.in_start, cone.in_tot + 1)
    push!(cone.in_len, len)
    cone.in_tot += len
    return length(cone.in_len)
end

function MOIU.allocate_constraint(optimizer::Optimizer, f::F, s::S) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
    return CI{F, S}(_allocate_constraint(optimizer.cone, f, s))
end

# Vectorized length for matrix dimension n
sympackedlen(n) = div(n*(n+1), 2)
# Matrix dimension for vectorized length n
sympackeddim(n) = div(isqrt(1+8n) - 1, 2)
# trimap(i::Integer, j::Integer) = i < j ? trimap(j, i) : div((i-1)*i, 2) + j
# trimapL(i::Integer, j::Integer, n::Integer) = i < j ? trimapL(j, i, n) : i + div((2n-j) * (j-1), 2)
# function _sympackedto(x, n, mapfrom, mapto)
#     @assert length(x) == sympackedlen(n)
#     y = similar(x)
#     for i in 1:n, j in 1:i
#         y[mapto(i, j)] = x[mapfrom(i, j)]
#     end
#     y
# end
# sympackedLtoU(x, n=sympackeddim(length(x))) = _sympackedto(x, n, (i, j) -> trimapL(i, j, n), trimap)
# sympackedUtoL(x, n=sympackeddim(length(x))) = _sympackedto(x, n, trimap, (i, j) -> trimapL(i, j, n))

# function sympackedUtoLidx(x::AbstractVector{<:Integer}, n)
#     y = similar(x)
#     map = sympackedLtoU(1:sympackedlen(n), n)
#     for i in eachindex(y)
#         y[i] = map[x[i]]
#     end
#     y
# end

# Scale coefficients depending on rows index
# rows: List of row indices
# coef: List of corresponding coefficients
# minus: if true, multiply the result by -1
# d: dimension of set
# rev: if true, we unscale instead (e.g. divide by √2 instead of multiply for PSD cone)
# _scalecoef(rows, coef, minus, ::Type{<:MOI.AbstractSet}, d, rev) = minus ? -coef : coef
# _scalecoef(rows, coef, minus, ::Union{Type{<:MOI.LessThan}, Type{<:MOI.Nonpositives}}, d, rev) = minus ? coef : -coef
# function _scalecoef(rows, coef, minus, ::Type{MOI.PositiveSemidefiniteConeTriangle}, d, rev)
#     scaling = minus ? -1 : 1
#     scaling * coef
#     #scaling2 = rev ? scaling / √2 : scaling * √2
#     #output = copy(coef)
#     #diagidx = BitSet()
#     #for i in 1:d
#     #    push!(diagidx, trimap(i, i))
#     #end
#     #for i in 1:length(output)
#     #    if rows[i] in diagidx
#     #        output[i] *= scaling
#     #    else
#     #        output[i] *= scaling2
#     #    end
#     #end
#     #output
# end
# # Unscale the coefficients in `coef` with respective rows in `rows` for a set `s` and multiply by `-1` if `minus` is `true`.
# scalecoef(rows, coef, minus, s) = _scalecoef(rows, coef, minus, typeof(s), MOI.dimension(s), false)
# # Unscale the coefficients in `coef` with respective rows in `rows` for a set of type `S` with dimension `d`
# unscalecoef(rows, coef, S::Type{<:MOI.AbstractSet}, d) = _scalecoef(rows, coef, false, S, d, true)
# unscalecoef(rows, coef, S::Type{MOI.PositiveSemidefiniteConeTriangle}, d) = _scalecoef(rows, coef, false, S, sympackeddim(d), true)

output_index(t::MOI.VectorAffineTerm) = t.output_index
variable_index_value(v::MOI.VariableIndex) = v.value
variable_index_value(t::MOI.ScalarAffineTerm) = t.variable_index.value
variable_index_value(t::MOI.VectorAffineTerm) = variable_index_value(t.scalar_term)
coefficient(t::MOI.ScalarAffineTerm) = t.coefficient
coefficient(t::MOI.VectorAffineTerm) = coefficient(t.scalar_term)
# constrrows: Recover the number of rows used by each constraint.
# When, the set is available, simply use MOI.dimension
# constrrows(::MOI.AbstractScalarSet) = 1
# constrrows(s::MOI.AbstractVectorSet) = 1:MOI.dimension(s)
# # When only the index is available, use the `optimizer.ncone.nrows` field
# constrrows(optimizer::Optimizer, ci::CI{<:MOI.AbstractScalarFunction, <:MOI.AbstractScalarSet}) = 1
# constrrows(optimizer::Optimizer, ci::CI{<:MOI.AbstractVectorFunction, <:MOI.AbstractVectorSet}) = 1:optimizer.cone.nrows[constroffset(optimizer, ci)]
# function MOIU.load_constraint(optimizer::Optimizer, ci::CI, f::MOI.SingleVariable,
#     s::MOI.AbstractScalarSet)
#     MOIU.load_constraint(optimizer, ci, MOI.ScalarAffineFunction{Float64}(f), s)
# end
# function MOIU.load_constraint(optimizer::Optimizer, ci::CI,
#     f::MOI.ScalarAffineFunction, s::MOI.AbstractScalarSet)
#     a = sparsevec(variable_index_value.(f.terms), coefficient.(f.terms))
#     # sparsevec combines duplicates with + but does not remove zeros created
#     # so we call dropzeros!
#     dropzeros!(a)
#     offset = constroffset(optimizer, ci)
#     row = constrrows(s)
#     i = offset + row
#     # The SCS format is b - Ax ∈ cone
#     # so minus=false for b and minus=true for A
#     setconstant = MOI.constant(s)
#     optimizer.cone.setconstant[offset] = setconstant
#     constant = f.constant - setconstant
#     optimizer.data.b[i] = scalecoef(row, constant, false, s)
#     append!(optimizer.data.I, fill(i, length(a.nzind)))
#     append!(optimizer.data.J, a.nzind)
#     append!(optimizer.data.V, scalecoef(row, a.nzval, true, s))
#     nothing
# end
# orderval(val, s) = val
# # function orderval(val, s::MOI.PositiveSemidefiniteConeTriangle)
# #     sympackedUtoL(val, s.side_dimension)
# # end
# orderidx(idx, s) = idx
# function orderidx(idx, s::MOI.PositiveSemidefiniteConeTriangle)
#     sympackedUtoLidx(idx, s.side_dimension)
# end

# function MOIU.load_constraint(optimizer::Optimizer, ci::CI, f::MOI.VectorOfVariables, s)
#     MOIU.load_constraint(optimizer, ci, MOI.VectorAffineFunction{Float64}(f), s)
# end
# function MOIU.load_constraint(optimizer::Optimizer, ci::CI,
#                               f::MOI.VectorOfVariables,
#                               s::MOI.PositiveSemidefiniteConeTriangle)
#                               try
#     push!(optimizer.data.sd, orderval(variable_index_value.(f.variables), s))
#                               catch e
#                                 @show f
#                                 @show variable_index_value.(f.variables)
#                                 @show s
#                                 rethrow(e)
#                               end
#     nothing
# end
# function MOIU.load_constraint(optimizer::Optimizer, ci::CI, f::MOI.VectorOfVariables,
#                               s::MOI.SecondOrderCone)
#     push!(optimizer.data.so, orderval(variable_index_value.(f.variables), s))
#     nothing
# end
function MOIU.load_constraint(optimizer::Optimizer, ci::CI,
                              f::MOI.VectorAffineFunction,
                              s::MOI.AbstractVectorSet)
    A = sparse(output_index.(f.terms), variable_index_value.(f.terms),
               coefficient.(f.terms))
    # sparse combines duplicates with + but does
    # not remove zeros created so we call dropzeros!
    dropzeros!(A)
    I, J, V = findnz(A)
    offset = offset(optimizer, set, ci)
    rows = 1:len(optimizer, set, ci)
    i = offset .+ rows

    rhs = matrix_rhs_vec(optimizer, set)
    rhs[i] .= f.constants

    # MOI:  Ax + b {==, <=} 0
    # psdp: Ax {==, <=} -b
    I_, J_, V_ = matrix_triplets(optimizer, set)
    append!(I_, offset .+ I)
    append!(J_, J)
    append!(V_, V)

    nothing
end
function offset(optimizer::Optimizer, set::MOI.Zeros, ci)
    return optimizer.cone.eq_start[ci.value]
end
function len(optimizer::Optimizer, set::MOI.Zeros, ci)
    return optimizer.cone.eq_len[ci.value]
end
function offset(optimizer::Optimizer, set::MOI.Nonpositives, ci)
    return optimizer.cone.in_start[ci.value]
end
function len(optimizer::Optimizer, set::MOI.Nonpositives, ci)
    return optimizer.cone.in_len[ci.value]
end
function matrix_triplets(optimizer::Optimizer, set::MOI.Zeros)
    return optimizer.data.I, optimizer.data.J, optimizer.data.V
end
function matrix_rhs_vec(optimizer::Optimizer, set::MOI.Zeros)
    return optimizer.data.b
end
function matrix_triplets(optimizer::Optimizer, set::MOI.Nonpositives)
    return optimizer.data.Ii, optimizer.data.Ji, optimizer.data.Vi
end
function matrix_rhs_vec(optimizer::Optimizer, set::MOI.Nonpositives)
    return optimizer.data.h
end

# first allocate-load method to be called
function MOIU.allocate_variables(optimizer::Optimizer, nvars::Integer)
    optimizer.cone = ConeData()
    VI.(1:nvars)
end

function MOIU.load_variables(optimizer::Optimizer, nvars::Integer)
    # @show nvars
    cone = optimizer.cone
    m = cone.eq_tot + cone.in_tot # + cone.q + cone.s + 3cone.ep + cone.ed
    I = Int[]
    J = Int[]
    V = Float64[]
    b = zeros(cone.eq_tot)
    Ii = Int[]
    Ji = Int[]
    Vi = Float64[]
    h = zeros(cone.in_tot)

    tot_vars = nvars + cone.cone_cols
    c = zeros(tot_vars)
    optimizer.data = ModelData(m, nvars, I, J, V, b, Ii, Ji, Vi, h, 0., c)
    # `optimizer.sol` contains the result of the previous optimization.
    # It is used as a warm start if its length is the same, e.g.
    # probably because no variable and/or constraint has been added.
    if length(optimizer.sol.primal) != tot_vars
        optimizer.sol.primal = zeros(tot_vars)
    end
    # TODO @joaquim
    # @assert length(optimizer.sol.dual) == length(optimizer.sol.slack)
    # if length(optimizer.sol.dual) != m
    #     optimizer.sol.dual = zeros(m)
    #     optimizer.sol.slack = zeros(m)
    # end
    return nothing
end

function MOIU.allocate(::Optimizer, ::MOI.VariablePrimalStart,
                       ::MOI.VariableIndex, ::Union{Nothing, Float64})
end
function MOIU.allocate(::Optimizer, ::MOI.ConstraintPrimalStart,
                       ::MOI.ConstraintIndex, ::Float64)
end
function MOIU.allocate(::Optimizer, ::MOI.ConstraintDualStart,
                       ::MOI.ConstraintIndex, ::Float64)
end
function MOIU.allocate(optimizer::Optimizer, ::MOI.ObjectiveSense,
                       sense::MOI.OptimizationSense)
    optimizer.maxsense = sense == MOI.MAX_SENSE
end
function MOIU.allocate(::Optimizer, ::MOI.ObjectiveFunction,
    ::MOI.Union{MOI.SingleVariable,
                MOI.ScalarAffineFunction{Float64}})
end

function MOIU.load(::Optimizer, ::MOI.VariablePrimalStart,
    ::MOI.VariableIndex, ::Nothing)
end
function MOIU.load(optimizer::Optimizer, ::MOI.VariablePrimalStart,
                   vi::MOI.VariableIndex, value::Float64)
    optimizer.sol.primal[vi.value] = value
end
function MOIU.load(::Optimizer, ::MOI.ConstraintPrimalStart,
    ::MOI.ConstraintIndex, ::Nothing)
end
# function MOIU.load(optimizer::Optimizer, ::MOI.ConstraintPrimalStart,
#                    ci::MOI.ConstraintIndex, value)
#     offset = constroffset(optimizer, ci)
#     rows = constrrows(optimizer, ci)
#     optimizer.sol.primal[offset .+ rows] .= value
# end
function MOIU.load(::Optimizer, ::MOI.ConstraintDualStart,
    ::MOI.ConstraintIndex, ::Nothing)
end
# function MOIU.load(optimizer::Optimizer, ::MOI.ConstraintDualStart,
#                    ci::MOI.ConstraintIndex, value)
#     offset = constroffset(optimizer, ci)
#     rows = constrrows(optimizer, ci)
#     optimizer.sol.primal[offset .+ rows] .= value
# end
function MOIU.load(::Optimizer, ::MOI.ObjectiveSense, ::MOI.OptimizationSense)
end
function MOIU.load(optimizer::Optimizer, ::MOI.ObjectiveFunction,
                   f::MOI.SingleVariable)
    MOIU.load(optimizer,
          MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
          MOI.ScalarAffineFunction{Float64}(f))
    return nothing
end
function MOIU.load(optimizer::Optimizer, ::MOI.ObjectiveFunction,
               f::MOI.ScalarAffineFunction)
    # @show f
    # @show optimizer.data.n
    c0 = Vector(sparsevec(variable_index_value.(f.terms), coefficient.(f.terms),
                        optimizer.data.n))
    optimizer.data.objective_constant = f.constant
    optimizer.data.c = optimizer.maxsense ? -c0 : c0
    return nothing
end

matindices(n::Integer) = (LinearIndices(tril(trues(n,n))))[findall(tril(trues(n,n)))]

function MOI.optimize!(optimizer::Optimizer)

    TimerOutputs.reset_timer!()

    @timeit "preproc" begin

    # parse options
    options = optimizer.options

    cone = optimizer.cone

    #= 
        Build linear sets
    =#

    m = optimizer.data.m #rows
    n = optimizer.data.n #cols

    objective_constant = optimizer.data.objective_constant
    c = optimizer.data.c

    # Build Prox SDP Affine Sets
    b = optimizer.data.b
    A = sparse(optimizer.data.I, optimizer.data.J, optimizer.data.V, length(b), n)
    h = optimizer.data.b
    G = sparse(optimizer.data.Ii, optimizer.data.Ji, optimizer.data.Vi, length(h), n)

    # Dimensions (of affine sets)
    n_variables = size(A)[2] # primal
    n_eqs = size(A)[1]
    n_ineqs = size(G)[1]
    aff = AffineSets(n_variables, n_eqs, n_ineqs, 0, A, G, b, h, c)

    #= 
        Build conic sets
    =#

    # Build SDP Sets
    con = ConicSets(
        SDPSet[],
        SOCSet[]
        )

    # preAt = sparse(preA')

    # create extra variables
    n_tot_variables = n_variables
    In, Jn, Vn = Int[], Int[], Float64[]

    # this way there is a single elements per column
    # because we assume VOV in SET and NOT AFF in SET
    # Asoc = preAt[:,cone.f+cone.l+1:cone.f+cone.l+cone.q]
    # A = Asoc
    # rows = rowvals(A)
    # first_ind_local = 1
    for i in eachindex(cone.soc)
        n_vars = length(cone.soc[i])
        vec_inds = cone.so[i] #get_indices_cone(A, rows, n_vars, first_ind_local)
        n_tot_variables += fix_duplicates!(vec_inds, n_tot_variables, In, Jn, Vn)
        push!(con.socone, SOCSet(vec_inds, n_vars))
        # first_ind_local += n_vars
    end

    # Asdp = preAt[:,cone.f+cone.l+cone.q+1:end]
    # A = Asdp
    # rows = rowvals(A)
    # first_ind_local = 1
    for i in eachindex(cone.sdc)
        n_vars = length(cone.sdc[i])
        vec_inds = cone.sdc[i]#get_indices_cone(A, rows, n_vars, first_ind_local)
        n_tot_variables += fix_duplicates!(vec_inds, n_tot_variables, In, Jn, Vn)
        sq_side = sympackeddim(n_vars)
        sq_len = sq_side*sq_side
        tri_len = n_vars
        mat_inds = matindices(sq_side)
        push!(con.sdpcone, SDPSet(vec_inds, mat_inds, tri_len, sq_len, sq_side))
        # first_ind_local += n_vars
    end

    optimizer.data = nothing # Allows GC to free optimizer.data before A is loaded

    #= 
        Pre-process to remove duplicates from differente cones
    =#

    for i in 1:length(con.socone)
        for j in i+1:length(con.socone)
            vec1 = con.socone[i].idx
            vec2 = con.socone[j].idx
            n_tot_variables += fix_duplicates!(vec1, vec2, n_tot_variables, In, Jn, Vn)
        end
    end

    for i in 1:length(con.sdpcone)
        for j in i+1:length(con.sdpcone)
            vec1 = con.sdpcone[i].vec_i
            vec2 = con.sdpcone[j].vec_i
            n_tot_variables += fix_duplicates!(vec1, vec2, n_tot_variables, In, Jn, Vn)
        end
    end

    for i in 1:length(con.sdpcone)
        for j in 1:length(con.socone)
            vec1 = con.sdpcone[i].vec_i
            vec2 = con.socone[j].idx
            n_tot_variables += fix_duplicates!(vec1, vec2, n_tot_variables, In, Jn, Vn)
        end
    end

    n_new_variables = n_tot_variables - n_variables

    A2 = sparse(In .- n_variables, Jn, Vn, n_new_variables, n_tot_variables)
    b2 = zeros(n_new_variables)

    append!(aff.b, b2)
    aff.A = hvcat((2,1), aff.A, spzeros(size(aff.A)[1], n_new_variables), A2)
    aff.G = hcat(aff.G, spzeros(size(aff.G)[1], n_new_variables))
    append!(aff.c, zeros(n_new_variables))
    aff.n += n_new_variables
    aff.p += n_new_variables
    aff.extra = n_new_variables

    #= 
        Solve modified problem
    =#

    end

    # warm = WarmStart()

    sol = @timeit "Main" chambolle_pock(aff, con, options)

    # sol = @enter chambolle_pock(aff, con, options)

    #= 
        Unload solution
    =#

    ret_val = sol.status
    primal = sol.primal[1:aff.n-aff.extra]
    dual_eq = sol.dual_eq[1:aff.p-aff.extra]
    slack_eq = sol.slack[1:aff.p-aff.extra]
    objval = sol.objval

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

    optimizer.sol = MOISolution(ret_val,
                                sol.status_string,
                                primal,
                                dual_eq,
                                sol.dual_in,
                                slack_eq,
                                sol.slack_in,
                                sol.primal_residual,
                                sol.dual_residual,
                                (optimizer.maxsense ? -1 : 1) * objval+objective_constant,
                                (optimizer.maxsense ? -1 : 1) * sol.dual_objval+objective_constant,
                                sol.gap,
                                sol.time,
                                sol.final_rank)
end

function get_indices_cone(A, rows, n_vars, first_ind_local)
    vec_inds = zeros(Int, n_vars)
    for (idx, col) in enumerate(first_ind_local:first_ind_local+n_vars-1)
        # columns -> pos in cone (because is tranposed)
        for j in nzrange(A, col)
            row = rows[j]
            position = idx
            var_idx = row # global
            vec_inds[position] = var_idx
        end
    end
    return vec_inds
end

function fix_duplicates!(vec1::Vector{Int}, vec2::Vector{Int}, n::Int,
                         In::Vector{Int}, Jn::Vector{Int}, Vn::Vector{Float64})
    duplicates = intersect(vec2, vec1)
    n_dups = length(duplicates)
    if n_dups == 0
        return 0
    end
    # if n_dups > 0
    #     error("SOC cones must be disjoint")
    # end
    append!(Jn, duplicates)
    append!(Jn, collect(1:n_dups) .+ n)
    append!(In, collect(1:n_dups) .+ n)
    append!(In, collect(1:n_dups) .+ n)
    append!(Vn,  ones(n_dups))
    append!(Vn, -ones(n_dups))
    cont = 1
    for i in eachindex(vec2)
        if vec2[i] == duplicates[cont]
            vec2[i] = n + cont
            cont += 1
        end
    end
    @assert cont-1 == n_dups
    return n_dups
end

function fix_duplicates!(vec::Vector{Int}, n::Int,
                         In::Vector{Int}, Jn::Vector{Int}, Vn::Vector{Float64})
    seen = Set{eltype(vec)}()
    dups = Vector{eltype(vec)}()
    new_vars = 0
    for i in eachindex(vec)
        x = vec[i]
        if in(x, seen)
            push!(dups, x)
            new_vars += 1
            vec[i] = new_vars + n
        else
            push!(seen, x)
        end
    end
    n_dups = length(dups)
    if n_dups == 0
        return 0
    end
    append!(Jn, dups)
    append!(Jn, collect(1:n_dups) .+ n)
    append!(In, collect(1:n_dups) .+ n)
    append!(In, collect(1:n_dups) .+ n)
    append!(Vn,  ones(n_dups))
    append!(Vn, -ones(n_dups))
    return n_dups
end

function ivech!(out::AbstractMatrix{T}, v::AbstractVector{T}) where T
    n = sympackeddim(length(v))
    n1, n2 = size(out)
    @assert n == n1 == n2
    c = 0
    for j in 1:n, i in 1:j
        c += 1
        out[i,j] = v[c]
    end
    return out
end
function ivech(v::AbstractVector{T}) where T
    n = sympackeddim(length(v))
    out = zeros(n, n)
    ivech!(out, v)
    
    return out
end

ivec(X) = Matrix(Symmetric(ivech(X),:U))

#=
    Status
=#

function MOI.get(optimizer::Optimizer, ::MOI.SolveTime)
    return optimizer.sol.time
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return optimizer.sol.raw_status
end

# Implements getter for result value and statuses
# ProxSDP returns one of the following integers:
# 0 - MOI.OPTIMIZE_NOT_CALLED
# 1 - MOI.OPTIMAL
# 2 - MOI.TIME_LIMIT
# 3 - MOI.ITERATION_LIMIT
# 4 - MOI.INFEASIBLE_OR_UNBOUNDED
function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    s = optimizer.sol.ret_val
    @assert 0 <= s <= 4
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
    end
end

function MOI.get(optimizer::Optimizer, ::MOI.ObjectiveValue)
    value = optimizer.sol.objval
    return value
end
function MOI.get(optimizer::Optimizer, ::MOI.DualObjectiveValue)
    value = optimizer.sol.dual_objval
    return value
end

function MOI.get(optimizer::Optimizer, ::MOI.PrimalStatus)
    s = optimizer.sol.ret_val
    if s == 1
        MOI.FEASIBLE_POINT
    else
        MOI.INFEASIBLE_POINT
    end
end
function MOI.get(optimizer::Optimizer, ::MOI.DualStatus)
    s = optimizer.sol.ret_val
    if s == 1
        MOI.FEASIBLE_POINT
    else
        MOI.INFEASIBLE_POINT
    end
end

#=
    Solution
=#

function MOI.get(optimizer::Optimizer, ::MOI.VariablePrimal, vi::VI)
    optimizer.sol.primal[vi.value]
end
function MOI.get(optimizer::Optimizer, a::MOI.VariablePrimal, vi::Vector{VI})
    return MOI.get.(optimizer, a, vi)
end

# _unshift(optimizer::Optimizer, offset, value, s) = value
# _unshift(optimizer::Optimizer, offset, value, s::Type{<:MOI.AbstractScalarSet}) = value + optimizer.cone.setconstant[offset]

# reorderval(val, s) = val
# function reorderval(val, ::Type{MOI.PositiveSemidefiniteConeTriangle})
#     sympackedLtoU(val)
# end

function MOI.get(optimizer::Optimizer, ::MOI.ConstraintPrimal,
                 ci::CI{<:MOI.AbstractFunction, MOI.Zeros})
    cone = optimizer.cone
    inds = cone.eq_start-1 .+ 1:cone.eq_len
    return optimizer.sol.slack_eq[inds]
end
function MOI.get(optimizer::Optimizer, ::MOI.ConstraintPrimal,
    ci::CI{<:MOI.AbstractFunction, MOI.Nonpositives})
    cone = optimizer.cone
    inds = cone.in_start-1 .+ 1:cone.in_len
    return optimizer.sol.slack_in[inds]
end

function MOI.get(optimizer::Optimizer, ::MOI.ConstraintPrimal,
        ci::CI{MOI.VectorOfVariables, MOI.PositiveSemidefiniteConeTriangle})
    optimizer.sol.primal[optimizer.cone.sdc[ci.value]]
end
function MOI.get(optimizer::Optimizer, ::MOI.ConstraintPrimal,
    ci::CI{MOI.VectorOfVariables, MOI.SecondOrderCone})
    optimizer.sol.primal[optimizer.cone.soc[ci.value]]
end

function MOI.get(optimizer::Optimizer, ::MOI.ConstraintDual,
    ci::CI{<:MOI.AbstractFunction, MOI.Zeros})
    cone = optimizer.cone
    inds = cone.eq_start-1 .+ 1:cone.eq_len
    return optimizer.sol.dual_eq[inds]
end
function MOI.get(optimizer::Optimizer, ::MOI.ConstraintDual,
ci::CI{<:MOI.AbstractFunction, MOI.Nonpositives})
    cone = optimizer.cone
    inds = cone.in_start-1 .+ 1:cone.in_len
    return optimizer.sol.dual_in[inds]
end

function MOI.get(optimizer::Optimizer, ::MOI.ConstraintDual,
    ci::CI{<:MOI.AbstractFunction, S}) where S <: MOI.PositiveSemidefiniteConeTriangle
    error("ProxSDP does not return duals for nonlinear Conic constraints."*
          " Only linear constraints (equalities and inequalities) can be queried.")
end

function MOI.get(optimizer::Optimizer, ::MOI.ResultCount)
    if MOI.get(optimizer, MOI.TerminationStatus()) == MOI.INFEASIBLE_OR_UNBOUNDED
        return 0
    else
        return 1
    end
end