const ZeroCones = Union{MOI.EqualTo, MOI.Zeros}
const LPCones = Union{MOI.GreaterThan, MOI.LessThan, MOI.Nonnegatives, MOI.Nonpositives}

_dim(s::MOI.AbstractScalarSet) = 1
_dim(s::MOI.AbstractVectorSet) = MOI.dimension(s)

mutable struct Cone
    f::Int # number of linear equality constraints
    fcur::Int
    l::Int # length of LP cone
    lcur::Int
    q::Int # length of SOC cone
    qcur::Int
    qa::Vector{Int} # array of second-order cone constraints
    s::Int # length of SD cone
    scur::Int
    sa::Vector{Int} # array of semi-definite constraints
    ep::Int # number of primal exponential cone triples
    epcur::Int
    ed::Int # number of dual exponential cone triples
    p::Vector{Float64} # array of power cone params
    function Cone()
        new(0, 0, 0, 0,
            0, 0, Int[],
            0, 0, Int[],
            0, 0, 0, Float64[])
    end
end

# Computes cone dimensions
constrcall(cone::Cone, ci, f, s::ZeroCones) = cone.f += _dim(s)
constrcall(cone::Cone, ci, f, s::LPCones) = cone.l += _dim(s)
function constrcall(cone::Cone, ci, f, s::MOI.SecondOrderCone)
    push!(cone.qa, s.dimension)
    cone.q += _dim(s)
end
function constrcall(cone::Cone, ci, f, s::MOI.PositiveSemidefiniteConeTriangle)
    push!(cone.sa, s.dimension)
    cone.s += _dim(s)
end
constrcall(cone::Cone, ci, f, s::MOI.ExponentialCone) = cone.ep += 1

# Fill constrmap
function constrcall(cone::Cone, constrmap::Dict, ci, f, s::ZeroCones)
    constrmap[ci.value] = cone.fcur
    cone.fcur += _dim(s)
end
function constrcall(cone::Cone, constrmap::Dict, ci, f, s::LPCones)
    constrmap[ci.value] = cone.f + cone.lcur
    cone.lcur += _dim(s)
end
function constrcall(cone::Cone, constrmap::Dict, ci, f, s::MOI.SecondOrderCone)
    constrmap[ci.value] = cone.f + cone.l + cone.qcur
    cone.qcur += _dim(s)
end
function constrcall(cone::Cone, constrmap::Dict, ci, f, s::MOI.PositiveSemidefiniteConeTriangle)
    constrmap[ci.value] = cone.f + cone.l + cone.q + cone.scur
    cone.scur += _dim(s)
end
function constrcall(cone::Cone, constrmap::Dict, ci, f, s::MOI.ExponentialCone)
    constrmap[ci.value] = cone.f + cone.l + cone.q + cone.s + cone.epcur
    cone.epcur += _dim(s)
end

# Vectorized length for matrix dimension n
sympackedlen(n) = (n*(n+1)) >> 1
trimap(i::Integer, j::Integer) = i < j ? trimap(j, i) : div((i-1)*i, 2) + j
trimapL(i::Integer, j::Integer, n::Integer) = i < j ? trimapL(j, i, n) : i + div((2n-j) * (j-1), 2)
function _sympackedto(x, n, mapfrom, mapto)
    @assert length(x) == sympackedlen(n)
    y = similar(x)
    for i in 1:n, j in 1:i
        y[mapto(i, j)] = x[mapfrom(i, j)]
    end
    y
end
sympackedLtoU(x, n=sympackeddim(length(x))) = _sympackedto(x, n, (i, j) -> trimapL(i, j, n), trimap)
sympackedUtoL(x, n=sympackeddim(length(x))) = _sympackedto(x, n, trimap, (i, j) -> trimapL(i, j, n))

function sympackedUtoLidx(x::AbstractVector{<:Integer}, n)
    y = similar(x)
    map = sympackedLtoU(1:sympackedlen(n), n)
    for i in eachindex(y)
        y[i] = map[x[i]]
    end
    y
end


# Build constraint matrix
scalecoef(rows, coef, minus, s, rev) = minus ? -coef : coef
scalecoef(rows, coef, minus, s::Union{MOI.LessThan, MOI.Nonpositives}, rev) = minus ? coef : -coef
function scalecoef(rows, coef, minus, s::MOI.PositiveSemidefiniteConeTriangle, rev)
    scaling = minus ? -1 : 1
    scaling2 = rev ? scaling / sqrt(2) : scaling * sqrt(2)
    output = copy(coef)
    diagidx = IntSet()
    for i in 1:s.dimension
        push!(diagidx, trimap(i, i))
    end
    for i in 1:length(output)
        if rows[i] in diagidx
            output[i] *= scaling
            # output[i] *= 1.0
        else
            # output[i] *= scaling2
            output[i] *= 2.0
        end
    end
    output
end
_varmap(varmap, f) = map(vi -> varmap[vi], f.variables)
_constant(s::MOI.EqualTo) = s.value
_constant(s::MOI.GreaterThan) = s.lower
_constant(s::MOI.LessThan) = s.upper
constrrows(::MOI.AbstractScalarSet) = 1
constrrows(s::MOI.AbstractVectorSet) = 1:MOI.dimension(s)
constrcall(I, J, V, b, varmap, constrmap, ci, f::MOI.SingleVariable, s) = constrcall(I, J, V, b, varmap, constrmap, ci, MOI.ScalarAffineFunction{Float64}(f), s)
function constrcall(I, J, V, b, varmap::Dict, constrmap::Dict, ci, f::MOI.ScalarAffineFunction, s)
    a = sparsevec(_varmap(varmap, f), f.coefficients)
    # sparsevec combines duplicates with + but does not remove zeros created so we call dropzeros!
    dropzeros!(a)
    offset = constrmap[ci.value]
    row = constrrows(s)
    i = offset + row
    # The ProxSDP format is b - Ax ∈ cone
    # so minus=false for b and minus=true for A
    constant = f.constant - _constant(s)
    b[i] = scalecoef(row, constant, false, s, false)
    append!(I, fill(i, length(a.nzind)))
    append!(J, a.nzind)
    append!(V, scalecoef(row, a.nzval, true, s, false))
end
constrcall(I, J, V, b, varmap, constrmap, ci, f::MOI.VectorOfVariables, s) = constrcall(I, J, V, b, varmap, constrmap, ci, MOI.VectorAffineFunction{Float64}(f), s)
orderval(val, s) = val
function orderval(val, s::MOI.PositiveSemidefiniteConeTriangle)
    sympackedUtoL(val, s.dimension)
end
orderidx(idx, s) = idx
function orderidx(idx, s::MOI.PositiveSemidefiniteConeTriangle)
    sympackedUtoLidx(idx, s.dimension)
end
function constrcall(I, J, V, b, varmap::Dict, constrmap::Dict, ci, f::MOI.VectorAffineFunction, s)
    A = sparse(f.outputindex, _varmap(varmap, f), f.coefficients)
    # sparse combines duplicates with + but does not remove zeros created so we call dropzeros!
    dropzeros!(A)
    colval = zeros(Int, length(A.rowval))
    for col in 1:A.n
        colval[A.colptr[col]:(A.colptr[col+1]-1)] = col
    end
    @assert !any(iszero.(colval))
    offset = constrmap[ci.value]
    rows = constrrows(s)
    i = offset + rows
    # The ProxSDP format is b - Ax ∈ cone
    # so minus=false for b and minus=true for A
    b[i] = scalecoef(rows, orderval(f.constant, s), false, s, false)
    append!(I, offset + orderidx(A.rowval, s))
    append!(J, colval)
    append!(V, scalecoef(A.rowval, A.nzval, true, s, false))
end

function constrcall(arg::Tuple, constrs::Vector)
    for constr in constrs
        constrcall(arg..., constr...)
    end
end
function MOI.optimize!(instance::ProxSDPSolverInstance)
    cone = Cone()
    MOIU.broadcastcall(constrs -> constrcall((cone,), constrs), instance.data)
    instance.constrmap = Dict{UInt64, Int}()
    MOIU.broadcastcall(constrs -> constrcall((cone, instance.constrmap), constrs), instance.data)
    vcur = 0
    instance.varmap = Dict{VI, Int}()
    # @show length(MOI.get(instance.data, MOI.ListOfVariableIndices()))
    for vi in MOI.get(instance.data, MOI.ListOfVariableIndices())
        vcur += 1
        instance.varmap[vi] = vcur
    end
    @assert vcur == MOI.get(instance.data, MOI.NumberOfVariables())
    m = cone.f + cone.l + cone.q + cone.s + 3cone.ep + cone.ed
    n = vcur
    I = Int[]
    J = Int[]
    V = Float64[]
    preb = zeros(m)
    MOIU.broadcastcall(constrs -> constrcall((I, J, V, preb, instance.varmap, instance.constrmap), constrs), instance.data)
    rows = maximum(I)
    cols = length(MOI.get(instance.data, MOI.ListOfVariableIndices()))
    preA = sparse(I, J, V, rows, cols)
    f = MOI.get(instance.data, MOI.ObjectiveFunction())
    c0 = full(sparsevec(_varmap(instance.varmap, f), f.coefficients, n))
    c = MOI.get(instance.data, MOI.ObjectiveSense()) == MOI.MaxSense ? -c0 : c0
    # @show instance.varmap
    # # @show cone
    # # @show preA
    # @show preb
    # @show c
    # # @show m
    # # @show n
    # @show full(preA)
    # # @show cone.f
    # @show full(preA[:, 1:cone.f])
    # @show cone.sa
    # BLAS.set_num_threads(2)
    TimerOutputs.reset_timer!()
    A = preA[1:cone.f,:]
    G = preA[cone.f+1:cone.f+cone.l,:]
    b = preb[1:cone.f]
    h = preb[cone.f+1:cone.f+cone.l]
    # aff = AffineSets(A, G, b, h, c)

    Asdp = preA[cone.f+cone.l+1:end,:]
    indices_sdp = Asdp.rowval

    aff = AffineSets(A, G, b, h, c)
    con = ConicSets(Tuple{Vector{Int},Vector{Int}}[(sortperm(indices_sdp), matindices(sympackeddim(length(indices_sdp))) )])
    # @show aff


    # @show size(A)
    # @show size(preA)
    # @show cone.sa

    # Apython = readcsv("/Users/mariosouto/Dropbox/proxsdp/A.csv")
    # bpython = readcsv("/Users/mariosouto/Dropbox/proxsdp/b.csv")
    # @show norm(vec(Apython - A))
    # @show norm(vec(bpython - b))
    # writecsv("/Users/mariosouto/Dropbox/proxsdp/b_jl.csv", b)
    # writecsv("/Users/mariosouto/Dropbox/proxsdp/c_jl.csv", c)

    dims = Dims(sympackeddim(size(A)[2]), size(A)[1], size(G)[1])
    sol = @timeit "Main" chambolle_pock(aff, con, dims)
    instance.ret_val = sol.status
    instance.primal = sol.primal
    instance.dual = sol.dual
    instance.slack = sol.slack
    instance.objval = sol.objval + f.constant

    if true
        TimerOutputs.print_timer(TimerOutputs.DEFAULT_TIMER)
        print("\n")
        TimerOutputs.print_timer(TimerOutputs.flatten(TimerOutputs.DEFAULT_TIMER))
        print("\n")
        f = open("time.log","w")
        TimerOutputs.print_timer(f,TimerOutputs.DEFAULT_TIMER)
        print(f,"\n")
        TimerOutputs.print_timer(f,TimerOutputs.flatten(TimerOutputs.DEFAULT_TIMER))
        print(f,"\n")
        close(f)
    end
end

matindices(n::Integer) = find(tril(trues(n,n)))
sympackeddim(n) = div(isqrt(1+8n) - 1, 2)
