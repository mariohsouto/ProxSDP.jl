module ProxSDP

using MathOptInterface, PyCall, TimerOutputs
@pyimport scipy.linalg as sp

include("mathoptinterface.jl")

immutable Dims
    m::Int
    n::Int
end

type AffineSets
    A::AbstractMatrix
    G::AbstractMatrix
    b::AbstractVector
    h::AbstractVector
    c::AbstractVector

    sdpcone::Vector{Tuple{Vector{Int},Vector{Int}}}
end

struct CPResult
    status::Int
    primal::Vector{Float64}
    dual::Vector{Float64}
    slack::Vector{Float64}
    primal_residual::Float64
    dual_residual::Float64
    objval::Float64
end

function chambolle_pock(
    affine_sets, dims, verbose=true, max_iter=Int(1e+3), primal_tol=1e-4, dual_tol=1e-4
)
    converged = false
    tic()
    @timeit "print" if verbose
        println(" Starting Chambolle-Pock algorithm")
        println("-----------------------------------------------------------")
        println("|  iter  | comb. res | prim. res |  dual res  |    rho    |")
        println("-----------------------------------------------------------")
    end

    @timeit "Init" begin
    # Given
    m, n = dims.m, dims.n

    # Initialization
    best_comb_residual = Inf
    print_iter, converged, status = 0, false, false
    comb_residual, dual_residual, primal_residual = Float64[], Float64[], Float64[]
    sizehint!(comb_residual, max_iter)
    sizehint!(dual_residual, max_iter)
    sizehint!(primal_residual, max_iter)
    # x, x_old = zeros(n^2), zeros(n^2)
    x, x_old = zeros(n*(n+1)/2), zeros(n*(n+1)/2)
    u, u_old = zeros(m), zeros(m)

    # Step size
    rho = 0.99 / svds(affine_sets.A, nsv=1)[1][:S][1]

    # Diagonal scaling
    K = affine_sets.A
    Kt = affine_sets.A'
    div = vec(sum(abs.(K), 1))
    div[find(x-> x == 0.0, div)] = 1.0
    T = sparse(diagm(1.0 ./ div))
    div = vec(sum(abs.(K), 2))
    div[find(x-> x == 0.0, div)] = 1.0
    S = sparse(diagm(1.0 ./ div))

    # Cache matrix multiplications
    TKt = T * affine_sets.A'
    SK = S * K
    Sb = S * affine_sets.b
    end

    # Fixed-point loop
    @timeit "CP loop" for k in 1:max_iter
        # Projetion onto sdp cone
        @timeit "primal" begin
            x = sdp_cone_projection(x - rho * TKt * u - affine_sets.c / rho, dims, affine_sets)::Vector{Float64}
        end
        # Projection onto the affine set
        @timeit "dual" begin
            u += rho * (SK * (2.0 * x - x_old) - Sb)::Vector{Float64}
        end

        # Compute residuals
        @timeit "logging" begin
            push!(primal_residual, norm(x - x_old))
            push!(dual_residual, rho * norm(u - u_old))
            push!(comb_residual, primal_residual[end] + dual_residual[end])

            if mod(k, 100) == 0
                print_progress(k, primal_residual[end], dual_residual[end])
            end
        end

        # Keep track
        @timeit "tracking" begin
            copy!(x_old, x)
            copy!(u_old, u)
        end

        # Check convergence
        if primal_residual[end] < primal_tol && dual_residual[end] < dual_tol
            converged = true
            break
        end
    end
    time = toq()
    println("Time = $time")

    ret = CPResult(Int(converged), x, u, 0.0*x, primal_residual[end], dual_residual[end], evaluate(x, affine_sets))
    return ret
    # return converged, evaluate(x, affine_sets), x, primal_residual[end], dual_residual[end]
end

function evaluate(x, affine_sets)
    return dot(affine_sets.c, x)
end

function sdp_cone_projection(v::Vector{Float64}, dims::Dims, aff)::Vector{Float64}

    M = zeros(dims.n, dims.n)
    iv = aff.sdpcone[1][1]
    im = aff.sdpcone[1][2]
    for i in eachindex(iv)
        M[im[i]] = v[iv[i]]
    end
    X = Symmetric(M,:U)
    # @show X

    # D, V = @timeit "py" sp.eigh(reshape(v, (dims.n, dims.n)))
    # D = diagm(max.(D, 0.0))
    # return vec(V * D * V')
    # X = @timeit "reshape" Symmetric(reshape(v, (dims.n, dims.n)))
    fact = @timeit "eig" eigfact!(X)
    D = diagm(max.(fact[:values], 0.0))
    M2 = fact[:vectors] * D * fact[:vectors]'

    for i in eachindex(iv)
        v[iv[i]] = M2[im[i]]
    end

    return v
end

function print_progress(k, primal_res, dual_res)
    @printf("%d %.4f %.4f %.4f \n", k, primal_res + dual_res, primal_res, dual_res)
end

function load_data(path::String)
    A = sparse(readcsv(path*"/A.csv"))
    b = vec(readcsv(path*"/b.csv"))
    c = vec(readcsv(path*"/C.csv"))
    return AffineSets(A, A, b, b, c), Dims(size(A)[1], sqrt(size(A)[2]))
end

function runpsdp(path::String)
    # BLAS.set_num_threads(2)
    TimerOutputs.reset_timer!()
    @timeit "load data" begin
        aff, dims = load_data(path)
    end
    @timeit "Main" begin
        ret = chambolle_pock(aff, dims)
    end

    TimerOutputs.print_timer(TimerOutputs.DEFAULT_TIMER)
    TimerOutputs.print_timer(TimerOutputs.flatten(TimerOutputs.DEFAULT_TIMER))
    f = open("time.log","w")
    TimerOutputs.print_timer(f,TimerOutputs.DEFAULT_TIMER)
    TimerOutputs.print_timer(f,TimerOutputs.flatten(TimerOutputs.DEFAULT_TIMER))
    close(f)
    return ret
end


end
