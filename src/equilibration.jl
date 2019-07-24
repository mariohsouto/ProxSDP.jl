
using LinearAlgebra

function equilibrate!(M::SparseMatrixCSC{Float64,Int64}, aff::AffineSets, opt::Options, cones::ConicSets)

    @timeit "eq init" begin
        max_iters = opt.equilibration_iters
        lb        = opt.equilibration_lb
        ub        = opt.equilibration_ub

        @show α = (aff.n / (aff.m + aff.p)) ^ .25
        @show β = ((aff.m + aff.p) / aff.n ) ^ .25
        α2, β2 = α^2, β^2
        γ = 1e-14

        u, v           = zeros(aff.m + aff.p), zeros(aff.n)
        u_next, v_next = zeros(aff.m + aff.p), zeros(aff.n)
        u_, v_         = zeros(aff.m + aff.p), zeros(aff.n)
        u_grad, v_grad = zeros(aff.m + aff.p), zeros(aff.n)
        row_norms, col_norms = zeros(aff.m + aff.p), zeros(aff.n)
        E = Diagonal(u)
        D = Diagonal(v)
        M_ = copy(M)
        rows_M_ = rowvals(M_)
    end

    max_iters = 300

    lb, ub = -1000., 1000.

    for iter in 1:max_iters
        @timeit "update diags" begin
            E.diag .= exp.(u)
            D.diag .= exp.(v)
        end
        @timeit "M_" begin 
            mul!(M_, M, D)
            mul!(M_, E, M_)
        end

        # Compute row and column norms
        @timeit "norms" begin
            fill!(row_norms, 0.0)
            fill!(col_norms, 0.0)
            for col in 1:aff.n
                for line in nzrange(M_, col)
                    row_norms[rows_M_[line]] += abs2(M_[rows_M_[line], col])
                    col_norms[col] += abs2(M_[rows_M_[line], col])
                end
            end
        end
        @show (minimum(row_norms), maximum(row_norms))
        @show (minimum(col_norms), maximum(col_norms))

        u .= log.(u)
        v .= log.(v)

        # Compute gradient 
        @timeit "grad" begin
            @. u_grad = row_norms - α2 + γ * u
            @. v_grad = col_norms - β2 + γ * v
        end

        # Armijo linesearch
        line_search_iter = linesearch_equil!(u, v, u_next, v_next, u_grad, v_grad, α2, β2, γ, M, rows_M_, aff.n, cones, lb, ub)
        if line_search_iter > 100
            break
        end
    end

    @show (minimum(row_norms), maximum(row_norms))
    @show (minimum(col_norms), maximum(col_norms))

    @timeit "update diags" begin 
        E.diag .= exp.(u)
        D.diag .= exp.(v)
    end

    return E, D
end

function box_project!(y::Vector{Float64}, lb::Float64, ub::Float64)
    y .= min.(ub, max.(y, lb))
    return nothing
end

function evaluate_equil(u::Vector{Float64}, v::Vector{Float64}, α2::Float64, β2::Float64, γ::Float64, M::SparseMatrixCSC{Float64,Int64}, rows_M_, n::Int64)::Float64

    obj_equil = 0.

    for col in 1:n
        for line in nzrange(M, col)
            obj_equil += .5 * abs2(M[rows_M_[line], col]) * exp(2 * u[rows_M_[line]]) * exp(2 * v[col])
        end
    end

    obj_equil -= α2 * sum(u)
    obj_equil -= β2 * sum(v)
    obj_equil += (γ / 2.) * (norm(u, 2)^2 + norm(v, 2)^2)

    return obj_equil
end

function linesearch_equil!(u::Vector{Float64}, v::Vector{Float64}, u_next::Vector{Float64}, v_next::Vector{Float64}, u_grad::Vector{Float64}, v_grad::Vector{Float64}, α2::Float64, β2::Float64, γ::Float64, M::SparseMatrixCSC{Float64,Int64}, rows_M_, n::Int64, cones, lb, ub)

    step_size = 1.
    u_next .= u - step_size * u_grad
    v_next .= v - step_size * v_grad

    @timeit "proj" begin
        # Projection of u onto box [lb, ub]
        box_project!(u_next, lb, ub)

        for sdp in cones.sdpcone
            sum_v = sum(v_next[sdp.vec_i])
            v_next[sdp.vec_i] .= sum_v / sdp.tri_len
        end

        # Projection of v onto box [lb, ub]
        box_project!(v_next, 0., 1.)
    end

    g_u = (u - u_next) / step_size
    g_v = (v - v_next) / step_size
    grad_norm2 = norm(g_u, 2)^2 + norm(g_v, 2)^2
    aux_linesearch = - step_size *(dot(g_u, u_grad) + dot(g_v, v_grad) - .5 * grad_norm2)

    line_search_iter = 0
    while evaluate_equil(u_next, v_next, α2, β2, γ, M, rows_M_, n) > evaluate_equil(u, v, α2, β2, γ, M, rows_M_, n) + aux_linesearch
        line_search_iter += 1
        step_size *= .8
        u_next .= u - step_size * u_grad
        v_next .= v - step_size * v_grad

        @timeit "proj" begin
            # Projection of u onto box [lb, ub]
            box_project!(u_next, lb, ub)

            for sdp in cones.sdpcone
                sum_v = sum(v_next[sdp.vec_i])
                v_next[sdp.vec_i] .= sum_v / sdp.tri_len
            end

            # Projection of v onto box [lb, ub]
            box_project!(v_next, 0., ub)
        end

        g_u = (u - u_next) / step_size
        g_v = (v - v_next) / step_size
        grad_norm2 = norm(g_u, 2)^2 + norm(g_v, 2)^2
        aux_linesearch = - step_size * (dot(g_u, u_grad) + dot(g_v, v_grad) - .5 * grad_norm2)

        if line_search_iter > 100
            break
        end
    end
    copyto!(u, u_next)
    copyto!(v, v_next)

    return line_search_iter
end