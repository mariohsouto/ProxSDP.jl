
using LinearAlgebra

function equilibrate!(M, aff::AffineSets, opt::Options)
    max_iters=opt.equilibration_iters
    lb = opt.equilibration_lb
    ub = opt.equilibration_ub
    γ = opt.equil_regularization

    @timeit "eq init" begin
        α = (aff.n / (aff.m + aff.p)) ^ .25
        β = ((aff.m + aff.p) / aff.n ) ^ .25
        α2, β2 = α^2, β^2

        u, v           = zeros(aff.m + aff.p), zeros(aff.n)
        u_next, v_next = zeros(aff.m + aff.p), zeros(aff.n)
        u_grad, v_grad = zeros(aff.m + aff.p), zeros(aff.n)
        row_norms, col_norms = zeros(aff.m + aff.p), zeros(aff.n)
        E = Diagonal(u)
        D = Diagonal(v)
        Einv = Diagonal(u)
        M_ = copy(M)
        rows_M_ = rowvals(M_)

        step_size = 1.
        previous_eval = Inf
        tol_equil = 1e-6
        linesearch_max_iter = 100
    end

    for iter in 1:max_iters
        @timeit "update diags" begin
            E.diag .= exp.(u)
            D.diag .= exp.(v)
        end
        @timeit "M_" begin 
            mul!(M_, M, D)
            mul!(M_, E, M_)
        end

        u .= log.(u)
        v .= log.(v)

        # Compute gradients
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
        @timeit "grad" begin
            @. u_grad = row_norms - α2 + γ * u
            @. v_grad = col_norms - β2 + γ * v
        end

        # Linesearch 
        @timeit "linesearch equil " begin
            for _iter in 1:linesearch_max_iter
                # Projected gradient step
                @timeit "proj grad equil " begin
                    # u update
                    @. u_next .= u - step_size * u_grad
                    # v update
                    @. v_next .= v - step_size * v_grad
                    sum_v = sum(v_next)
                    v_next .= sum_v / aff.n
                    # Box projection
                    box_project!(u_next, lb, ub)
                    box_project!(v_next, lb, ub)
                end

                # Evaluate incumbent
                next_eval = evaluate_equil(u_next, v_next, α2, β2, γ, M, rows_M_, aff.n)

                if next_eval <= previous_eval
                    # Check convergence
                    # if (norm(u - u_next) + norm(v - v_next)) / step_size <= tol_equil
                    if previous_eval - next_eval <= tol_equil
                        @show _iter, iter
                        @timeit "update diags" begin 
                            E.diag .= exp.(u_next)
                            D.diag .= exp.(v_next)
                            Einv.diag .= 1. ./ u_next
                        end
                        return E, D, Einv
                    end

                    # Accept update
                    copyto!(u, u_next)
                    copyto!(v, v_next)
                    previous_eval = next_eval

                    # Increase stepsize
                    step_size *= 1.2

                    # @show _iter, previous_eval
                    break
                else
                    # Decrease stepsize
                    step_size *= .5
                end
            end
        end
    end

    @timeit "update diags" begin 
        E.diag .= exp.(u)
        D.diag .= exp.(v)
        Einv.diag .= 1. ./ u
    end

    return E, D, Einv
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

function box_project!(y, lb, ub)
    y .= min.(ub, max.(y, lb))
    nothing
end