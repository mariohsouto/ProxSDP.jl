
using LinearAlgebra

function equilibrate!(M, aff, opt)
    max_iters = opt.equilibration_iters
    lb = opt.equilibration_lb
    ub = opt.equilibration_ub

    @timeit "eq init" begin
        α = (aff.n / (aff.m + aff.p)) ^ .25
        β = ((aff.m + aff.p) / aff.n ) ^ .25
        α2, β2 = α^2, β^2
        γ = 1e-1

        u, v           = zeros(aff.m + aff.p), zeros(aff.n)
        u_grad, v_grad = zeros(aff.m + aff.p), zeros(aff.n)
        row_norms, col_norms = zeros(aff.m + aff.p), zeros(aff.n)
        E = Diagonal(zeros(aff.m + aff.p))
        D = Diagonal(zeros(aff.n))
        M_ = copy(M)
        rows_M_ = rowvals(M_)
    end

    max_iters = 300
    lb, ub = -100., 100.

    for iter in 1:max_iters
        @timeit "update diags" begin
            E.diag .= exp.(u)
            D.diag .= exp.(v)
        end
        @timeit "M_" begin 
            mul!(M_, M, D)
            mul!(M_, E, M_)
        end

        step_size = 2. / (γ * (iter + 1.))

        # u gradient step
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
        @timeit "proj " begin
            # u
            @. u -= step_size * u_grad
            box_project!(u, lb, ub)
            # v
            @. v -= step_size * v_grad
            sum_v = sum(v)
            v .= sum_v / aff.n
            box_project!(v, lb, ub)
        end
    end

    return E, D
end

function box_project!(y, lb, ub)
    y .= min.(ub, max.(y, lb))
    return nothing
end