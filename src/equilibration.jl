function equilibrate!(M, aff, opt)
    max_iters=opt.equilibration_iters
    lb = opt.equilibration_lb
    ub = opt.equilibration_ub

    @timeit "eq init" begin
        α = (aff.n / (aff.m + aff.p)) ^ .25
        β = ((aff.m + aff.p) / aff.n ) ^ .25
        α2, β2 = α^2, β^2
        γ = .1

        u, v           = zeros(aff.m + aff.p), zeros(aff.n)
        u_, v_         = zeros(aff.m + aff.p), zeros(aff.n)
        u_grad, v_grad = zeros(aff.m + aff.p), zeros(aff.n)
        row_norms, col_norms = zeros(aff.m + aff.p), zeros(aff.n)
        E = LinearAlgebra.Diagonal(u)
        D = LinearAlgebra.Diagonal(v)
        M_ = copy(M)

        rows_M_ = SparseArrays.rowvals(M_)
    end

    for iter in 1:max_iters
        @timeit "update diags" begin
            E.diag .= exp.(u)
            D.diag .= exp.(v)
        end
        @timeit "M_" begin 
            LinearAlgebra.mul!(M_, M, D)
            LinearAlgebra.mul!(M_, E, M_)
        end

        step_size = 2. / (γ * (iter + 1.))

        # u gradient step
        @timeit "norms" begin
            fill!(row_norms, 0.0)
            fill!(col_norms, 0.0)
            for col in 1:aff.n
                for line in SparseArrays.nzrange(M_, col)
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
            box_project!(v, 0., ub)
        end
        # Update averages.
        @timeit "update" begin
            @. u_ = 2 * u / (iter + 2) + iter * u_ / (iter + 2)
            @. v_ = 2 * v / (iter + 2) + iter * v_ / (iter + 2)
        end
    end

    @timeit "update diags" begin 
        E.diag .= exp.(u_)
        D.diag .= exp.(v_)
    end

    return E, D
end

function box_project!(y, lb, ub)
    y .= min.(ub, max.(y, lb))
    nothing
end