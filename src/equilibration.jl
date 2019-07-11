
using LinearAlgebra

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
        E = Diagonal(u)
        D = Diagonal(v)
        M_ = copy(M)

        rows_M_ = rowvals(M_)
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

        step_size = 2. / (γ * (iter + 1.))

        # u gradient step
        @timeit "row norms" begin
            fill!(row_norms, 0.0)
            for col in 1:aff.n
                for line in nzrange(M_, col)
                    row_norms[rows_M_[line]] += abs2(M_[rows_M_[line], col])
                end
            end
        end
        @timeit "u grad" begin
            @. u_grad = row_norms - α2 + γ * u
        end
        @timeit "u proj " begin
            @. u -= step_size * u_grad
            box_project!(u, lb, ub)
        end

        # v grad estimate
        @timeit "col norms" begin
            fill!(col_norms, 0.0)
            for col in 1:aff.n
                for line in nzrange(M_, col)
                    col_norms[col] += abs2(M_[rows_M_[line], col])
                end
            end
        end
        @timeit "v grad" begin
            @. v_grad = col_norms - β2 + γ * v
        end
        @timeit "v proj" begin
            @. v -= step_size * v_grad
            v .= sum(v) / aff.n
            box_project!(v, 0., ub)
        end
        
        # Update averages.
        @timeit "u update" begin
            @. u_ = 2 * u / (iter + 2) + iter * u_ / (iter + 2)
        end
        @timeit "v update" begin
            @. v_ = 2 * v / (iter + 2) + iter * v_ / (iter + 2)
        end
    end

    @timeit "update diag E" E[diagind(E)] .= exp.(u_)
    @timeit "update diag D" D[diagind(D)] .= exp.(v_)

    return E, D
end

function box_project!(y, lb, ub)
    y .= min.(ub, max.(y, lb))
    nothing
end