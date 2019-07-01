
using LinearAlgebra

function equilibrate!(M, aff, max_iters=100, lb=-10., ub=10.)

    α = (aff.n / (aff.m + aff.p)) ^ .25
    β = ((aff.m + aff.p) / aff.n ) ^ .25
    γ = .1

    u, v           = zeros(aff.m + aff.p), zeros(aff.n)
    u_, v_         = zeros(aff.m + aff.p), zeros(aff.n)
    u_grad, v_grad = zeros(aff.m + aff.p), zeros(aff.n)
    row_norms, col_norms = zeros(aff.m + aff.p), zeros(aff.n)

    E = Diagonal(u)
    D = Diagonal(v)

    (I, J, V) = findnz(M)
    cols4row, rows4col = Dict(i => [] for i in 1:aff.m + aff.p), Dict(j => [] for j in 1:aff.n)
    for idx in 1:length(V)
        append!(cols4row[I[idx]], J[idx])
        append!(rows4col[J[idx]], I[idx])
    end

    M_ = copy(M)

    for iter in 1:max_iters
        @timeit "update diag E" E[diagind(E)] .= exp.(u)
        @timeit "update diag D" D[diagind(D)] .= exp.(v)
        @timeit "M_" M_ .= E * M * D

        step_size = 2. / (γ * (iter + 1.))

        # u gradient step
        @timeit "row norms" begin
            for (i, cols) in cols4row
                row_norm2 = 0.
                for j in cols
                    row_norm2 += M_[i, j]^2
                end
                row_norms[i] = row_norm2
            end
        end
        @timeit "u grad " u_grad .= row_norms .- α ^ 2 .+ γ * u
        @timeit "u proj " u = box_project(u - step_size * u_grad, lb, ub)

        # v grad estimate
        @timeit "col norms" begin
            for (j, rows) in rows4col
                col_norm2 = 0.
                for i in rows
                    col_norm2 += M_[i, j]^2
                end
                col_norms[j] = col_norm2
            end
        end
        @timeit "v grad " v_grad .= col_norms .- β ^ 2 .+ γ * v
        @timeit "v proj 1" v .= sum(v - step_size * v_grad) / aff.n
        @timeit "v proj 2" v = box_project(v, 0., ub)
        
        # Update averages.
        @timeit "u update" u_ = 2 * u / (iter + 2) + iter * u_ / (iter + 2)
        @timeit "v update" v_ = 2 * v / (iter + 2) + iter * v_ / (iter + 2)
    end

    @timeit "update diag E" E[diagind(E)] .= exp.(u_)
    @timeit "update diag D" D[diagind(D)] .= exp.(v_)

    return E, D
end

function box_project(y, lb, ub)
    return min.(ub, max.(y, lb))
end