
using LinearAlgebra

function equilibrate!(M, Mt, aff, α=1.)

    all_cols = 1:aff.n
    all_rows = 1:aff.m + aff.p

    Σ    = Matrix{Float64}(I, aff.m + aff.p, aff.m + aff.p)
    Σinv = Matrix{Float64}(I, aff.m + aff.p, aff.m + aff.p)
    T    = Matrix{Float64}(I, aff.n, aff.n)
    Tinv = Matrix{Float64}(I, aff.n, aff.n)

    # Columns of M that are entirely filled with zeros
    cols = [j for j in 1:aff.n if !iszero(M[:, j])]
    rows = [i for i in 1:aff.m + aff.p if !iszero(M[i, :])]

    for j in cols
        T[j, j] = 1. / sum(abs(M[i, j])^(2 - α) for i in all_rows)
        if isnan(T[j, j]) || isinf(T[j, j])
            T[j, j] = 1.
        end
        Tinv[j, j] = 1. / T[j, j]
        if isnan(Tinv[j, j]) || isinf(Tinv[j, j])
            Tinv[j, j] = 1.
        end
    end

    for i in rows
        Σ[i, i] = 1. / sum(abs(M[i, j])^α for j in all_cols)
        if isnan(Σ[i, i]) || isinf(Σ[i, i])
            Σ[i, i] = 1.
        end
        Σinv[i, i] = 1. / Σinv[i, i]
        if isnan(Σinv[i, i]) || isinf(Σinv[i, i])
            Σinv[i, i] = 1.
        end
    end

    sum_sigma = sum(diag(Σ)) / (aff.m + aff.p)
    sum_inv_sigma = sum(diag(Σinv)) / (aff.m + aff.p)
    for i in rows
        Σ[i, i] = sum_sigma
        Σinv[i, i] = sum_sigma
    end

    return Σ, Σinv, T, Tinv
end

function _equilibrate!(M, Mt, aff, max_iters=100000, l=100., ϵ1=1e-4, ϵ2=1e-4)

    D    = Matrix{Float64}(I, aff.m + aff.p, aff.m + aff.p)
    Dinv = Matrix{Float64}(I, aff.m + aff.p, aff.m + aff.p)
    E    = Matrix{Float64}(I, aff.n, aff.n)
    Einv = Matrix{Float64}(I, aff.n, aff.n)

    D_    = Matrix{Float64}(I, aff.m + aff.p, aff.m + aff.p)
    E_    = Matrix{Float64}(I, aff.n, aff.n)

    iter, r1, r2 = 0, 1., 1.

    # Columns of M that are entirely filled with zeros
    cols = [j for j in 1:aff.n if !iszero(M[:, j])]
    rows = [i for i in 1:aff.m + aff.p if !iszero(M[i, :])]

    while r1 > ϵ1 && r2 > ϵ2 && iter < max_iters
        iter += 1

        @show size(D)
        @show size(M)
        @show size(E)
        D = M * E
        for i in rows
            D[i, i] = 1. / D[i, i]
            if isnan(D[i, i]) || isinf(D[i, i])
                D[i, i] = 1.
            end
        end

        E = Mt * D
        for j in cols
            E[j, j] = 1. / E[j, j]
            if isnan(E[j, j]) || isinf(E[j, j])
                E[j, j] = 1.
            end
        end
        @show size(D)
        @show size(M)
        @show size(E)
        M_ = D * M * E

        # Compute residuals
        r1 = maximum([norm(M_[i, :]) for i in rows]) / minimum([norm(M_[i, :]) for i in rows])
        r2 = maximum([norm(M_[:, j]) for j in cols]) / minimum([norm(M_[:, j]) for j in cols])
        @show (r1, r2)
    end

    for i in 1:aff.m + aff.p
        Dinv[i, i] = 1. / D[i, i]
    end
    for j in 1:aff.n
        Einv[j, j] = 1. / Einv[j, j]
    end

    return E, D, Einv, Dinv
end

function __equilibrate!(M, Mt, aff, max_iters=100, l=100., ϵ1=1e-4, ϵ2=1e-4)

    D    = Matrix{Float64}(I, aff.m + aff.p, aff.m + aff.p)
    Dinv = Matrix{Float64}(I, aff.m + aff.p, aff.m + aff.p)
    E    = Matrix{Float64}(I, aff.n, aff.n)
    Einv = Matrix{Float64}(I, aff.n, aff.n)

    iter, r1, r2 = 0, 1., 1.

    # Columns of M that are entirely filled with zeros
    cols = [j for j in 1:aff.n if !iszero(M[:, j])]
    rows = [i for i in 1:aff.m + aff.p if !iszero(M[i, :])]

    while r1 > ϵ1 && r2 > ϵ2 && iter < max_iters
        iter += 1
        for i in rows
            D[i, i] = D[i, i] * norm(M[i, :], 2)^(-.5)
            if isnan(D[i, i]) || isinf(D[i, i])
                D[i, i] = 1.
            end
        end
        for j in cols
            E[j, j] = E[j, j] * ((aff.m + aff.p) / aff.n)^.5 * norm(M[:, j], 2)^(-.5)
            if isnan(E[j, j]) || isinf(E[j, j])
                E[j, j] = 1.
            end
        end
        M = D * M * E

        # Compute residuals
        r1 = maximum([norm(M[i, :]) for i in rows]) / minimum([norm(M[i, :]) for i in rows])
        r2 = maximum([norm(M[:, j]) for j in cols]) / minimum([norm(M[:, j]) for j in cols])
        @show (r1, r2)
    end

    for i in 1:aff.m + aff.p
        Dinv[i, i] = 1. / D[i, i]
    end
    for j in 1:aff.n
        Einv[j, j] = 1. / Einv[j, j]
    end

    return E, D, Einv, Dinv
end

function _equilibrate!_(M, Mt, aff, max_iters=10000, lb=-100., ub=100.)

    α = (aff.n / (aff.m + aff.p)) ^ .25
    β = ((aff.m + aff.p) / aff.n ) ^ .25
    γ = 1e-1

    u, v   = zeros(aff.m + aff.p), zeros(aff.n)
    u_, v_ = zeros(aff.m + aff.p), zeros(aff.n)

    in_buf  = zeros(aff.n)
    out_buf = zeros(aff.m + aff.p)

    # Columns of M that are entirely filled with zeros
    cols = [j for j in 1:aff.n if !iszero(M[:, j])]
    rows = [i for i in 1:aff.m + aff.p if !iszero(M[i, :])]

    # r1 = maximum([norm(M[i, :]) for i in rows]) / minimum([norm(M[i, :]) for i in rows])
    # r2 = maximum([norm(M[:, j]) for j in cols]) / minimum([norm(M[:, j]) for j in cols])
    # @show (r1, r2)

    for iter in 1:max_iters
        step_size = 2. / (γ * (iter + 1.))

        # u grad estimate
        s = rand([-1., +1.], aff.n)
        out_buf .= M * (exp.(v) .* s)
        u_grad = exp.(2 * u) .* out_buf .^ 2 .- α ^ 2 .+ γ * u

        # v grad estimate
        w = rand([-1., +1.], aff.m + aff.p)
        in_buf .= Mt * (exp.(u) .* w)
        v_grad = exp.(2 * v) .* in_buf .^ 2 .- β ^ 2 .+ γ * v

        # Project onto box 
        u = box_project(u - step_size * u_grad, 0., ub)
        v = box_project(v - step_size * v_grad, lb, ub)

        u .= sum(u) / aff.n
        
        # Update averages.
        u_ = 2 * u / (iter + 2) + iter * u_ / (iter + 2)
        v_ = 2 * v / (iter + 2) + iter * v_ / (iter + 2)
    end

    u_ .= exp.(u)
    v_ .= exp.(v)

    D    = Diagonal(u_)
    Dinv = Diagonal(1. ./ u_)
    E    = Diagonal(v_)
    Einv = Diagonal(1. ./ v_)

    # M_ = D * M * E

    # r1 = maximum([norm(M_[i, :]) for i in rows]) / minimum([norm(M_[i, :]) for i in rows])
    # r2 = maximum([norm(M_[:, j]) for j in cols]) / minimum([norm(M_[:, j]) for j in cols])
    # @show (r1, r2)

    return D, Dinv, E, Einv
end

function box_project(y, lb, ub)
    return min.(ub, max.(y, lb))
end