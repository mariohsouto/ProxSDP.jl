# Load packages
path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)
using ProxSDP, JuMP, LinearAlgebra

function sdp_relax(W::Array{Float64,2})::Tuple{Array{Float64,2}, Float64}

    # Build Max-Cut SDP relaxation via JuMP
    model = Model(with_optimizer(ProxSDP.Optimizer, log_verbose=true))
    @variable(model, X[1:n, 1:n], PSD)
    @objective(model, Max, .25 * dot(W, X))
    @constraint(model, diag(X) .== 1)

    # Solve optimization problem with ProxSDP
    JuMP.optimize!(model)

    # Retrieve solution
    Xsdp = JuMP.value.(X)

    # Upper bound on the max-cut
    @show ub = objective_value(model)

    return Xsdp, ub
end

function cholesky_factorization(Xsdp::Array{Float64,2})::Transpose{Float64,Array{Float64,2}}

    # Cholesky factorization of SDP solution
    chol = cholesky(Hermitian(Xsdp, :U), Val(true); check=false)
    V = transpose(chol.P * chol.L)

    # Normalize Cholesky factorization
    for j in 1:n
        V[:, j] /= norm(V[:, j])
    end

    return V
end

function randomized_round(W::Array{Float64,2}, V::Transpose{Float64,Array{Float64,2}}, ub::Float64, max_iter_round::Int64)::Tuple{Array{Float64,1}, Float64}

    # Initialize
    best_cut_value = - Inf
    max_iter_round = 100
    n = size(W)[1]
    x = ones(n)

    # Perform consecutive randomized rounding  and save best solution
    for _ in 1:max_iter_round
        # Generate random hyperplane onto the unit sphere
        r = rand(n)
        r ./= norm(r)

        # Iterate over vertices, and assign each vertex to a side of cut.
        x = ones(n)
        for i in 1:n
            if dot(r, V[:, i]) <= 0
                x[i] = -1
            end
        end

        cut_value = .25 * sum(W .* (x * x'))
        @show cut_value

        # Save best incumbent
        if cut_value > best_cut_value
            best_cut_value = cut_value
        end

        # Optimality gap
        if cut_value >= ub
            println(" Optimal cut found!")
            break
        end
    end

    return x, best_cut_value
end

# Number of vertices
n = 100

# Graph weights
W = rand(n, n)
W = (W + W') / 2.

# Solve SDP relaxation
Xsdp, ub = sdp_relax(W)

# Cholesky factorization of SDP relaxation
V = cholesky_factorization(Xsdp)

# Perform randomized rouding and return best incumbent cut
x, best_cut_value = randomized_round(W, V, ub, 100)