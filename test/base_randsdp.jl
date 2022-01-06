import Random
import LinearAlgebra

function randsdp_data(seed, m, n)
    rng = Random.MersenneTwister(seed)
    # n = 15  # Instance size
    # m = 10  # Number of constraints
    # Objective function
    c_sqrt = Random.rand(rng, Float64, (n, n))
    C = c_sqrt * c_sqrt'
    # C[1, 2] *= 0.5
    # C[2, 1] = C[1, 2]
    # Generate m-dimensional feasible system
    A, b = Dict(), Dict()
    X_ = Random.randn(rng, (n, n))
    X_ = X_ * X_'
    for i in 1:m
        A[i] = Random.rand(rng, Float64, (n, n))
        A[i] = A[i] * A[i]'
        b[i] = tr(A[i] * X_)
    end
    return A, b, C
end
function randsdp_eval(A,b,C,n,m,XX)
    @show minus_rank = length([eig for eig in LinearAlgebra.eigen(XX).values if eig < -1e-10])

    @show rank = length([eig for eig in LinearAlgebra.eigen(XX).values if eig > 1e-10])

    @show tr(C * XX)
    for i in 1:m
        @show tr(A[i] * XX)-b[i]
    end

    nothing
end