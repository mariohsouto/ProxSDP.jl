import Random

function mimo_data(seed, m, n)
    rng = Random.MersenneTwister(seed)

    # Channel
    H = Random.randn(rng, (m, n))
    # Gaussian noise
    v = Random.randn(rng, (m, 1))
    # True signal
    s = Random.rand(rng, [-1, 1], n)
    # Received signal
    sigma = .0001
    y = H * s + sigma * v
    L = [hcat(H' * H, -H' * y); hcat(-y' * H, y' * y)]
    return s, H, y, L
end

function mimo_eval(s, H, y, L, XX)
    x_hat = sign.(XX[1:end-1, end])
    rank = length([eig for eig in LinearAlgebra.eigen(XX).values if eig > 1e-7])
    @show decode_error = sum(abs.(x_hat - s))
    @show rank
    @show LinearAlgebra.norm(y - H * x_hat)
    @show LinearAlgebra.norm(y - H * s)
    @show tr(L * XX)
    return nothing
end