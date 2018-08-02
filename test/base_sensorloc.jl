function sensorloc_data(seed, n)

    # Instance size
    m = 10 * n
    # Sensor true position
    x_true = rand((n, 1))
    # Anchor positions
    a = Dict(i => rand((n, 1)) for i in 1:m)
    d = Dict(i => norm(x_true - a[i]) for i in 1:m)
    A = Dict()
    for i in 1:m
        A[i] = [hcat(eye(n), -a[i]); hcat(-a[i]', 0.0)]
    end

    return m, x_true, a, d, A
end
function sensorloc_eval(n, m, x_true, XX)
    @show norm(x_true - XX[1:n, end])
    @show rank = length([eig for eig in eigfact(XX)[:values] if eig > 1e-7])
    nothing
end