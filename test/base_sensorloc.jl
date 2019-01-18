
function sensorloc_data(seed, n)

    # n = number of sensors points
    # m = number of anchor points
    m = floor(Int, 0.1 * n)

    # Sensor true position (2 dimensional)
    x_true = rand(Float64, (2, n))
    # Distances from sensors to sensors
    d = Dict((i, j) => norm(x_true[:, i] - x_true[:, j]) for i in 1:n for j in 1:i)

    # Anchor positions
    a = Dict(i => rand(Float64, (2, 1)) for i in 1:m)
    # Distances from anchor to sensors
    d_bar = Dict((k, j) => norm(x_true[:, j] - a[k]) for k in 1:m for j in 1:n)

    return m, x_true, a, d, d_bar
end

function sensorloc_eval(n, m, x_true, XX)
    @show norm(x_true - XX[1:2, 3:n + 2])
    @show rank = length([eig for eig in eigen(XX).values if eig > 1e-7])
    return nothing
end