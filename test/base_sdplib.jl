function sdplib_data(path)
    # Read data from file
    data = readdlm(path, use_mmap=true)

    println("000")

    # Parse SDPLIB data
    m = data[1, 1]
    if isa(data[3, 1], Float64) || isa(data[3, 1], Int64)
        blks = data[3, :]
        for elem = 1:length(blks)
            if blks[elem] == ""
                blks = blks[1:elem-1]
                break
            end
        end
    else
        blks = parse.(Float64, split(data[3, 1][2:end - 1], ","))
    end
    println("001")
    cum_blks = unshift!(cumsum(blks), 0)
    if isa(data[4, 1], Float64) || isa(data[4, 1], Int64) 
        c = data[4, :]
    else
        c = [parse(Float64,string) for string in split(data[4, 1][2:end - 1], ",")]
    end
    n = abs(cum_blks[end])
    n = length(c)
    println("002")
    # n = 335
    println("111")
    F = Dict(i => spzeros(n, n) for i = 0:m)
    for k=5:size(data)[1]
        idx = cum_blks[data[k, 2]]
        # if data[k, 2] == 1
        #     idx = 0
        # else
        #     idx = 161
        # end

        i, j = data[k, 3] + idx, data[k, 4] + idx
        if data[k, 1] == 0
            F[0][i, j] = - data[k, 5]
            F[0][j, i] = - data[k, 5]
        else
            F[data[k, 1]][i, j] = data[k, 5]
            F[data[k, 1]][j, i] = data[k, 5]
        end
    end
    println("222")
    return n, m, F, c
end
function sdplib_eval(F,c,n,m,XX)
    rank = length([eig for eig in eigfact(XX)[:values] if eig > 1e-10])
    @show rank
    @show trace(F[0] * XX)

    nothing
end