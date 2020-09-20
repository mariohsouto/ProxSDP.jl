
function moi_sensorloc(optimizer, seed, n; verbose = false, test = false, scalar = false)

    rng = Random.MersenneTwister(seed)
    MOI.empty!(optimizer)
    if test
        @test MOI.is_empty(optimizer)
    end
    # Generate randomized problem data
    m, x_true, a, d, d_bar = sensorloc_data(seed, n)

    # Decision variable
    nvars = ProxSDP.sympackedlen(n + 2) 
    X = MOI.add_variables(optimizer, nvars)
    Xsq = Matrix{MOI.VariableIndex}(undef, n + 2, n + 2)
    ivech!(Xsq, X)
    Xsq = Matrix(Symmetric(Xsq, :U))
    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(n + 2))

    # Constraint with distances from anchors to sensors
    for j in 1:n
        for k in 1:m
            if scalar
                MOI.add_constraint(optimizer,
                    MOI.ScalarAffineFunction([
                            MOI.ScalarAffineTerm(a[k][1]*a[k][1], Xsq[1,1]),
                            MOI.ScalarAffineTerm(a[k][2]*a[k][2], Xsq[2,2]),
                            MOI.ScalarAffineTerm(-2 * a[k][1], Xsq[1, j+2]),
                            MOI.ScalarAffineTerm(-2 * a[k][2], Xsq[2, j+2]),
                            MOI.ScalarAffineTerm(1.0, Xsq[j+2, j+2]),
                        ], 0.0), MOI.EqualTo(d_bar[k, j]^2))
            else
                MOI.add_constraint(optimizer,
                MOI.VectorAffineFunction(
                    MOI.VectorAffineTerm.([1], [
                        MOI.ScalarAffineTerm(a[k][1]*a[k][1], Xsq[1,1]),
                        MOI.ScalarAffineTerm(a[k][2]*a[k][2], Xsq[2,2]),
                        MOI.ScalarAffineTerm(-2 * a[k][1], Xsq[1, j+2]),
                        MOI.ScalarAffineTerm(-2 * a[k][2], Xsq[2, j+2]),
                        MOI.ScalarAffineTerm(1.0, Xsq[j+2, j+2]),
                    ]), -[d_bar[k, j]^2]), MOI.Zeros(1))
            end
        end
    end

    # Constraint with distances from sensors to sensors
    count, count_all = 0, 0
    for i in 1:n
        for j in 1:i - 1
            count_all += 1
            if Random.rand(rng) > 0.9
                count += 1
                if scalar
                    MOI.add_constraint(optimizer, 
                        MOI.ScalarAffineFunction([
                            MOI.ScalarAffineTerm(1.0, Xsq[i+2,i+2] ),
                            MOI.ScalarAffineTerm(1.0, Xsq[j+2,j+2] ),
                            MOI.ScalarAffineTerm(-2.0, Xsq[i+2,j+2]),
                        ], 0.0), MOI.EqualTo(d[i, j]^2))
                else
                    MOI.add_constraint(optimizer, 
                        MOI.VectorAffineFunction(
                            MOI.VectorAffineTerm.([1],[
                                MOI.ScalarAffineTerm(1.0, Xsq[i+2,i+2] ),
                                MOI.ScalarAffineTerm(1.0, Xsq[j+2,j+2] ),
                                MOI.ScalarAffineTerm(-2.0, Xsq[i+2,j+2]),
                            ]), -[d[i, j]^2]), MOI.Zeros(1))
                end
            end
        end
    end
    if verbose
        @show count_all, count
    end
    if scalar
        MOI.add_constraint(optimizer, MOI.SingleVariable(Xsq[1, 1]), MOI.EqualTo(1.0))
        MOI.add_constraint(optimizer, MOI.SingleVariable(Xsq[1, 2]), MOI.EqualTo(0.0))
        MOI.add_constraint(optimizer, MOI.SingleVariable(Xsq[2, 1]), MOI.EqualTo(0.0))
        MOI.add_constraint(optimizer, MOI.SingleVariable(Xsq[2, 2]), MOI.EqualTo(1.0))
    else
        MOI.add_constraint(optimizer, MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.([1],[MOI.ScalarAffineTerm(1.0, Xsq[1, 1])]), [-1.0]), MOI.Zeros(1))
        MOI.add_constraint(optimizer, MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.([1],[MOI.ScalarAffineTerm(1.0, Xsq[1, 2])]), [ 0.0]), MOI.Zeros(1))
        MOI.add_constraint(optimizer, MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.([1],[MOI.ScalarAffineTerm(1.0, Xsq[2, 1])]), [ 0.0]), MOI.Zeros(1))
        MOI.add_constraint(optimizer, MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.([1],[MOI.ScalarAffineTerm(1.0, Xsq[2, 2])]), [-1.0]), MOI.Zeros(1))
    end

    objf_t = [MOI.ScalarAffineTerm(0.0, Xsq[1, 1])]
    if false
        objf_t = [MOI.ScalarAffineTerm(1.0, Xsq[i,i]) for i in 1:n + 2]
    end
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(objf_t, 0.0))

    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(optimizer)

    objval = MOI.get(optimizer, MOI.ObjectiveValue())

    stime = -1.0
    try
        stime = MOI.get(optimizer, MOI.SolveTime())
    catch
        println("could not query time")
    end

    Xsq_s = MOI.get.(optimizer, MOI.VariablePrimal(), Xsq)

    verbose && sensorloc_eval(n, m, x_true, Xsq_s)

    rank = -1
    status = 0
    if MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        status = 1
    end
    return (objval, stime, rank, status)
end