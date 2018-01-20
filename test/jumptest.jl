using JuMP
using Mosek
using Base.Test

using MathOptInterface
const MOI = MathOptInterface

# @testset "LP0" begin
#     m = Model()
#     @variable(m,x[1:4,1:4],PSD)
#     @constraint(m, 1x[1,1]+2x[2,2]+x[3,3]==4)
#     @constraint(m, 2x[1,1]+1x[2,2]+x[4,4]==4)
#     @objective(m, Min, 1x[1,1]+2x[2,2])
#     JuMP.attach(m, ProxSDPSolverInstance())
#     JuMP.solve(m)
# end


@testset "Linear Programming" begin
    @testset "LP1" begin

        m = Model(solver=MOSEKSolver())

        srand(0)
        n = 30
        # Channel
        H = randn((n, n))
        # Gaussian noise
        v = randn((n, 1))
        # True signal
        s = rand([-1, 1], n)
        # Received signal
        sigma = 0.01
        y = H * s + sigma * v
        L = [hcat(H' * H, -H' * y); hcat(-y' * H, y' * y)]

        # n = 100
        # L = readcsv("/Users/mariosouto/Dropbox/proxsdp/L.csv")

        @variable(m, X[1:n+1, 1:n+1], PSD)
        @objective(m, Min, sum(L[i, j] * X[i, j] for i in 1:n+1, j in 1:n+1))
        @constraint(m, ctr[i in 1:n+1], X[i, i] == 1.0)

        # JuMP.attach(m, ProxSDPSolverInstance())
        teste = JuMP.solve(m)
        println("Objective value: ", getobjectivevalue(m))

        println(teste)

        # @test JuMP.isattached(m)
        # @test JuMP.hasvariableresult(m)

        # @test JuMP.terminationstatus(m) == MOI.Success
        # @test JuMP.primalstatus(m) == MOI.FeasiblePoint
        # @test JuMP.dualstatus(m) == MOI.FeasiblePoint

        # @test JuMP.resultvalue(x) ≈ 1.0 atol=1e-6
        # @test JuMP.resultvalue(y) ≈ 0.0 atol=1e-6
        # @test JuMP.resultvalue(x + y) ≈ 1.0 atol=1e-6
        # @test JuMP.objectivevalue(m) ≈ -1.0 atol=1e-6

        # @test JuMP.resultdual(c) ≈ -1 atol=1e-6
    end
end