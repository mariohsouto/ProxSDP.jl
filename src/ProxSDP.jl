module ProxSDP

using PrecompileTools: @setup_workload, @compile_workload

import Arpack
import KrylovKit
import MathOptInterface
import TimerOutputs
import TimerOutputs: @timeit

import Printf
import SparseArrays
import LinearAlgebra
import LinearAlgebra: BlasInt
import Random
import Logging


include("structs.jl")
include("options.jl")
include("util.jl")
include("printing.jl")
include("scaling.jl")
include("equilibration.jl")
include("pdhg.jl")
include("residuals.jl")
include("eigsolver.jl")
include("prox_operators.jl")
include("MOI_wrapper.jl")

# PrecompileTools
@setup_workload begin

    @compile_workload begin
        include("../test/run_mini_benchmark.jl")
    end

end

end