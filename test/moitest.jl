using MathOptInterfaceTests
const MOIT = MathOptInterfaceTests

config = MOIT.TestConfig(atol = 1e-3, rtol = 1e-3, solve = true, duals = false, infeas_certificates = false)
vecofvars = true
const solver = () -> ProxSDP.ProxSDPSolverInstance()

MOIT._lin1test(solver, config, vecofvars)
