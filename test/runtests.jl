path = joinpath(dirname(@__FILE__),"..","..")
push!(Base.LOAD_PATH, path)
datapath = joinpath(dirname(@__FILE__),"data")

using ProxSDP

# ProxSDP.runpsdp(datapath)

# include("moitest.jl")

include("max_cut.jl")