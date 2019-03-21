
function get_solution(opt::MOIU.CachingOptimizer{Optimizer,T}) where T
    return opt.optimizer.sol
end

function map_socs!(v::Vector{Float64}, conic_sets::ConicSets, a::AuxiliaryData)
    cont = 0
    for (idx, sdp) in enumerate(conic_sets.sdpcone)
        cont += div(sdp.sq_side*(sdp.sq_side+1), 2)
    end
    sizehint!(a.soc_v, length(conic_sets.socone))
    sizehint!(a.soc_s, length(conic_sets.socone))
    for (idx, soc) in enumerate(conic_sets.socone)
        len = soc.len
        # push!(a.soc_v, view(v, cont+1:cont+len-1))
        # push!(a.soc_s, view(v, cont+len))
        push!(a.soc_s, view(v, cont+1))
        push!(a.soc_v, view(v, cont+2:cont+len))
        cont += len
    end
    return nothing
end