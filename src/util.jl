
function map_socs!(v::Vector{Float64}, cones::ConicSets, a::AuxiliaryData)
    cont = 0
    for (idx, sdp) in enumerate(cones.sdpcone)
        cont += div(sdp.sq_side * (sdp.sq_side + 1), 2)
    end
    sizehint!(a.soc_v, length(cones.socone))
    sizehint!(a.soc_s, length(cones.socone))
    for (idx, soc) in enumerate(cones.socone)
        len = soc.len
        push!(a.soc_s, view(v, cont + 1))
        push!(a.soc_v, view(v, cont + 2:cont + len))
        cont += len
    end
    return nothing
end