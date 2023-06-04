
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

function ivech!(out::AbstractMatrix{T}, v::AbstractVector{T}) where T
    n = MOI.Utilities.side_dimension_for_vectorized_dimension(length(v))
    n1, n2 = size(out)
    @assert n == n1 == n2
    c = 0
    for j in 1:n, i in 1:j
        c += 1
        out[i,j] = v[c]
    end
    return out
end

function ivech(v::AbstractVector{T}) where T
    n = MOI.Utilities.side_dimension_for_vectorized_dimension(length(v))
    out = zeros(n, n)
    ivech!(out, v)
    
    return out
end

ivec(X) = Matrix(LinearAlgebra.Symmetric(ivech(X),:U))
