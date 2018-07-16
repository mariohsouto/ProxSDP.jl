
import Base.LinAlg: BlasInt, ARPACKException

using TimerOutputs

Base.LinAlg.ARPACK

mutable struct ARPACKAlloc{T}

    # eigs
    n::Int
    nev::Int
    ncv::Int
    maxiter::Int

    bmat::String
    which::String
    mode::Int

    A::Function
    Amat::Symmetric{Float64,Matrix{Float64}}
    x::Vector{T}
    y::Vector{T}

    lworkl::Int

    TOL::Vector{T}#Ref{T}

    v::Matrix{T}
    workd::Vector{T}
    workl::Vector{T}
    rwork::Vector{T}

    resid::Vector{T}
    info::Vector{BlasInt}#Ref{BlasInt}

    iparam::Vector{BlasInt}
    ipntr::Vector{BlasInt}
    ido::Vector{BlasInt}#Ref{BlasInt}

    zernm1::UnitRange{Int}

    # eupd

    howmny::String
    select::Vector{BlasInt}
    info_e::Vector{BlasInt}#Ref{BlasInt}

    d::Vector{T}
    sigmar::Vector{T}#Ref{T}

    converged::Bool
    arpackerror::Bool

    function ARPACKAlloc{T}() where T
        new{T}()
    end
end

hasconverged(arc::ARPACKAlloc) = arc.converged

function Base.getindex(arc::ARPACKAlloc, s::Symbol)
    if s == :Values
        return arc.d
    elseif s == :Vector
        return arc.v
    else
        error("Not a field")
    end
end

function Base.getindex(arc::ARPACKAlloc{T}, s::Symbol, i::Integer)::T where T
    if s == :Values && i <= arc.nev
        return arc.d[i]
    else
        error("Not allowed")
    end
end

function getvector(arc::ARPACKAlloc{T}, s::Symbol, i::Integer)::Vector{T} where T
    if i <= arc.nev
        return arc.v[:,i]
    else
        error("Bounds out of range")
    end
end

function unsafe_getvalues(arc::ARPACKAlloc)
    return arc.d
end

function unsafe_getvectors(arc::ARPACKAlloc)
    return arc.v
end

function ARPACKAlloc(T::DataType, n::Integer=1, nev::Integer=1)
    arc = ARPACKAlloc{T}()
    ARPACKAlloc_reset!(arc::ARPACKAlloc, Symmetric(eye(T,n,n)), 1)
    return arc
end

function ARPACKAlloc(A, nev::Integer)
    arc = ARPACKAlloc{T}()
    ARPACKAlloc_reset!(arc::ARPACKAlloc, A, nev)
    return arc
end

function ARPACKAlloc_reset!(arc::ARPACKAlloc{T}, A::Symmetric{T,Matrix{T}}, nev::Integer) where T

    tol = 0.0
    v0=zeros(eltype(A),(0,))
    #, v0::Vector{T} = zeros(T,(0,))

    newT = eltype(A)
    n = Base.LinAlg.checksquare(A)

    if newT != T
        error("Element type change is not allowed")
    end

    # eigs
    arc.n = n
    arc.nev = nev
    arc.ncv = max(20, 2*arc.nev+1)
    arc.maxiter = Int(1e+3)

    arc.bmat = "I"
    arc.which = "LA"

    arc.mode = 1
    # arc.solveSI = x->x
    # arc.B = x->x
    matvecA!(y, x) = A_mul_B!(y, A, x)
    arc.A = matvecA!
    arc.Amat = A
    arc.x = Vector{T}(arc.n)
    arc.y = Vector{T}(arc.n)

    arc.lworkl = arc.ncv * (arc.ncv + 8)

    arc.TOL = 1e-8 * ones(T, 1)

    arc.v = Matrix{T}(arc.n, arc.ncv)
    arc.workd = Vector{T}(3*arc.n)
    arc.workl = Vector{T}(arc.lworkl)
    arc.rwork = Vector{T}() # cmplx ? Vector{TR}(ncv) : Vector{TR}()

    if isempty(v0)
        arc.resid = Vector{T}(arc.n)
        arc.info  = zeros(BlasInt, 1)#Ref{BlasInt}(0)
    else
        # arc.resid = v0#deepcopy(v0)
        arc.info  = ones(BlasInt, 1)#Ref{BlasInt}(1)
    end

    arc.iparam = zeros(BlasInt, 11)
    arc.ipntr = zeros(BlasInt, 11) # = zeros(BlasInt, (sym && !cmplx) ? 11 : 14)
    arc.ido = zeros(BlasInt, 1) #Ref{BlasInt}(0)

    arc.iparam[1] = BlasInt(1)       # ishifts
    arc.iparam[3] = BlasInt(arc.maxiter) # maxiter
    arc.iparam[7] = BlasInt(1)    # mode

    arc.zernm1 = 0:(arc.n-1)

    arc.howmny = "A"
    arc.select = Vector{BlasInt}(arc.ncv)
    arc.info_e = zeros(BlasInt, 1)#Ref{BlasInt}(0)

    arc.d = Vector{T}(arc.nev)
    arc.sigmar = zeros(T,1)#Ref{T}(zero(T))

    arc.converged = false
    arc.arpackerror = false

    return nothing
end

function _AUPD!(arc::ARPACKAlloc)
    while true
        Base.LinAlg.ARPACK.saupd(arc.ido, arc.bmat, arc.n, arc.which, arc.nev, arc.TOL, arc.resid, arc.ncv, arc.v, arc.n,
        arc.iparam, arc.ipntr, arc.workd, arc.workl, arc.lworkl, arc.info)

        # x = view(arc.workd, arc.ipntr[1] + arc.zernm1)
        # y = view(arc.workd, arc.ipntr[2] + arc.zernm1)
        # arc.A(y, x)

        if arc.ido[] == 1
            @inbounds @simd for i in 1:arc.n
                arc.x[i] = arc.workd[i-1+arc.ipntr[1]]
            end
            # arc.A(arc.y, arc.x)
            A_mul_B!(arc.y, arc.Amat, arc.x)
            @inbounds @simd for i in 1:arc.n
                 arc.workd[i-1+arc.ipntr[2]] = arc.y[i]
            end
        elseif arc.ido[] == 99
            break
        else
            arc.converged = false
            arc.arpackerror = true
            return nothing
            # throw(ARPACKException("unexpected behavior"))
        end
    end
    arc.converged = true
    return nothing
end

function _INIT!(arc::ARPACKAlloc, A::Symmetric{T1,Matrix{T1}}, nev::Integer) where T1

    n = Base.LinAlg.checksquare(A)
    T = eltype(A)

    if eltype(arc.v) != T || n != arc.n || nev != arc.nev
        return ARPACKAlloc_reset!(arc, A, nev)
    end

    matvecA!(y, x) = A_mul_B!(y, A, x)
    arc.A = matvecA!
    arc.Amat = A

    # arc.info[1] = BlasInt(0)# = zeros(BlasInt, 1)#Ref{BlasInt}(0)
    arc.info[1] = BlasInt(1)# hotstart

    # arc.iparam = zeros(BlasInt, 11)
    # arc.ipntr = zeros(BlasInt, 11) # = zeros(BlasInt, (sym && !cmplx) ? 11 : 14)
    arc.ido[1] = BlasInt(0)# zeros(BlasInt, 1)#Ref{BlasInt}(0)

    arc.iparam[1] = BlasInt(1)       # ishifts
    arc.iparam[3] = BlasInt(arc.maxiter) # maxiter
    arc.iparam[7] = BlasInt(1)    # mode

    arc.info_e[1]   = BlasInt(0)# zeros(BlasInt, 1)#Ref{BlasInt}(0)

    arc.sigmar[1] = 0.0#BlasInt(0)# zeros(T,1)#Ref{T}(zero(T))

    arc.converged = false
    arc.arpackerror = false

    return nothing
end

function _EUPD!(arc)

    if !arc.converged
        arc.converged = false        
        return nothing
    end

    # d = Vector{T}(nev)
    # sigmar = Ref{T}(sigma)
    Base.LinAlg.ARPACK.seupd(true, arc.howmny, arc.select, arc.d, arc.v, arc.n, arc.sigmar,
    arc.bmat, arc.n, arc.which, arc.nev, arc.TOL, arc.resid, arc.ncv, arc.v, arc.n,
    arc.iparam, arc.ipntr, arc.workd, arc.workl, arc.lworkl, arc.info_e)
    if arc.info_e[] != 0
        arc.converged = false
        arc.arpackerror = true
        return nothing
        # throw(ARPACKException(arc.info_e[]))
    end

    nconv = arc.iparam[5]
    if nconv < arc.nev 
        arc.converged = false
        arc.arpackerror = false
        return nothing        
    end

    arc.converged = true
    arc.arpackerror = false
    # p = sortperm(dmap(d), rev=true)
    # return d[p], v[1:n, p]#,iparam[5],iparam[3],iparam[9],resid
    return nothing
end


function eig!(arc, A, nev)

    @timeit "_INIT!" _INIT!(arc, A, nev)
    @timeit "_AUPD!" _AUPD!(arc)
    @timeit "_EUPD!" _EUPD!(arc)

    return nothing
end
