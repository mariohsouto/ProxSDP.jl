
import LinearAlgebra: BlasInt
import Random

mutable struct ARPACKAlloc{T}

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
    TOL::Vector{T}
    v::Matrix{T}
    workd::Vector{T}
    workl::Vector{T}
    rwork::Vector{T}
    resid::Vector{T}
    info::Vector{BlasInt}
    iparam::Vector{BlasInt}
    ipntr::Vector{BlasInt}
    ido::Vector{BlasInt}
    zernm1::UnitRange{Int}
    howmny::String
    select::Vector{BlasInt}
    info_e::Vector{BlasInt}
    d::Vector{T}
    sigmar::Vector{T}
    converged::Bool
    arpackerror::Bool

    function ARPACKAlloc{T}() where T
        new{T}()
    end
end

hasconverged(arc::ARPACKAlloc) = arc.converged

function unsafe_getvalues(arc::ARPACKAlloc)
    return arc.d
end

function unsafe_getvectors(arc::ARPACKAlloc)
    return arc.v
end

function ARPACKAlloc(T::DataType, n::Integer=1, nev::Integer=1)
    arc = ARPACKAlloc{T}()
    # ARPACKAlloc_reset!(arc::ARPACKAlloc, Symmetric(Matrix{T}(I, n, n)), 1)
    return arc
end

function ARPACKAlloc_reset!(arc::ARPACKAlloc{T}, A::Symmetric{T,Matrix{T}}, nev::Integer) where T

    # Don't need to be computed everytime
    n = LinearAlgebra.checksquare(A)
    # Dimension of the eigenproblem
    arc.n = n
    # Number of eigenvalues of OP to be computed
    arc.nev = nev
    # Standard eigenvalue problem
    arc.bmat = "I"
    # Compute the NEV largest (algebraic) eigenvalues
    arc.which = "LA"
    # Flag for A*x = lambda*x
    arc.mode = 1

    # How many Lanczos vectors are generated
    arc.ncv = max(20, 2 * arc.nev + 1)
    # arc.ncv = arc.nev + 1
    # Maximum number of iterations
    arc.maxiter = Int(1e+4)
    # Stopping criterion
    arc.TOL = 1e-10 * ones(T, 1)

    # If info != 0, RESID contains the initial residual vector
    arc.info = ones(BlasInt, 1)
    arc.info_e = ones(BlasInt, 1) #Ref{BlasInt}(0)
    # Resid contains the initial residual vector
    Random.seed!(1234);
    # arc.resid = rand(arc.n)
    arc.resid = ones(BlasInt, arc.n)

    # Build linear operator (?)
    matvecA!(y, x) = mul!(y, A, x)
    arc.A = matvecA!
    arc.Amat = A
    # arc.x = Vector{T}(undef, arc.n)
    # arc.y = Vector{T}(undef, arc.n)
    arc.x = ones(arc.n)
    arc.y = ones(arc.n)

    # Lanczos basis vectors (output)
    # arc.v = Matrix{T}(undef, arc.n, arc.ncv)
    arc.v = ones(BlasInt, arc.n, arc.ncv)

    # Workspace
    # arc.workd = Vector{T}(undef, 3 * arc.n)
    arc.workd = ones(3 * arc.n)
    arc.lworkl = arc.ncv * (arc.ncv + 8)
    # arc.workl = Vector{T}(undef, arc.lworkl)
    arc.workl = ones(arc.lworkl)
    arc.rwork = Vector{T}()
    arc.iparam = ones(BlasInt, 11)
    arc.iparam[1] = BlasInt(1)       # ishifts
    arc.iparam[3] = BlasInt(arc.maxiter) # maxiter
    arc.iparam[4] = BlasInt(1)
    arc.iparam[7] = BlasInt(arc.mode)
    arc.ipntr = zeros(BlasInt, 11) 
    # IDO must be zero on the first call to dsaupd
    arc.ido = zeros(BlasInt, 1)

    # Parameters for _EUPD! routine
    arc.zernm1 = 0:(arc.n-1)
    arc.howmny = "A"
    # arc.select = Vector{BlasInt}(undef, arc.ncv)
    # arc.d = Vector{T}(undef, arc.nev)
    arc.select = ones(arc.ncv)
    arc.d = ones(arc.nev)
    arc.sigmar = zeros(T,1)#Ref{T}(zero(T))

    # Flags created for ProxSDP use
    arc.converged = false
    arc.arpackerror = false

    return nothing
end

function _INIT!(arc::ARPACKAlloc, A::Symmetric{T1,Matrix{T1}}, nev::Integer, n::Int64) where T1

    # T = eltype(A)

    # if eltype(arc.v) != T || n != arc.n || nev != arc.nev
        return ARPACKAlloc_reset!(arc, A, nev)
    # end

    # matvecA!(y, x) = mul!(y, A, x)
    # arc.A = matvecA!
    # arc.Amat = A

    # arc.info[1] = BlasInt(1) # hotstart
    # arc.info_e[1] = BlasInt(1)
    # arc.sigmar[1] = 0.0

    # # IDO must be zero on the first call to dsaupd
    # arc.ido[1] = BlasInt(0)
    # arc.iparam[1] = BlasInt(1)       # ishifts
    # arc.iparam[3] = BlasInt(arc.maxiter) # maxiter
    # arc.iparam[7] = BlasInt(1)    # mode

    # # Flags created for PrxSDP use
    # arc.converged = false
    # arc.arpackerror = false

    # return nothing
end

function _AUPD!(arc::ARPACKAlloc{T}) where T
    
    while true
        Arpack.saupd(arc.ido, arc.bmat, arc.n, arc.which, arc.nev, Ref(arc.TOL[1]), arc.resid, arc.ncv, arc.v, arc.n,
        arc.iparam, arc.ipntr, arc.workd, arc.workl, arc.lworkl, arc.info)

        # ????
        # x = view(arc.workd, arc.ipntr[1] + arc.zernm1)
        # y = view(arc.workd, arc.ipntr[2] + arc.zernm1)
        # arc.A(y, x)

        if arc.ido[] == 1

            # ????
            @inbounds @simd for i in 1:arc.n
                arc.x[i] = arc.workd[i-1+arc.ipntr[1]]
            end
            # arc.A(arc.y, arc.x)
            mul!(arc.y, arc.Amat, arc.x)
            @inbounds @simd for i in 1:arc.n
                 arc.workd[i-1+arc.ipntr[2]] = arc.y[i]
            end

        elseif arc.ido[] == 99
            break
        else
            # I'm still not sure about this last statement
            arc.converged = false
            arc.arpackerror = true
            return nothing
        end
    end

    # Check if _AUPD has converged properly
    if !(arc.info[] in [0, 1])
        arc.converged = false
        arc.arpackerror = true
    else
        arc.converged = true
    end

    return nothing
end

function _EUPD!(arc)

    # Check if _AUPD has converged properly
    if !arc.converged
        arc.converged = false        
        return nothing
    end

    Arpack.seupd(true, arc.howmny, arc.select, arc.d, arc.v, arc.n, arc.sigmar,
    arc.bmat, arc.n, arc.which, arc.nev, Ref(arc.TOL[1]), arc.resid, arc.ncv, arc.v, arc.n,
    arc.iparam, arc.ipntr, arc.workd, arc.workl, arc.lworkl, arc.info_e)

    # Check if seupd has converged properly
    if arc.info_e[] != 0
        arc.converged = false
        arc.arpackerror = true
        return nothing
    end

    # Check number of converged eigenvalues (maybe some can still be used?)
    nconv = arc.iparam[5]
    if nconv < arc.nev 
        arc.converged = false
        arc.arpackerror = false
        return nothing        
    end

    arc.converged = true
    arc.arpackerror = false

    return nothing
end

function eig!(arc::ARPACKAlloc, A::Symmetric{T1,Matrix{T1}}, nev::Integer, n::Int64)::Nothing where T1

    # Initialize parameters and do memory allocation
    @timeit "_INIT!" _INIT!(arc, A, nev, n)::Nothing

    # Top level reverse communication interface to solve real double precision symmetric problems.
    @timeit "_AUPD!" _AUPD!(arc)::Nothing

    # Post processing routine (eigenvector purification)
    @timeit "_EUPD!" _EUPD!(arc)::Nothing

    return nothing
end