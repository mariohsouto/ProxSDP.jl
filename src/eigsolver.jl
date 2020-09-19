
mutable struct ARPACKAlloc{T}

    n::Int
    nev::Int
    ncv::Int
    maxiter::Int
    bmat::String
    which::String
    mode::Int
    A::Function
    Amat::Symmetric{T,Matrix{T}}
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
    x::Vector{T}
    y::Vector{T}
    tol::T

    rng::Random.MersenneTwister

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

function ARPACKAlloc(T::DataType, n::Int64)::ARPACKAlloc
    arc = ARPACKAlloc{T}()
    @timeit "init_arc" _init_arc!(arc::ARPACKAlloc, Symmetric(Matrix{T}(I, n, n), :U), 1, n)
    return arc
end

function _init_arc!(arc::ARPACKAlloc{T}, A::Symmetric{T,Matrix{T}}, nev::Int64, n::Int64) where T

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
    arc.ncv = max(2 * arc.nev, 10)
    
    # Stopping criterion
    arc.tol = 1e-10
    arc.TOL = arc.tol * ones(T, 1)
    # Maximum number of iterations
    arc.maxiter = Int(1e+4)

    # Build linear operator
    arc.Amat = A
    arc.x = Vector{T}(undef, arc.n)
    arc.y = Vector{T}(undef, arc.n)

    # Lanczos basis vectors (output)
    arc.v = Matrix{T}(undef, arc.n, arc.ncv)

    # If info != 0, RESID contains the initial residual vector
    arc.info = ones(BlasInt, 1)
    arc.info_e = ones(BlasInt, 1)
    # Resid contains the initial residual vector
    arc.rng = Random.MersenneTwister(1234)
    arc.resid = rand(arc.rng, arc.n)

    # Workspace
    arc.workd = Vector{T}(undef, 3 * arc.n)
    arc.lworkl = arc.ncv * (arc.ncv + 8)
    arc.workl = Vector{T}(undef, arc.lworkl)
    arc.rwork = Vector{T}()
    arc.iparam = ones(BlasInt, 11)
    arc.iparam[1] = BlasInt(1)
    arc.iparam[3] = BlasInt(arc.maxiter)
    arc.iparam[4] = BlasInt(1)
    arc.iparam[7] = BlasInt(arc.mode)
    
    arc.ipntr = zeros(BlasInt, 11)
    # IDO must be zero on the first call to dsaupd
    arc.ido = zeros(BlasInt, 1)

    # Parameters for _EUPD! routine
    arc.zernm1 = 0:(arc.n-1)
    arc.howmny = "A"
    arc.select = Vector{BlasInt}(undef, arc.ncv)
    arc.d = Vector{T}(undef, arc.nev)
    arc.sigmar = zeros(T, 1)

    # Flags created for ProxSDP use
    arc.converged = false
    arc.arpackerror = false

    return nothing
end

function _update_arc!(arc::ARPACKAlloc{T}, A::Symmetric{T,Matrix{T}}, nev::Int64, iter::Int64) where T

    # Number of eigenvalues of OP to be computed. Needs to be 0 < NEV < N.
    arc.nev = nev

    # How many Lanczos vectors are generated. Needs to be NCV > NEV. It is recommended that NCV >= 2*NEV.
    # After the startup phase in which NEV Lanczos vectors are generated, 
    # the algorithm generates NCV-NEV Lanczos vectors at each subsequent update iteration.
    arc.ncv = max(2 * arc.nev, 10)

    # Lanczos basis vectors (output)
    arc.v = Matrix{T}(undef, arc.n, arc.ncv)

    # Workspace, LWORKL must be at least NCV**2 + 8*NCV
    arc.lworkl = arc.ncv * (arc.ncv + 8)

    # Allocate memory for workl
    arc.workl = Vector{T}(undef, arc.lworkl)

    # Parameters for _EUPD! routine
    arc.select = Vector{BlasInt}(undef, arc.ncv)
    arc.d = Vector{T}(undef, arc.nev)

    # Build linear operator
    arc.Amat = A

    # If info != 0, RESID contains the initial residual vector,
    # If info == 0, randomly initial residual vector is used.
    arc.info[] = BlasInt(0)
    arc.info_e[] = BlasInt(0)

    # Iparam
    fill!(arc.iparam, BlasInt(1))
    arc.iparam[1] = BlasInt(1)

    # IPARAM(3) = MXITER
    # On INPUT:  maximum number of Arnoldi update iterations allowed. 
    # On OUTPUT: actual number of Arnoldi update iterations taken. 
    arc.iparam[3] = BlasInt(arc.maxiter)

    # IPARAM(4) = NB: blocksize to be used in the recurrence.
    # The code currently works only for NB = 1.
    arc.iparam[4] = BlasInt(1)

    # Determines what type of eigenproblem is being solved.
    arc.iparam[7] = BlasInt(arc.mode)

    # 
    fill!(arc.ipntr, BlasInt(0))

    # IDO must be zero on the first call to dsaupd
    fill!(arc.ido, BlasInt(0))

    # Stopping criterion
    # arc.tol = max(arc.tol / (iter / 10.), 1e-10)
    arc.tol = 1e-10
    arc.TOL[] = arc.tol * one(T)

    return nothing
end

function _saupd!(arc::ARPACKAlloc{T})::Nothing where T
    
    while true
        Arpack.saupd(arc.ido, arc.bmat, arc.n, arc.which, arc.nev, Ref(arc.TOL[1]), arc.resid, arc.ncv, arc.v, arc.n,
        arc.iparam, arc.ipntr, arc.workd, arc.workl, arc.lworkl, arc.info)

        if arc.ido[] == 1
            @inbounds @simd for i in 1:arc.n
                arc.x[i] = arc.workd[i-1+arc.ipntr[1]]
            end
            mul!(arc.y, arc.Amat, arc.x)
            @inbounds @simd for i in 1:arc.n
                 arc.workd[i-1+arc.ipntr[2]] = arc.y[i]
            end
            
        elseif arc.ido[] == 99
            # In this case, don't call _EUPD! (according to https://help.scilab.org/docs/5.3.3/en_US/dseupd.html)
            break
        else
            arc.converged = false
            arc.arpackerror = true
            break
        end
    end

    # Check if _AUPD has converged properly
    if !(0 <= arc.info[] <= 1)
        arc.converged = false
        arc.arpackerror = true
    else
        arc.converged = true
    end

    return nothing
end

function _seupd!(arc::ARPACKAlloc{T})::Nothing where T

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

function eig!(arc::ARPACKAlloc, A::Symmetric{T1,Matrix{T1}}, nev::Integer, iter::Int64, warm_start_eig::Bool)::Nothing where T1

    # Initialize parameters and do memory allocation
    @timeit "update_arc" _update_arc!(arc, A, nev, iter)::Nothing

    # Top level reverse communication interface to solve real double precision symmetric problems.
    @timeit "saupd" _saupd!(arc)::Nothing

    # Post processing routine (eigenvalues and eigenvectors purification)
    @timeit "seupd" _seupd!(arc)::Nothing

    return nothing
end