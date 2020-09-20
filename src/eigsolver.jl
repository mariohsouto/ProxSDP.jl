#=

Docs from DSAUPD from https://www.caam.rice.edu/software/ARPACK/UG/node136.html
c-----------------------------------------------------------------------
c\BeginDoc
c
c\Name: dsaupd
c
c\Description: 
c
c  Reverse communication interface for the Implicitly Restarted Arnoldi 
c  Iteration.  For symmetric problems this reduces to a variant of the Lanczos 
c  method.  This method has been designed to compute approximations to a 
c  few eigenpairs of a linear operator OP that is real and symmetric 
c  with respect to a real positive semi-definite symmetric matrix B, 
c  i.e.
c                   
c       B*OP = (OP')*B.  
c
c  Another way to express this condition is 
c
c       < x,OPy > = < OPx,y >  where < z,w > = z'Bw  .
c  
c  In the standard eigenproblem B is the identity matrix.  
c  ( A' denotes transpose of A)
c
c  The computed approximate eigenvalues are called Ritz values and
c  the corresponding approximate eigenvectors are called Ritz vectors.
c
c  dsaupd is usually called iteratively to solve one of the 
c  following problems:
c
c  Mode 1:  A*x = lambda*x, A symmetric 
c           ===> OP = A  and  B = I.
c
c  Mode 2:  A*x = lambda*M*x, A symmetric, M symmetric positive definite
c           ===> OP = inv[M]*A  and  B = M.
c           ===> (If M can be factored see remark 3 below)
c
c  Mode 3:  K*x = lambda*M*x, K symmetric, M symmetric positive semi-definite
c           ===> OP = (inv[K - sigma*M])*M  and  B = M. 
c           ===> Shift-and-Invert mode
c
c  Mode 4:  K*x = lambda*KG*x, K symmetric positive semi-definite, 
c           KG symmetric indefinite
c           ===> OP = (inv[K - sigma*KG])*K  and  B = K.
c           ===> Buckling mode
c
c  Mode 5:  A*x = lambda*M*x, A symmetric, M symmetric positive semi-definite
c           ===> OP = inv[A - sigma*M]*[A + sigma*M]  and  B = M.
c           ===> Cayley transformed mode
c
c  NOTE: The action of w <- inv[A - sigma*M]*v or w <- inv[M]*v
c        should be accomplished either by a direct method
c        using a sparse matrix factorization and solving
c
c           [A - sigma*M]*w = v  or M*w = v,
c
c        or through an iterative method for solving these
c        systems.  If an iterative method is used, the
c        convergence test must be more stringent than
c        the accuracy requirements for the eigenvalue
c        approximations.
c
c\Usage:
c  call dsaupd 
c     ( IDO, BMAT, N, WHICH, NEV, TOL, RESID, NCV, V, LDV, IPARAM,
c       IPNTR, WORKD, WORKL, LWORKL, INFO )
c
c\Arguments
c  IDO     Integer.  (INPUT/OUTPUT)
c          Reverse communication flag.  IDO must be zero on the first 
c          call to dsaupd.  IDO will be set internally to
c          indicate the type of operation to be performed.  Control is
c          then given back to the calling routine which has the
c          responsibility to carry out the requested operation and call
c          dsaupd with the result.  The operand is given in
c          WORKD(IPNTR(1)), the result must be put in WORKD(IPNTR(2)).
c          (If Mode = 2 see remark 5 below)
c          -------------------------------------------------------------
c          IDO =  0: first call to the reverse communication interface
c          IDO = -1: compute  Y = OP * X  where
c                    IPNTR(1) is the pointer into WORKD for X,
c                    IPNTR(2) is the pointer into WORKD for Y.
c                    This is for the initialization phase to force the
c                    starting vector into the range of OP.
c          IDO =  1: compute  Y = OP * Z  and Z = B * X where
c                    IPNTR(1) is the pointer into WORKD for X,
c                    IPNTR(2) is the pointer into WORKD for Y,
c                    IPNTR(3) is the pointer into WORKD for Z.
c          IDO =  2: compute  Y = B * X  where
c                    IPNTR(1) is the pointer into WORKD for X,
c                    IPNTR(2) is the pointer into WORKD for Y.
c          IDO =  3: compute the IPARAM(8) shifts where
c                    IPNTR(11) is the pointer into WORKL for
c                    placing the shifts. See remark 6 below.
c          IDO = 99: done
c          -------------------------------------------------------------
c          After the initialization phase, when the routine is used in 
c          either the "shift-and-invert" mode or the Cayley transform
c          mode, the vector B * X is already available and does not 
c          need to be recomputed in forming OP*X.
c             
c  BMAT    Character*1.  (INPUT)
c          BMAT specifies the type of the matrix B that defines the
c          semi-inner product for the operator OP.
c          B = 'I' -> standard eigenvalue problem A*x = lambda*x
c          B = 'G' -> generalized eigenvalue problem A*x = lambda*B*x
c
c  N       Integer.  (INPUT)
c          Dimension of the eigenproblem.
c
c  WHICH   Character*2.  (INPUT)
c          Specify which of the Ritz values of OP to compute.
c
c          'LA' - compute the NEV largest (algebraic) eigenvalues.
c          'SA' - compute the NEV smallest (algebraic) eigenvalues.
c          'LM' - compute the NEV largest (in magnitude) eigenvalues.
c          'SM' - compute the NEV smallest (in magnitude) eigenvalues. 
c          'BE' - compute NEV eigenvalues, half from each end of the
c                 spectrum.  When NEV is odd, compute one more from the
c                 high end than from the low end.
c           (see remark 1 below)
c
c  NEV     Integer.  (INPUT)
c          Number of eigenvalues of OP to be computed. 0 < NEV < N.
c
c  TOL     Double precision scalar.  (INPUT)
c          Stopping criterion: the relative accuracy of the Ritz value 
c          is considered acceptable if BOUNDS(I) .LE. TOL*ABS(RITZ(I)).
c          If TOL .LE. 0. is passed a default is set:
c          DEFAULT = DLAMCH('EPS')  (machine precision as computed
c                    by the LAPACK auxiliary subroutine DLAMCH).
c
c  RESID   Double precision array of length N.  (INPUT/OUTPUT)
c          On INPUT: 
c          If INFO .EQ. 0, a random initial residual vector is used.
c          If INFO .NE. 0, RESID contains the initial residual vector,
c                          possibly from a previous run.
c          On OUTPUT:
c          RESID contains the final residual vector. 
c
c  NCV     Integer.  (INPUT)
c          Number of columns of the matrix V (less than or equal to N).
c          This will indicate how many Lanczos vectors are generated 
c          at each iteration.  After the startup phase in which NEV 
c          Lanczos vectors are generated, the algorithm generates 
c          NCV-NEV Lanczos vectors at each subsequent update iteration.
c          Most of the cost in generating each Lanczos vector is in the 
c          matrix-vector product OP*x. (See remark 4 below).
c
c  V       Double precision N by NCV array.  (OUTPUT)
c          The NCV columns of V contain the Lanczos basis vectors.
c
c  LDV     Integer.  (INPUT)
c          Leading dimension of V exactly as declared in the calling
c          program.
c
c  IPARAM  Integer array of length 11.  (INPUT/OUTPUT)
c          IPARAM(1) = ISHIFT: method for selecting the implicit shifts.
c          The shifts selected at each iteration are used to restart
c          the Arnoldi iteration in an implicit fashion.
c          -------------------------------------------------------------
c          ISHIFT = 0: the shifts are provided by the user via
c                      reverse communication.  The NCV eigenvalues of
c                      the current tridiagonal matrix T are returned in
c                      the part of WORKL array corresponding to RITZ.
c                      See remark 6 below.
c          ISHIFT = 1: exact shifts with respect to the reduced 
c                      tridiagonal matrix T.  This is equivalent to 
c                      restarting the iteration with a starting vector 
c                      that is a linear combination of Ritz vectors 
c                      associated with the "wanted" Ritz values.
c          -------------------------------------------------------------
c
c          IPARAM(2) = LEVEC
c          No longer referenced. See remark 2 below.
c
c          IPARAM(3) = MXITER
c          On INPUT:  maximum number of Arnoldi update iterations allowed. 
c          On OUTPUT: actual number of Arnoldi update iterations taken. 
c
c          IPARAM(4) = NB: blocksize to be used in the recurrence.
c          The code currently works only for NB = 1.
c
c          IPARAM(5) = NCONV: number of "converged" Ritz values.
c          This represents the number of Ritz values that satisfy
c          the convergence criterion.
c
c          IPARAM(6) = IUPD
c          No longer referenced. Implicit restarting is ALWAYS used. 
c
c          IPARAM(7) = MODE
c          On INPUT determines what type of eigenproblem is being solved.
c          Must be 1,2,3,4,5; See under \Description of dsaupd for the 
c          five modes available.
c
c          IPARAM(8) = NP
c          When ido = 3 and the user provides shifts through reverse
c          communication (IPARAM(1)=0), dsaupd returns NP, the number
c          of shifts the user is to provide. 0 < NP <=NCV-NEV. See Remark
c          6 below.
c
c          IPARAM(9) = NUMOP, IPARAM(10) = NUMOPB, IPARAM(11) = NUMREO,
c          OUTPUT: NUMOP  = total number of OP*x operations,
c                  NUMOPB = total number of B*x operations if BMAT='G',
c                  NUMREO = total number of steps of re-orthogonalization.        
c
c  IPNTR   Integer array of length 11.  (OUTPUT)
c          Pointer to mark the starting locations in the WORKD and WORKL
c          arrays for matrices/vectors used by the Lanczos iteration.
c          -------------------------------------------------------------
c          IPNTR(1): pointer to the current operand vector X in WORKD.
c          IPNTR(2): pointer to the current result vector Y in WORKD.
c          IPNTR(3): pointer to the vector B * X in WORKD when used in 
c                    the shift-and-invert mode.
c          IPNTR(4): pointer to the next available location in WORKL
c                    that is untouched by the program.
c          IPNTR(5): pointer to the NCV by 2 tridiagonal matrix T in WORKL.
c          IPNTR(6): pointer to the NCV RITZ values array in WORKL.
c          IPNTR(7): pointer to the Ritz estimates in array WORKL associated
c                    with the Ritz values located in RITZ in WORKL.
c          Note: IPNTR(8:10) is only referenced by dseupd. See Remark 2.
c          IPNTR(8): pointer to the NCV RITZ values of the original system.
c          IPNTR(9): pointer to the NCV corresponding error bounds.
c          IPNTR(10): pointer to the NCV by NCV matrix of eigenvectors
c                     of the tridiagonal matrix T. Only referenced by
c                     dseupd if RVEC = .TRUE. See Remarks.
c          Note: IPNTR(8:10) is only referenced by dseupd. See Remark 2.
c          IPNTR(11): pointer to the NP shifts in WORKL. See Remark 6 below.
c          -------------------------------------------------------------
c          
c  WORKD   Double precision work array of length 3*N.  (REVERSE COMMUNICATION)
c          Distributed array to be used in the basic Arnoldi iteration
c          for reverse communication.  The user should not use WORKD 
c          as temporary workspace during the iteration. Upon termination
c          WORKD(1:N) contains B*RESID(1:N). If the Ritz vectors are desired
c          subroutine dseupd uses this output.
c          See Data Distribution Note below.  
c
c  WORKL   Double precision work array of length LWORKL.  (OUTPUT/WORKSPACE)
c          Private (replicated) array on each PE or array allocated on
c          the front end.  See Data Distribution Note below.
c
c  LWORKL  Integer.  (INPUT)
c          LWORKL must be at least NCV**2 + 8*NCV .
c
c  INFO    Integer.  (INPUT/OUTPUT)
c          If INFO .EQ. 0, a randomly initial residual vector is used.
c          If INFO .NE. 0, RESID contains the initial residual vector,
c                          possibly from a previous run.
c          Error flag on output.
c          =  0: Normal exit.
c          =  1: Maximum number of iterations taken.
c                All possible eigenvalues of OP has been found. IPARAM(5)  
c                returns the number of wanted converged Ritz values.
c          =  2: No longer an informational error. Deprecated starting
c                with release 2 of ARPACK.
c          =  3: No shifts could be applied during a cycle of the 
c                Implicitly restarted Arnoldi iteration. One possibility 
c                is to increase the size of NCV relative to NEV. 
c                See remark 4 below.
c          = -1: N must be positive.
c          = -2: NEV must be positive.
c          = -3: NCV must be greater than NEV and less than or equal to N.
c          = -4: The maximum number of Arnoldi update iterations allowed
c                must be greater than zero.
c          = -5: WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.
c          = -6: BMAT must be one of 'I' or 'G'.
c          = -7: Length of private work array WORKL is not sufficient.
c          = -8: Error return from trid. eigenvalue calculation;
c                Informational error from LAPACK routine dsteqr.
c          = -9: Starting vector is zero.
c          = -10: IPARAM(7) must be 1,2,3,4,5.
c          = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatable.
c          = -12: IPARAM(1) must be equal to 0 or 1.
c          = -13: NEV and WHICH = 'BE' are incompatable.
c          = -9999: Could not build an Arnoldi factorization.
c                   IPARAM(5) returns the size of the current Arnoldi
c                   factorization. The user is advised to check that
c                   enough workspace and array storage has been allocated.
c
c
c\Remarks
c  1. The converged Ritz values are always returned in ascending 
c     algebraic order.  The computed Ritz values are approximate
c     eigenvalues of OP.  The selection of WHICH should be made
c     with this in mind when Mode = 3,4,5.  After convergence, 
c     approximate eigenvalues of the original problem may be obtained 
c     with the ARPACK subroutine dseupd. 
c
c  2. If the Ritz vectors corresponding to the converged Ritz values
c     are needed, the user must call dseupd immediately following completion
c     of dsaupd. This is new starting with version 2.1 of ARPACK.
c
c  3. If M can be factored into a Cholesky factorization M = LL'
c     then Mode = 2 should not be selected.  Instead one should use
c     Mode = 1 with  OP = inv(L)*A*inv(L').  Appropriate triangular 
c     linear systems should be solved with L and L' rather
c     than computing inverses.  After convergence, an approximate
c     eigenvector z of the original problem is recovered by solving
c     L'z = x  where x is a Ritz vector of OP.
c
c  4. At present there is no a-priori analysis to guide the selection
c     of NCV relative to NEV.  The only formal requirement is that NCV > NEV.
c     However, it is recommended that NCV .ge. 2*NEV.  If many problems of
c     the same type are to be solved, one should experiment with increasing
c     NCV while keeping NEV fixed for a given test problem.  This will 
c     usually decrease the required number of OP*x operations but it
c     also increases the work and storage required to maintain the orthogonal
c     basis vectors.   The optimal "cross-over" with respect to CPU time
c     is problem dependent and must be determined empirically.
c
c  5. If IPARAM(7) = 2 then in the Reverse communication interface the user
c     must do the following. When IDO = 1, Y = OP * X is to be computed.
c     When IPARAM(7) = 2 OP = inv(B)*A. After computing A*X the user
c     must overwrite X with A*X. Y is then the solution to the linear set
c     of equations B*Y = A*X.
c
c  6. When IPARAM(1) = 0, and IDO = 3, the user needs to provide the 
c     NP = IPARAM(8) shifts in locations: 
c     1   WORKL(IPNTR(11))           
c     2   WORKL(IPNTR(11)+1)         
c                        .           
c                        .           
c                        .      
c     NP  WORKL(IPNTR(11)+NP-1). 
c
c     The eigenvalues of the current tridiagonal matrix are located in 
c     WORKL(IPNTR(6)) through WORKL(IPNTR(6)+NCV-1). They are in the
c     order defined by WHICH. The associated Ritz estimates are located in
c     WORKL(IPNTR(8)), WORKL(IPNTR(8)+1), ... , WORKL(IPNTR(8)+NCV-1).
c
c-----------------------------------------------------------------------------
=#
mutable struct ARPACKAlloc{T}

    n::Int
    nev::Int
    ncv::Int
    maxiter::Int
    bmat::String
    which::String
    mode::Int
    Amat::Symmetric{T,Matrix{T}}
    lworkl::Int
    TOL::Base.RefValue{T}
    v::Matrix{T}
    workd::Vector{T}
    workl::Vector{T}
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

function ARPACKAlloc(T::DataType, n::Int64, opt::Options)::ARPACKAlloc
    arc = ARPACKAlloc{T}()
    @timeit "init_arc" _init_arc!(arc, Symmetric(Matrix{T}(I, n, n), :U), 1, n, opt)
    return arc
end

function _init_arc!(arc::ARPACKAlloc{T}, A::Symmetric{T,Matrix{T}}, nev::Int64, n::Int64, opt::Options) where T

    # IDO - Integer.  (INPUT/OUTPUT)  (in julia its is a integer vector)
    # Reverse communication flag.
    # IDO must be zero on the first call to dsaupd.
    arc.ido = zeros(BlasInt, 1)

    # BMAT - Character*1.  (INPUT)
    # Standard eigenvalue problem
    # 'I' -> standard eigenvalue problem A*x = lambda*x
    arc.bmat = "I"

    # N - Integer.  (INPUT)
    # Dimension of the eigenproblem
    arc.n = n

    # WHICH - Character*2.  (INPUT)
    # 'LA' - compute the NEV largest (algebraic) eigenvalues.
    arc.which = "LA"

    # NEV - Integer.  (INPUT)
    # Number of eigenvalues of OP to be computed. 0 < NEV < N.
    arc.nev = nev

    # TOL - Double precision scalar.  (INPUT)
    # Stopping criterion
    arc.TOL = Ref(opt.arpack_tol)

    #  RESID - Double precision array of length N.  (INPUT/OUTPUT)
    # Resid contains the initial residual vector
    # Double precision array of length N.  (INPUT/OUTPUT)
    # On INPUT: 
    # If INFO .EQ. 0, a random initial residual vector is used.
    # If INFO .NE. 0, RESID contains the initial residual vector,
    #                 possibly from a previous run.
    # On OUTPUT:
    # RESID contains the final residual vector. 
    # reset the curent mersenne twister to keep determinism
    if opt.arpack_resid_init == 2
        arc.rng = Random.MersenneTwister(opt.arpack_resid_seed)
        arc.resid = Random.rand(arc.rng, arc.n)
    elseif opt.arpack_resid_init == 3
        arc.rng = Random.MersenneTwister(opt.arpack_resid_seed)
        arc.resid = Random.randn(arc.rng, arc.n)
        LinearAlgebra.normalize!(arc.resid)
    elseif opt.arpack_resid_init == 1
        arc.resid = ones(arc.n)
    else
        arc.resid = zeros(arc.n)
    end

    # NCV - Integer.  (INPUT)
    # How many Lanczos vectors are generated
    # Remark
    # 4. At present there is no a-priori analysis to guide the selection
    #    of NCV relative to NEV.  The only formal requirement is that NCV > NEV.
    #    However, it is recommended that NCV .ge. 2*NEV.  If many problems of
    #    the same type are to be solved, one should experiment with increasing
    #    NCV while keeping NEV fixed for a given test problem.  This will 
    #    usually decrease the required number of OP*x operations but it
    #    also increases the work and storage required to maintain the orthogonal
    #    basis vectors.   The optimal "cross-over" with respect to CPU time
    #    is problem dependent and must be determined empirically.
    arc.ncv = max(2 * arc.nev + 1, opt.arpack_min_lanczos)
    # TODO: 10 might be a way to tight bound
    # why not 20

    # V - Double precision N by NCV array.  (OUTPUT)
    # Double precision N by NCV array.  (OUTPUT)
    # The NCV columns of V contain the Lanczos basis vectors.
    # Lanczos basis vectors (output)
    arc.v = Matrix{T}(undef, arc.n, arc.ncv)

    #  LDV     Integer.  (INPUT)
    #          Leading dimension of V exactly as declared in the calling
    #          program.
    # same as N here !!!
    # arc.n = n

    # IPARAM  Integer array of length 11.  (INPUT/OUTPUT)
    arc.iparam = zeros(BlasInt, 11)
    # IPARAM(1) = ISHIFT
    # ISHIFT = 1: exact shifts
    arc.iparam[1] = BlasInt(1)
    # IPARAM(2) = LEVEC
    # IGNORED
    # IPARAM(3) = MXITER
    # On INPUT:  maximum number of Arnoldi update iterations allowed. 
    # On OUTPUT: actual number of Arnoldi update iterations taken.
    arc.iparam[3] = BlasInt(opt.arpack_max_iter)
    # IPARAM(4) = NB: blocksize to be used in the recurrence.
    # The code currently works only for NB = 1.
    arc.iparam[4] = BlasInt(1)
    # IPARAM(5) = NCONV: number of "converged" Ritz values.
    # IGNORED -> OUTPUT
    # IPARAM(6) = IUPD
    # IGNORED
    # IPARAM(7) = MODE
    # On INPUT determines what type of eigenproblem is being solved.
    # Must be 1,2,3,4,5; See under \Description of dsaupd for the 
    # five modes available.
    # Mode 1:  A*x = lambda*x, A symmetric 
    #          ===> OP = A  and  B = I.
    arc.mode = 1
    arc.iparam[7] = BlasInt(arc.mode)
    # IPARAM(8) = NP
    # IGNORED BY US (only need if user provides shift)
    # IPARAM(9) = NUMOP, IPARAM(10) = NUMOPB, IPARAM(11) = NUMREO,
    # IGNORED -> are all OUTPUTS

    # IPNTR   Integer array of length 11.  (OUTPUT)
    arc.ipntr = zeros(BlasInt, 11)

    # WORKD   Double precision work array of length 3*N.  (REVERSE COMMUNICATION)
    arc.workd = Vector{T}(undef, 3 * arc.n)
    # LWORKL  Integer.  (INPUT)
    # LWORKL must be at least NCV**2 + 8*NCV .
    arc.lworkl = arc.ncv^2 + 8*arc.ncv
    # WORKL   Double precision work array of length LWORKL.  (OUTPUT/WORKSPACE)
    arc.workl = Vector{T}(undef, arc.lworkl)

    # INFO    Integer.  (INPUT/OUTPUT)
    #         If INFO .EQ. 0, a randomly initial residual vector is used.
    #         If INFO .NE. 0, RESID contains the initial residual vector,
    #                         possibly from a previous run.
    if opt.arpack_resid_init == 0
        arc.info = zeros(BlasInt, 1)
        arc.info_e = zeros(BlasInt, 1)
    else
        arc.info = ones(BlasInt, 1)
        arc.info_e = ones(BlasInt, 1)
    end

    # Build linear operator
    arc.Amat = A
    arc.x = Vector{T}(undef, arc.n)
    arc.y = Vector{T}(undef, arc.n)

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

function _update_arc!(arc::ARPACKAlloc{T}, A::Symmetric{T,Matrix{T}}, nev::Int64, opt::Options, up_ncv::Bool) where T

    need_resize = up_ncv
    if !need_resize
        need_resize = nev != arc.nev
    end
    if !need_resize
        need_resize = arc.ncv != max(2 * arc.nev + 1, opt.arpack_min_lanczos)
    end

    # IDO must be zero on the first call to dsaupd
    fill!(arc.ido, BlasInt(0))

    # Number of eigenvalues of OP to be computed. Needs to be 0 < NEV < N.
    arc.nev = nev

    # Stopping criterion
    arc.TOL[] = opt.arpack_tol

    #  RESID - Double precision array of length N.  (INPUT/OUTPUT)
    if opt.arpack_reset_resid
        if opt.arpack_resid_init == 2
            Random.seed!(arc.rng, opt.arpack_resid_seed)
            Random.rand!(arc.rng, arc.resid)
        elseif opt.arpack_resid_init == 3
            Random.seed!(arc.rng, opt.arpack_resid_seed)
            Random.randn!(arc.rng, arc.resid)
            LinearAlgebra.normalize!(arc.resid)
        elseif opt.arpack_resid_init == 1
            fill!(arc.resid, 1.0)
        else
            fill!(arc.resid, 0.0)
        end
    end

    # How many Lanczos vectors are generated. Needs to be NCV > NEV. It is recommended that NCV >= 2*NEV.
    # After the startup phase in which NEV Lanczos vectors are generated, 
    # the algorithm generates NCV-NEV Lanczos vectors at each subsequent update iteration.
    if need_resize
        if up_ncv
            arc.ncv += max(arc.nev, 10)
        else
            arc.ncv = max(2 * arc.nev + 1, opt.arpack_min_lanczos)
        end
    end

    # Lanczos basis vectors (output)
    if need_resize
        arc.v = Matrix{T}(undef, arc.n, arc.ncv)
    end

    # Iparam
    fill!(arc.iparam, BlasInt(0))
    # IPARAM(1) = ISHIFT
    # ISHIFT = 1: exact shifts
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

    # IPNTR   Integer array of length 11.  (OUTPUT)
    fill!(arc.ipntr, BlasInt(0))

    # Workspace, LWORKL must be at least NCV**2 + 8*NCV
    if need_resize
        arc.lworkl = arc.ncv * (arc.ncv + 8)
    end

    # Allocate memory for workl
    if need_resize
        arc.workl = Vector{T}(undef, arc.lworkl)
    end

    # INFO    Integer.  (INPUT/OUTPUT)
    #         If INFO .EQ. 0, a randomly initial residual vector is used.
    #         If INFO .NE. 0, RESID contains the initial residual vector,
    #                         possibly from a previous run.
    if opt.arpack_resid_init == 0
        arc.info[] = zero(BlasInt)
        arc.info_e[] = zero(BlasInt)
    else
        arc.info[] = one(BlasInt)
        arc.info_e[] = one(BlasInt)
    end

    # Build linear operator
    arc.Amat = A

    # Parameters for _EUPD! routine
    if need_resize
        arc.select = Vector{BlasInt}(undef, arc.ncv)
        arc.d = Vector{T}(undef, arc.nev)
    end

    # Flags created for ProxSDP use
    arc.converged = false
    arc.arpackerror = false

    return nothing
end

function _saupd!(arc::ARPACKAlloc{T})::Nothing where T

    while true
        Arpack.saupd(arc.ido, arc.bmat, arc.n, arc.which, arc.nev, arc.TOL, arc.resid, arc.ncv, arc.v, arc.n,
        arc.iparam, arc.ipntr, arc.workd, arc.workl, arc.lworkl, arc.info)

        if arc.ido[] == 1
            @simd for i in 1:arc.n
                @inbounds arc.x[i] = arc.workd[i-1+arc.ipntr[1]]
            end
            mul!(arc.y, arc.Amat, arc.x)
            @simd for i in 1:arc.n
                 @inbounds arc.workd[i-1+arc.ipntr[2]] = arc.y[i]
            end
            # using views
            # x = view(arc.workd, arc.ipntr[1] .+ arc.zernm1)
            # y = view(arc.workd, arc.ipntr[2] .+ arc.zernm1)
            # mul!(y, arc.Amat, x)
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

    # check convergence for all ritz pairs
    # https://github.com/JuliaLinearAlgebra/Arpack.jl/blob/a7cdb6d7f19f076f5fadd8b58a9c5a061c48322f/src/Arpack.jl#L188
    # @assert arc.nev <= arc.iparam[5]
    # if arc.nev > arc.iparam[5]
    #     arc.converged = false
    #     @show arc.nev , arc.iparam[5]
    # end

    return nothing
end

function _seupd!(arc::ARPACKAlloc{T})::Nothing where T

    # Check if _AUPD has converged properly
    if !arc.converged
        arc.converged = false
        return nothing
    end

    Arpack.seupd(true, arc.howmny, arc.select, arc.d, arc.v, arc.n, arc.sigmar,
    arc.bmat, arc.n, arc.which, arc.nev, arc.TOL, arc.resid, arc.ncv, arc.v, arc.n,
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

function eig!(arc::ARPACKAlloc, A::Symmetric{T,Matrix{T}}, nev::Integer, opt::Options)::Nothing where T

    up_ncv = false
    for i in 1:1
        # Initialize parameters and do memory allocation
        @timeit "update_arc" _update_arc!(arc, A, nev, opt, up_ncv)::Nothing

        # Top level reverse communication interface to solve real double precision symmetric problems.
        @timeit "saupd" _saupd!(arc)::Nothing

        if arc.nev > arc.iparam[5]
            up_ncv = true
        else
            break
        end
    end

    # Post processing routine (eigenvalues and eigenvectors purification)
    @timeit "seupd" _seupd!(arc)::Nothing

    return nothing
end