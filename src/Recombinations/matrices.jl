"""
    RecombineMatrix{T} <: AbstractMatrix{T}

Matrix for transformation from coefficients of the recombined basis, to the
corresponding B-spline basis coefficients.

# Extended help

The transformation matrix ``M`` is defined by
```math
ϕ_j = M_{ij} b_i,
```
where ``b_j(x)`` and ``ϕ_i(x)`` are elements of the B-spline and recombined
bases, respectively.

This matrix allows to pass from known coefficients ``u_j`` in the recombined
basis ``ϕ_j``, to the respective coefficients ``v_i`` in the B-spline basis
``b_i``:
```math
\\bm{v} = \\mathbf{M} \\bm{u}.
```

Note that the matrix is not square: it has dimensions ``N × M``, where ``N``
is the length of the B-spline basis, and ``M = N - δ`` is that of the
recombined basis (see [`RecombinedBSplineBasis`](@ref) for details).

Due to the local support of B-splines, basis recombination can be performed
by combining just a small set of B-splines near the boundaries (as discussed in
[`RecombinedBSplineBasis`](@ref)).
This leads to a recombination matrix which is almost a diagonal of ones, plus
a few extra super- and sub-diagonal elements in the upper left and lower right
corners, respectively.
The matrix is stored in a memory-efficient way that also allows fast access to
its elements.

Efficient implementations of matrix-vector products (using the `*` operator or
`LinearAlgebra.mul!`) and of left division of vectors (using `\\` or
`LinearAlgebra.ldiv!`) are included.
These two operations can be used to transform between coefficients in the
original and recombined bases.

Note that, since the recombined basis forms a subspace of the original basis
(which is related to the rectangular shape of the matrix),
it is generally not possible to obtain recombined coefficients from original
coefficients, unless the latter already satisfy the constraints encoded in the
recombined basis.
The left division operation will throw a [`NoUniqueSolutionError`](@ref) if that
is not the case.

---

    RecombineMatrix(ops::Tuple{Vararg{AbstractDifferentialOp}},
                    B::BSplineBasis, [T])

Construct recombination matrix describing a B-spline basis recombination.

The default element type `T` is generally `Float64`, except for specific
differential operators which yield a matrix of zeroes and ones, for which `Bool`
is the default.

See the [`RecombinedBSplineBasis`](@ref) constructor for details on the
`ops` argument.
"""
struct RecombineMatrix{T,
                       DiffOps <: Tuple{Vararg{AbstractDifferentialOp}},
                       n, n1,   # nc = n1 - n
                       Corner <: SMatrix{n1,n,T}} <: AbstractMatrix{T}
    ops :: DiffOps  # list of differential operators for BCs
    M :: Int      # length of recombined basis (= N - 2nc)
    N :: Int      # length of B-spline basis
    ul :: Corner  # upper-left corner of matrix, size (n1, n)
    lr :: Corner  # lower-right corner of matrix, size (n1, n)
    function RecombineMatrix(ops::Tuple{Vararg{AbstractDifferentialOp}},
                             N::Integer, ul::SMatrix{n1,n}, lr::SMatrix{n1,n}) where {n1,n}
        if n1 <= n
            throw(ArgumentError("matrices must have dimensions (m, n) with m > n"))
        end
        M = N - 2 * length(ops)
        T = eltype(ul)
        Corner = typeof(ul)
        Ops = typeof(ops)
        new{T, Ops, n, n1, Corner}(ops, M, N, ul, lr)
    end
end

# Default element type of recombination matrix.
# In some specific cases we can use Bool...
_default_eltype(::Vararg{AbstractDifferentialOp}) = Float64
_default_eltype(::Derivative{0}) = Bool  # Dirichlet BCs
_default_eltype(::Derivative{1}) = Bool  # Neumann BCs

# Case (D(0), D(1), D(2), ...)
_default_eltype(::Derivative{0}, ::Derivative{1}, ::Vararg{Derivative}) = Bool

RecombineMatrix(ops::Tuple{Vararg{AbstractDifferentialOp}}, B::BSplineBasis) =
    RecombineMatrix(ops, B, _default_eltype(ops...))

# Specialisation for Dirichlet BCs.
function RecombineMatrix(op::Tuple{Derivative{0}}, B::BSplineBasis,
                         ::Type{T}) where {T}
    _check_bspline_order(op, B)
    N = length(B)
    ul = SMatrix{1,0,T}()
    lr = copy(ul)
    RecombineMatrix(op, N, ul, lr)
end

# Specialisation for Neumann BCs.
function RecombineMatrix(op::Tuple{Derivative{1}}, B::BSplineBasis,
                         ::Type{T}) where {T}
    _check_bspline_order(op, B)
    N = length(B)
    ul = SMatrix{2,1,T}(1, 1)
    lr = copy(ul)
    RecombineMatrix(op, N, ul, lr)
end

# Generalisation to higher orders and to more general differential operators
# (this includes Robin BCs, for instance).
#
# If more than one operator is passed, they must be of the form
#
#    (D(0), D(1), ..., D(m - 2), D(m - 1), [D(n)]),
#
# where the last D(n) is optional, and satisfies n ≥ m + 1.
# In this case, the first `m` B-splines are dropped, and the next `n - m + 1`
# B-splines are recombined into `q = n - m` functions.
#
# That last operator is allowed to be a linear combination of `Derivative`s.
# In that case, `n` is the maximum degree of the operator.
function RecombineMatrix(ops::Tuple{Vararg{AbstractDifferentialOp}},
                         B::BSplineBasis, ::Type{T}) where {T}
    _check_bspline_order(ops, B)
    op = last(ops)
    n = max_order(op)
    m = _bsplines_to_drop(ops...)
    _make_matrix(Val(n + 1), Val(m), ops, B, T)
end

# Case of mixed BCs (D(0), D(1), ..., D(n)).
# B-splines are dropped, and new functions are not created.
# This is a generalisation of the Dirichlet case.
function _make_matrix(::Val{n}, ::Val{n}, ops, B, ::Type{T}) where {n,T}
    Nc = length(ops)
    @assert Nc >= 2  # case Nc = 1 is treated by different functions
    N = length(B)
    ul = SMatrix{Nc,0,T}()
    lr = copy(ul)
    RecombineMatrix(ops, N, ul, lr)
end

# Case of single BC, or mixed BCs where the last one is "different" from the
# others.
function _make_matrix(::Val{n1}, ::Val{m}, ops, B, ::Type{T}) where {n1,m,T}
    n = n1 - 1
    q = n - m
    @assert q >= 1

    Nc = length(ops)  # this is the number of constraints
    a, b = boundaries(B)
    Ca = zeros(T, q + Nc, q)
    Cb = copy(Ca)

    N = length(B)
    op = last(ops)

    let x = a
        # Indices of B-splines to recombine.
        is = ntuple(d -> m + d, Val(q + 1))

        # Coefficients of odd derivatives must change sign, since d/dn = -d/dx
        # on the left boundary.
        opn = dot(op, LeftNormal())

        # Evaluate n-th derivatives of bⱼ at the boundary.
        # TODO replace with analytical formula?
        bs = evaluate.(B, is, x, opn)
        for l = 1:q
            b0, b1 = bs[l], bs[l + 1]
            @assert !(b0 ≈ b1)
            r = 2 / (b0 - b1)  # normalisation factor
            Ca[l + Nc - 1, l] = -b1 * r
            Ca[l + Nc, l] = b0 * r
        end
    end

    let x = b
        M = N - m  # index of last undropped B-spline
        is = ntuple(d -> M - d + 1, Val(q + 1))
        opn = dot(op, RightNormal())
        @assert opn === op  # the right normal is in the x direction
        bs = evaluate.(B, is, x, opn)
        for l = 1:q
            b0, b1 = bs[q - l + 2], bs[q - l + 1]
            @assert !(b0 ≈ b1)
            r = 2 / (b0 - b1)
            Cb[l, l] = -b1 * r
            Cb[l + 1, l] = b0 * r
        end
    end

    ul = SMatrix{q + Nc, q}(Ca)
    lr = SMatrix{q + Nc, q}(Cb)

    RecombineMatrix(ops, N, ul, lr)
end

# Verify that the B-spline order is compatible with the given differential
# operators.
function _check_bspline_order(ops::Tuple, B::BSplineBasis)
    n = max_order(ops...)
    k = order(B)
    if n >= k
        throw(ArgumentError(
            "cannot resolve operators $ops with B-splines of order $k"))
    end
    nothing
end

# Single BC: no B-splines are dropped.
# (Except if op = Derivative{0}, but that case is treated separately.)
_bsplines_to_drop(op::AbstractDifferentialOp) = 0

_unval(::Val{n}) where {n} = n
_bsplines_to_drop(ops...) = _bsplines_to_drop(Val(0), ops...) |> _unval

function _bsplines_to_drop(::Val{n}, a, b, etc...) where {n}
    if a !== Derivative(n)
        throw(ArgumentError("unsupported combination of boundary conditions"))
    end
    _bsplines_to_drop(Val(n + 1), b, etc...)
end

# This is the case of the last operator, which is allowed to be something
# different than Derivative{n}.
function _bsplines_to_drop(::Val{n}, op) where {n}
    if op === Derivative(n)
        return Val(n + 1)
    end
    if max_order(op) ≤ n
        throw(ArgumentError("order of last operator is too low"))
    end
    Val(n)
end

# Unsupported cases
RecombineMatrix(ops::Tuple{Vararg{AbstractDifferentialOp}}, args...) =
    throw(ArgumentError(
        "boundary condition combination is currently unsupported: $ops"))

Base.size(A::RecombineMatrix) = (A.N, A.M)

constraints(A::RecombineMatrix) = (A.ops, A.ops)
constraints(::UniformScaling) = ((), ())

num_constraints(A::RecombineMatrix) = map(length, constraints(A))
num_constraints(::UniformScaling) = (0, 0)  # case of non-recombined bases

# Returns (left, right) tuple with maximum BC order on each boundary.
_max_order(A::RecombineMatrix) = map(ops -> max_order(ops...), constraints(A))

# Returns number of basis functions that are recombined from the original basis
# near each boundary.
# For instance, Neumann BCs have a single recombined function, ϕ_1 = b_1 + b_2;
# while mixed Dirichlet + Neumann have none, ϕ_1 = b_3.
num_recombined(A::RecombineMatrix) =
    map((m, c) -> m + 1 - c, _max_order(A), num_constraints(A))
num_recombined(::UniformScaling) = (0, 0)

# j: index in recombined basis
# M: length of recombined basis
@inline function which_recombine_block(A::RecombineMatrix, j)
    @boundscheck checkbounds(A, :, j)
    M = A.M
    nl, nr = num_recombined(A)  # left/right number of recombined basis functions
    j <= nl && return 1
    j > M - nr && return 3
    2
end

"""
    nzrows(A::RecombineMatrix, col::Integer) -> UnitRange{Int}

Returns the range of row indices `i` such that `A[i, col]` is non-zero.
"""
@propagate_inbounds function nzrows(A::RecombineMatrix,
                                    j::Integer) :: UnitRange{Int}
    block = which_recombine_block(A, j)
    j += num_constraints(A)[1]
    if block == 1
        (j - 1):j
    elseif block == 2
        j:j  # shifted diagonal of ones
    else
        j:(j + 1)
    end
end

# Pretty-printing, adapted from BandedMatrices.jl code.
function Base.replace_in_print_matrix(
        A::RecombineMatrix, i::Integer, j::Integer, s::AbstractString)
    iszero(A[i, j]) ? Base.replace_with_centered_mark(s) : s
end

@inline function Base.getindex(A::RecombineMatrix, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    T = eltype(A)
    @inbounds block = which_recombine_block(A, j)

    cl, cr = num_constraints(A)  # left/right constraints

    if block == 2
        return T(i == j + cl)  # δ_{i, j+c}
    end

    if block == 1
        C = A.ul
        if i <= size(C, 1)
            return @inbounds C[i, j] :: T
        end
        return zero(T)
    end

    # The lower-right corner starts at column h + 1.
    M = size(A, 2)
    nl, nr = num_recombined(A)
    h = M - nr

    @assert j > h
    C = A.lr
    ii = i - h - cr
    if ii ∈ axes(C, 1)
        return @inbounds C[ii, j - h] :: T
    end

    zero(T)
end

# Efficient implementation of matrix-vector multiplication.
# Generally much faster than using a regular sparse array.
function LinearAlgebra.mul!(y::AbstractVector, A::RecombineMatrix,
                            x::AbstractVector)
    checkdims_mul(y, A, x)
    N, M = size(A)

    nl, nr = num_recombined(A)
    cl, cr = num_constraints(A)
    n1 = nl + cl
    h = M - nr

    @inbounds y[1:n1] = A.ul * @view x[1:nl]

    for i = (nl + 1):h
        @inbounds y[i + cl] = x[i]
    end

    @inbounds y[(N - nr - cr + 1):N] = A.lr * @view x[(h + 1):M]

    y
end

# Five-argument mul!.
# Note that, since I have this, I could remove the 3-argument mul!  (which is
# equal to the 5-argument one with α = 1 and β = 0), but it would be a bit more
# inefficient to compute.
function LinearAlgebra.mul!(y::AbstractVector, A::RecombineMatrix,
                            x::AbstractVector, α::Number, β::Number)
    checkdims_mul(y, A, x)
    N, M = size(A)

    nl, nr = num_recombined(A)
    cl, cr = num_constraints(A)
    n1 = nl + cl
    h = M - nr

    @inbounds y[1:n1] = α * A.ul * view(x, 1:nl) + β * SVector{n1}(view(y, 1:n1))

    for i = (nl + 1):h
        @inbounds y[i + cl] = α * x[i] + β * y[i + cl]
    end

    js = (N - nr - cr + 1):N
    @inbounds y[js] =
        α * A.lr * view(x, (h + 1):M) + β * SVector{n1}(view(y, js))

    y
end

function LinearAlgebra.ldiv!(x::AbstractVector, A::RecombineMatrix,
                             y::AbstractVector)
    checkdims_mul(y, A, x)
    N, M = size(A)

    nl, nr = num_recombined(A)
    cl, cr = num_constraints(A)
    n1 = nl + cl
    h = M - nr

    @inbounds x[1:nl] = _ldiv_unique_solution(A.ul, view(y, 1:n1))

    for i = (nl + 1):h
        @inbounds x[i] = y[i + cl]
    end

    js = (N - nr - cr + 1):N
    @inbounds x[(h + 1):M] = _ldiv_unique_solution(A.lr, view(y, js))

    x
end

function LinearAlgebra.:\(A::RecombineMatrix, y::AbstractVector)
    x = similar(y, size(A, 2))
    ldiv!(x, A, y)
end

"""
    NoUniqueSolutionError <: Exception

Exception thrown when solving linear system using [`RecombineMatrix`](@ref),
when the system has no unique solution.
"""
struct NoUniqueSolutionError <: Exception end

function Base.showerror(io::IO, ::NoUniqueSolutionError)
    print(io,
          "overdetermined system has no unique solution.",
          " This means that the given function expanded in the parent basis",
          " has no exact representation (i.e. cannot be expanded) in the recombined basis."
         )
end

# Solve A \ y for overdetermined system with rectangular matrix A, assuming that
# the system has exactly one solution. Throws error if that's not the case.
function _ldiv_unique_solution(A::SMatrix, y::AbstractVector)
    N, M = size(A)
    @assert length(y) == N
    @assert N > M "system must be overdetermined"

    # Solve the first non-trivial M equations, skipping rows of zeroes.
    n = 1
    if M != 0
        @inbounds while n ≤ N && all(iszero.(A[n, :]))
            n += 1
        end
    end
    m = n + M - 1
    @assert m <= N
    @inbounds Am = SMatrix{M,M}(view(A, n:m, :))
    @inbounds ym = SVector{M}(view(y, n:m))
    x = Am \ ym

    # Check that equations (m+1):N are satisfied.
    @inbounds for i = (m + 1):N
        res = zero(eltype(x))
        for j in axes(A, 2)
            res += A[i, j] * x[j]
        end
        res ≈ y[i] || throw(NoUniqueSolutionError())
    end

    x
end

function checkdims_mul(y, A, x)
    M, N = size(A)
    if length(y) != M
        throw(DimensionMismatch("first dimension of A, $M, " *
                                "does not match length of y, $(length(y))"))
    end
    if length(x) != N
        throw(DimensionMismatch("second dimension of A, $N, " *
                                "does not match length of x, $(length(x))"))
    end
    nothing
end
