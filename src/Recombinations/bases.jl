"""
    RecombinedBSplineBasis{k, T}

Functional basis defined from the recombination of a [`BSplineBasis`](@ref)
in order to satisfy certain homogeneous boundary conditions (BCs).

# Extended help

The basis recombination technique is a common way of applying BCs in Galerkin
methods. It is described for instance in Boyd 2000 (ch. 6), in the context of
a Chebyshev basis. In this approach, the original basis, ``\\{b_i(x), 1 ≤ i ≤ N\\}``,
is "recombined" into a new basis, ``\\{ϕ_j(x), 1 ≤ j ≤ M\\}``, so that each
basis function ``ϕ_j`` individually satisfies the chosen BCs.

The length ``M`` of the recombined basis is always smaller than the length ``N``
of the original basis.
The difference, ``δ = N - M``, is equal to the number of boundary conditions.
In the simplest (and most common) case, a single BC is applied on each boundary,
leading to ``δ = 2``.
More generally, as described further below, it is possible to simultaneously
impose different BCs, which further decreases the number of degrees of freedom
(increasing ``δ``).

Thanks to the local support of B-splines, basis recombination involves only a
little portion of the original B-spline basis. For instance, since there is only
one B-spline that is non-zero at each boundary, removing that function from the
basis is enough to apply homogeneous Dirichlet BCs. Imposing BCs for derivatives
is slightly more complex, but still possible.

Note that, when combining basis recombination with [collocation methods](@ref
collocation-api), there must be **no** collocation points at the boundaries,
or the resulting collocation matrices may not be invertible.

## Order of the boundary condition

In this section, we consider the simplest case where a single homogeneous
boundary condition, ``\\mathrm{d}^n u / \\mathrm{d}x^n = 0``, is to be satisfied
by the basis.

The recombined basis requires the specification of a `Derivative` object
determining the order of the homogeneous BCs to be applied at the two
boundaries.
Linear combinations of `Derivative`s are also supported.
The order of the B-spline needs to be ``k ≥ n + 1``, since a B-spline
of order ``k`` is a ``C^{k - 1}``-continuous function (except on the knots
where it is ``C^{k - 1 - p}``, with ``p`` the knot multiplicity).

Some usual choices are:

- `Derivative(0)` sets homogeneous [Dirichlet
  BCs](https://en.wikipedia.org/wiki/Dirichlet_boundary_condition) (``u = 0`` at
  the boundaries) by removing the first and last B-splines, i.e. ``ϕ_1 = b_2``;

- `Derivative(1)` sets homogeneous [Neumann
  BCs](https://en.wikipedia.org/wiki/Neumann_boundary_condition) (``u' = 0`` at
  the boundaries) by adding the two first (and two last) B-splines, i.e. ``ϕ_1 =
  b_1 + b_2``.

- more generally, `α Derivative(0) + β Derivative(1)` sets homogeneous [Robin
  BCs](https://en.wikipedia.org/wiki/Robin_boundary_condition) by defining
  ``ϕ_1`` as a linear combination of ``b_1`` and ``b_2``.
  Here it's important to note that `Derivative(1)` denotes the [normal
  derivative](https://en.wikipedia.org/wiki/Directional_derivative#Normal_derivative)
  at the boundary, ``\\frac{∂u}{∂n}``, which is equal to ``-\\frac{∂u}{∂x}`` on
  the left boundary.

Higher order BCs are also possible.
For instance, `Derivative(2)` recombines the first three B-splines into two
basis functions that satisfy ``ϕ_1'' = ϕ_2'' = 0`` at the left boundary, while
ensuring that lower and higher-order derivatives keep degrees of freedom at the
boundary.
Note that simply adding the first three B-splines, as in ``ϕ_1 = b_1 + b_2 +
b_3``, makes the first derivative vanish as well as the second one, which is
unwanted.

For `Derivative(2)`, the chosen solution is to set ``ϕ_i = α_i b_i + β_i b_{i +
1}`` for ``i ∈ \\{1, 2\\}``.
The ``α_i`` and ``β_i`` coefficients are chosen such that ``ϕ_i'' = 0`` at the
boundary.
Moreover, they satisfy the (somewhat arbitrary) constraint ``α_i + β_i = 2`` for
each ``i``, for consistency with the Neumann case described above.
This generalises to higher order BCs.
Note that, since each boundary function ``ϕ_i`` is defined from only two
neighbouring B-splines, its local support stays minimal, hence preserving the
small bandwidth of the resulting matrices.

Finally, note that in the current implementation, it is not possible to impose
different boundary conditions on both boundaries.

## Multiple boundary conditions

As an option, the recombined basis may simultaneously satisfy homogeneous BCs of
different orders. In this case, a tuple of `Derivative`s must be passed.

For more details on the supported combinations of BCs, see the different
`RecombinedBSplineBasis` constructors documented further below.

---

    RecombinedBSplineBasis(op::AbstractDifferentialOp, B::BSplineBasis)

Construct `RecombinedBSplineBasis` from B-spline basis `B`, satisfying
homogeneous boundary conditions associated to the given differential operator.

For instance, `op = Derivative(0)` and `op = Derivative(1)` correspond to
homogeneous Dirichlet and Neumann BCs, respectively.

Linear combinations of differential operators are also supported.
For instance, `op = Derivative(0) + λ Derivative(1)` corresponds to homogeneous
Robin BCs.

Higher-order derivatives are also allowed, being only limited by the order of
the B-spline basis.

## Examples

```jldoctest RecombinedBSplineBasis
julia> B = BSplineBasis(BSplineOrder(4), -1:0.2:1)
13-element BSplineBasis of order 4, domain [-1.0, 1.0]
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]

julia> R_neumann = RecombinedBSplineBasis(Derivative(1), B)
11-element RecombinedBSplineBasis of order 4, domain [-1.0, 1.0]
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
 BCs left:  (D{1},)
 BCs right: (D{1},)

julia> R_robin = RecombinedBSplineBasis(Derivative(0) + 3Derivative(1), B)
11-element RecombinedBSplineBasis of order 4, domain [-1.0, 1.0]
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
 BCs left:  (D{0} + 3 * D{1},)
 BCs right: (D{0} + 3 * D{1},)
```

---

    RecombinedBSplineBasis(ops::Tuple{Vararg{AbstractDifferentialOp}},
                           B::BSplineBasis)

Construct `RecombinedBSplineBasis` simultaneously satisfying homogeneous BCs
associated to multiple differential operators.

Currently, the following specific combinations of differential operators are
supported:

1. all derivatives up to order `m`:

        ops = (Derivative(0), ..., Derivative(m))

   This boundary condition is obtained by removing the first `m + 1` B-splines
   from the original basis.

   For instance, if `(Derivative(0), Derivative(1))` is passed, then the basis
   simultaneously satisfies homogeneous Dirichlet and Neumann BCs at the two
   boundaries.
   The resulting basis is ``ϕ_1 = b_3, ϕ_2 = b_4, …, ϕ_{N - 4} = b_{N - 2}``.

2. an extension of the above, with an additional differential operator of order
   `n` at the end:

        ops = (Derivative(0), ..., Derivative(m), D(n))

   The operator `D(n)` may be a `Derivative`, or a linear combination of
   derivatives.
   The only restriction is that its maximum degree must satisfy `n ≥ m + 1`.

   One example is the combination of homogeneous Dirichlet BCs, ``u = 0`` on the
   boundaries, with Robin BCs for the derivative, ``u' + λ u'' = 0``, which
   corresponds to `ops = (Derivative(0), Derivative(1) + λ Derivative(2))`.

In all cases, the degrees of the differential operators must be in increasing
order. For instance, `ops = (Derivative(1), Derivative(0))` fails with an error.

## Examples

```jldoctest RecombinedBSplineBasis
julia> ops = (Derivative(0), Derivative(1));

julia> R1 = RecombinedBSplineBasis(ops, B)
9-element RecombinedBSplineBasis of order 4, domain [-1.0, 1.0]
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
 BCs left:  (D{0}, D{1})
 BCs right: (D{0}, D{1})

julia> ops = (Derivative(0), Derivative(1) - 4Derivative(2));

julia> R2 = RecombinedBSplineBasis(ops, B)
9-element RecombinedBSplineBasis of order 4, domain [-1.0, 1.0]
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
 BCs left:  (D{0}, D{1} + -4 * D{2})
 BCs right: (D{0}, D{1} + -4 * D{2})
```
"""
struct RecombinedBSplineBasis{
            k, T, Parent <: BSplineBasis{k,T},
            DiffOps <: Tuple{Vararg{AbstractDifferentialOp}},
            RMatrix <: RecombineMatrix{Q,DiffOps} where Q,
        } <: AbstractBSplineBasis{k,T}
    B :: Parent   # original B-spline basis
    ops :: DiffOps  # list of differential operators for BCs
    M :: RMatrix  # basis recombination matrix

    function RecombinedBSplineBasis(ops::Tuple{Vararg{AbstractDifferentialOp}},
                                    B::BSplineBasis{k,T}) where {k,T}
        Parent = typeof(B)
        M = RecombineMatrix(ops, B)
        RMatrix = typeof(M)
        Ops = typeof(ops)
        new{k,T,Parent,Ops,RMatrix}(B, ops, M)
    end

    RecombinedBSplineBasis(op::AbstractDifferentialOp, args...) =
        RecombinedBSplineBasis((op, ), args...)
end

Base.:(==)(A::RecombinedBSplineBasis, B::RecombinedBSplineBasis) =
    A === B ||
    constraints(A) == constraints(B) &&
    parent(A) == parent(B)

function Base.summary(io::IO, R::RecombinedBSplineBasis)
    summary_basis(io, R)
    cl, cr = constraints(R)
    print(io, ", BCs {left => ", cl, ", right => ", cr, "}")
    nothing
end

function Base.show(io::IO, R::RecombinedBSplineBasis)
    summary_basis(io, R)
    cl, cr = constraints(R)
    let io = IOContext(io, :compact => true, :limit => true)
        print(io, "\n knots: ", knots(R))
        print(io, "\n BCs left:  ", cl)
        print(io, "\n BCs right: ", cr)
    end
end

"""
    RecombinedBSplineBasis(ops, args...; kwargs...)

Construct [`RecombinedBSplineBasis`](@ref) from B-spline basis, satisfying
homogeneous boundary conditions associated one or more differential operators.

This variant does not require a previously constructed [`BSplineBasis`](@ref).
Arguments are passed to the `BSplineBasis` constructor.
"""
RecombinedBSplineBasis(order, args...; kwargs...) =
    RecombinedBSplineBasis(order, BSplineBasis(args...; kwargs...))

"""
    parent(R::RecombinedBSplineBasis)

Get original B-spline basis.
"""
Base.parent(R::RecombinedBSplineBasis) = R.B

"""
    recombination_matrix(R::AbstractBSplineBasis)

Get [`RecombineMatrix`](@ref) associated to the recombined basis.

For non-recombined bases such as [`BSplineBasis`](@ref), this returns the
identity matrix (`LinearAlgebra.I`).
"""
recombination_matrix(R::RecombinedBSplineBasis) = R.M
recombination_matrix(B::AbstractBSplineBasis) = LinearAlgebra.I

"""
    length(R::RecombinedBSplineBasis)

Returns the number of functions in the recombined basis.
"""
@inline Base.length(R::RecombinedBSplineBasis) =
    length(parent(R)) - sum(num_constraints(R))

boundaries(R::RecombinedBSplineBasis) = boundaries(parent(R))

knots(R::RecombinedBSplineBasis) = knots(parent(R))
order(::Type{<:RecombinedBSplineBasis{k}}) where {k} = k
Base.eltype(::Type{RecombinedBSplineBasis{k,T}}) where {k,T} = T

"""
    constraints(R::AbstractBSplineBasis) -> (left, right)
    constraints(A::RecombineMatrix) -> (left, right)

Return the constraints (homogeneous boundary conditions) that the basis
satisfies on each boundary.

Constraints are returned as a tuple `(left, right)` indicating the BCs that are
satisfied on each boundary. Each element is a tuple of differential operators
specifying the BCs.

For example, if both Dirichlet and Neumann BCs are satisfied on the left
boundary, then `left = (Derivative(0), Derivative(1))`.

For non-recombined bases such as [`BSplineBasis`](@ref), this returns a tuple of
empty tuples: `((), ())`, since no BCs are satisfied on either boundary.
"""
constraints(R::AbstractBSplineBasis) = constraints(recombination_matrix(R))

"""
    num_constraints(R::AbstractBSplineBasis) -> (Int, Int)
    num_constraints(A::RecombineMatrix) -> (Int, Int)

Returns the number of constraints (number of BCs to satisfy) on each boundary.

For instance, if `R` simultaneously satisfies Dirichlet and Neumann boundary
conditions on each boundary, this returns `(2, 2)`.

Note that for non-recombined bases such as [`BSplineBasis`](@ref), the number of
constraints is zero, and this returns `(0, 0)`.
"""
num_constraints(B::AbstractBSplineBasis) = num_constraints(recombination_matrix(B))

"""
    num_recombined(R::AbstractBSplineBasis) -> (Int, Int)
    num_recombined(A::RecombineMatrix) -> (Int, Int)

Returns the number of recombined functions in the recombined basis for each
boundary.

For instance, if `R` satisfies Neumann boundary conditions on both boundaries,
then only the first and last basis functions are different from the original
B-spline basis, e.g. ``ϕ_1 = b_1 + b_2``, and this returns `(1, 1)`.

For non-recombined bases such as [`BSplineBasis`](@ref), this returns zero.
"""
num_recombined(R::AbstractBSplineBasis) = num_recombined(recombination_matrix(R))

function nonzero_in_segment(R::RecombinedBSplineBasis, n)
    # Same as the original B-spline basis on the interval n - δl.
    δl = num_constraints(R)[1]
    nonzero_in_segment(parent(R), n - δl; N = length(R))
end

function support(R::RecombinedBSplineBasis, j)
    δl = num_constraints(R)[1]
    support(parent(R), j + δl)
end

@propagate_inbounds function evaluate(R::RecombinedBSplineBasis, j, args...)
    B = parent(R)
    A = recombination_matrix(R)
    is = nzrows(A, j)
    @assert !isempty(is)

    i1 = is[1]
    ϕ = evaluate(B, i1, args...)
    T = typeof(ϕ)

    if length(is) == 1  # avoid accessing A[i1, j] (we know it's 1)
        # @assert A[i1, j] == 1
        return ϕ
    end

    ϕ::T *= A[i1, j]

    for i ∈ is[2:end]
        ϕ::T += A[i, j] * evaluate(B, i, args...)
    end

    ϕ::T
end

@propagate_inbounds function BSplines.evaluate_all(
        R::RecombinedBSplineBasis, x, ::Type{T},
    ) where {T}
    B = parent(R)
    k = order(B)
    is, bs = BSplines.evaluate_all(B, x, T)
    A = recombination_matrix(R)

    cl, cr = num_constraints(A)
    nl, nr = num_recombined(A)
    M = size(A, 2)

    js = is .- cl  # possible range of output ϕ_j's

    # If we're outside of the recombination blocks, simply return (js, bs).
    if js[begin] > 0 && js[end] ≤ M
        lblock = which_recombine_block(A, js[begin])
        rblock = which_recombine_block(A, js[end])
        if lblock == rblock == 2
            return js, bs
        end
    end

    ϕs = @MVector zeros(T, k)
    δi = is[1] - 1
    for l = 1:k
        j = js[l]
        if j ≤ 0 || j > M
            continue
        end
        for i ∈ nzrows(A, j)
            # If i ∉ is, then b[i] wasn't evaluated because x is not in its
            # support.
            # In other words, b[i](x) = 0, and we can skip this B-spline.
            i ∈ is || continue
            ϕs[l] += bs[i - δi] * A[i, j]
        end
    end

    js, Tuple(ϕs)
end
