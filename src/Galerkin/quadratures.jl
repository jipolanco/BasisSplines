# Generate quadrature information for B-spline product.
# Returns weights and nodes for integration in [-1, 1].
#
# See https://en.wikipedia.org/wiki/Gaussian_quadrature.
#
# Some notes:
#
# - On each interval between two neighbouring knots, each B-spline is a
#   polynomial of degree (k - 1). Hence, the product of two B-splines has degree
#   (2k - 2).
#
# - The Gauss--Legendre quadrature rule is exact for polynomials of degree
#   <= (2n - 1), where n is the number of nodes and weights.
#
# - Conclusion: on each knot interval, `k` nodes should be enough to get an
#   exact integral. (I verified that results don't change when using more than
#   `k` nodes.)
#
# Here, p is the polynomial order (p = 2k - 2 for the product of two B-splines).
function _quadrature_prod(::Val{p}) where {p}
    n = cld(p + 1, 2)
    _gausslegendre(Val(n))
end

# Compile-time computation of Gauss--Legendre nodes and weights.
# If the @generated branch is taken, then the computation is done at compile
# time, making sure that no runtime allocations are performed.
function _gausslegendre(::Val{n}) where {n}
    if @generated
        vecs = _gausslegendre_impl(Val(n))
        :( $vecs )
    else
        _gausslegendre_impl(Val(n))
    end
end

function _gausslegendre_impl(::Val{n}) where {n}
    data = gausslegendre(n)
    map(SVector{n}, data)
end

# Metric for integration on interval [a, b].
# This is to transform from integration on the interval [-1, 1], in which
# quadratures are defined.
# See https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
struct QuadratureMetric{T}
    α :: T
    β :: T

    function QuadratureMetric(a::T, b::T) where {T}
        α = (b - a) / 2
        β = (a + b) / 2
        new{T}(α, β)
    end
end

# Apply metric to normalised coordinate (x ∈ [-1, 1]).
Base.:*(M::QuadratureMetric, x::Real) = M.α * x + M.β
Broadcast.broadcastable(M::QuadratureMetric) = Ref(M)

# Evaluate elements of basis `B` (given by indices `is`) at points `xs`.
# The length of `xs` is assumed static.
# The length of `is` is generally equal to the B-spline order, but may me
# smaller near the boundaries (this is not significant if knots are "augmented",
# as is the default).
# TODO implement evaluation of all B-splines at once (should be much faster...)
function eval_basis_functions(B, is, xs, deriv)
    k = order(B)
    @assert length(is) ≤ k
    ntuple(Val(k)) do n
        # In general, `is` will have length equal `k`.
        # If its length is less than `k` (may happen near boundaries, if knots
        # are not "augmented"), we repeat values for the first B-spline, just
        # for type stability concerns. These values are never used.
        i = n > length(is) ? is[1] : is[n]
        map(xs) do x
            B[i](x, deriv)
        end
    end
end

# TODO
# - Extend this to derivatives!
# - Try to avoid data transposition...
# - Avoid computing and passing `is` in the calling functions.
#   Instead, this function should return the `js`.
function eval_basis_functions(B, is, xs, ::Derivative{0})
    Nx = length(xs)

    bs_x = ntuple(Val(Nx)) do ℓ
        js, bs = BSplines.evaluate_all(B, xs[ℓ], Float64)
        if is === js  # this is always the case for non-recombined bases
            bs
        else
            # Make sure that bs[1] corresponds to is[1]
            @assert is ⊆ js
            δi = 1 - searchsortedlast(js, is[1])
            _circshift(bs, δi)
        end
    end

    # "Transpose" data
    k = order(B)
    @assert all(bs -> length(bs) == k, bs_x)
    ntuple(Val(k)) do n
        ntuple(Val(Nx)) do ℓ
            bs_x[ℓ][n]
        end
    end
end

# This is equivalent to Base.circshift for AbstractVector's.
function _circshift(vs::NTuple, shift)
    N = length(vs)
    ntuple(Val(N)) do i
        j = mod1(i - shift, N)
        @inbounds vs[j]
    end
end
