using Base: @propagate_inbounds

"""
    AugmentedKnots{T,k} <: AbstractVector{T}

Pads from both sides a vector of B-spline breakpoints, making sure that the
first and last values are repeated `k` times.
"""
struct AugmentedKnots{T, k, Breakpoints <: AbstractVector{T}} <: AbstractVector{T}
    Nt :: Int  # total number of knots
    x  :: Breakpoints
    function AugmentedKnots{k}(x::AbstractVector) where {k}
        Base.require_one_based_indexing(x)
        T = eltype(x)
        Nt = 2 * (k - 1) + length(x)
        new{T, k, typeof(x)}(Nt, x)
    end
end

# This is the number of knots added on each side.
padding(t::AugmentedKnots{T,k}) where {T,k} = k - 1
breakpoints(t::AugmentedKnots) = t.x

Base.size(t::AugmentedKnots) = (t.Nt, )
Base.IndexStyle(::Type{<:AugmentedKnots}) = IndexLinear()

# Custom definition of searchsortedlast, for sligthly faster determination of
# the knot interval corresponding to a given point `x` (used when evaluating
# splines; see `knot_interval`).
# TODO does this actually improve performance??
function Base.searchsortedlast(t::AugmentedKnots, x; kws...)
    ξ = breakpoints(t)
    if x < first(ξ)
        0
    elseif x ≥ last(ξ)
        length(t)
    else
        searchsortedlast(ξ, x; kws...) + padding(t)
    end
end

@propagate_inbounds function Base.getindex(t::AugmentedKnots, i::Int)
    @boundscheck checkbounds(t, i)
    x = breakpoints(t)
    j = clamp(i - padding(t), 1, length(x))
    @inbounds x[j]
end

"""
    augment_knots!(breaks::AbstractVector, k::Union{Integer,BSplineOrder})

Modifies the input breakpoints to make sure that the first and last knot have
multiplicity `k` for splines of order `k`.

To prevent allocations, this function will modify the input when this is a
standard `Vector`. Otherwise, the input will be wrapped inside an
[`AugmentedKnots`](@ref) object.

It is assumed that the input breakpoints have multiplicity 1 at the borders.
That is, border coordinates should *not* be repeated in the input.
"""
augment_knots!(breaks::AbstractVector, ::BSplineOrder{k}) where {k} =
    AugmentedKnots{k}(breaks)

@inline augment_knots!(breaks, k::Integer) = augment_knots!(breaks, BSplineOrder(k))

function augment_knots!(t::Vector, ::BSplineOrder{k}) where {k}
    Base.require_one_based_indexing(t)
    Nt = length(t) + 2 * (k - 1)  # number of knots
    ta, tb = first(t), last(t)
    resize!(t, Nt)
    δ = k - 1
    for i = Nt:-1:(Nt - δ)
        @inbounds t[i] = tb
    end
    for i = (Nt - k):-1:(k + 1)
        @inbounds t[i] = t[i - δ]  # shift values of original vector
    end
    for i = k:-1:1
        @inbounds t[i] = ta
    end
    t
end

"""
    multiplicity(knots, i)

Determine multiplicity of knot `knots[i]`.
"""
function multiplicity(knots::AbstractVector, i)
    Base.require_one_based_indexing(knots)
    v = knots[i]
    m = 1

    # Check in both directions
    j = i - 1
    while j > 0 && knots[j] == v
        j -= 1
        m += 1
    end

    j = i + 1
    N = length(knots)
    while j <= N && knots[j] == v
        j += 1
        m += 1
    end

    m
end

function knot_interval(t::AbstractVector, x)
    n = searchsortedlast(t, x)  # t[n] <= x < t[n + 1]
    n == 0 && return nothing    # x < t[1]

    Nt = length(t)

    if n == Nt  # i.e. if x >= t[end]
        t_last = t[n]
        x > t_last && return nothing
        # If x is exactly on the last knot, decrease the index as necessary.
        while t[n] == t_last
            n -= one(n)
        end
    end

    n
end
