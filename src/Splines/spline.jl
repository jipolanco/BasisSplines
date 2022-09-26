using StaticArrays: MVector

"""
    Spline{T}

Represents a spline function.

---

    Spline(B::AbstractBSplineBasis, coefs::AbstractVector)

Construct a spline from a B-spline basis and a vector of B-spline coefficients.

# Examples

```jldoctest; filter = r"coefficients: \\[.*\\]"
julia> B = BSplineBasis(BSplineOrder(4), -1:0.2:1);

julia> coefs = rand(length(B));

julia> S = Spline(B, coefs)
13-element Spline{Float64}:
 basis: 13-element BSplineBasis of order 4, domain [-1.0, 1.0]
 order: 4
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
 coefficients: [0.815921, 0.076499, 0.433472, 0.672844, 0.468371, 0.348423, 0.868621, 0.0831675, 0.369734, 0.401199, 0.990734, 0.565907, 0.984855]
```

---

    Spline{T = Float64}(undef, B::AbstractBSplineBasis)

Construct a spline with uninitialised vector of coefficients.

---

    (S::Spline)(x)

Evaluate spline at coordinate `x`.
"""
struct Spline{
        T,  # type of coefficient (e.g. Float64, ComplexF64)
        Basis <: AbstractBSplineBasis,
        CoefVector <: AbstractVector{T},
    }
    basis :: Basis
    coefs :: CoefVector

    function Spline(B::AbstractBSplineBasis, cs::AbstractVector)
        coefs = wrap_coefficients(B, cs)  # used for periodic bases
        length(coefs) == length(B) ||
            throw(ArgumentError("wrong number of coefficients"))
        Basis = typeof(B)
        T = eltype(coefs)
        CoefVector = typeof(coefs)
        k = order(B)
        @assert k >= 1
        new{T, Basis, CoefVector}(B, coefs)
    end
end

# By default coefficients are not wrapped.
wrap_coefficients(::AbstractBSplineBasis, cs::AbstractVector) = cs

# This is mainly useful for periodic bases.
unwrap_coefficients(S::Spline) = unwrap_coefficients(basis(S), coefficients(S))
unwrap_coefficients(::AbstractBSplineBasis, cs::AbstractVector) = cs

Broadcast.broadcastable(S::Spline) = Ref(S)

Base.copy(S::Spline) = Spline(basis(S), copy(coefficients(S)))

function Base.show(io::IO, S::Spline)
    println(io, length(S), "-element ", nameof(typeof(S)), '{', eltype(S), '}', ':')
    print(io, " basis: ")
    summary(io, basis(S))
    println(io, "\n order: ", order(S))
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, " knots: ", knots(S))
        print(io, " coefficients: ", coefficients(S))
    end
    nothing
end

Base.:(==)(P::Spline, Q::Spline) =
    basis(P) == basis(Q) && coefficients(P) == coefficients(Q)

Base.isapprox(P::Spline, Q::Spline; kwargs...) =
    basis(P) == basis(Q) &&
    isapprox(coefficients(P), coefficients(Q); kwargs...)

function Spline{T}(init, B::AbstractBSplineBasis) where {T}
    coefs = Vector{T}(init, length(B))
    Spline(B, coefs)
end

Spline(init, B::AbstractBSplineBasis) = Spline{Float64}(init, B)

# TODO deprecate?
Spline(init, B::AbstractBSplineBasis, ::Type{T}) where {T} =
    Spline{T}(init, B)

parent_spline(S::Spline) = parent_spline(basis(S), S)
parent_spline(::AbstractBSplineBasis, S::Spline) = S

"""
    coefficients(S::Spline)

Get B-spline coefficients of the spline.
"""
coefficients(S::Spline) = S.coefs

"""
    length(S::Spline)

Returns the number of coefficients in the spline.

Note that this is equal to the number of basis functions, `length(basis(S))`.
"""
Base.length(S::Spline) = length(coefficients(S))

"""
    eltype(::Type{<:Spline})
    eltype(S::Spline)

Returns type of element returned when evaluating the [`Spline`](@ref).
"""
Base.eltype(::Type{<:Spline{T}}) where {T} = T

"""
    basis(S::Spline) -> AbstractBSplineBasis

Returns the associated B-spline basis.
"""
basis(S::Spline) = S.basis

knots(S::Spline) = knots(basis(S))
order(::Type{<:Spline{T,Basis}}) where {T,Basis} = order(Basis)
order(S::Spline) = order(typeof(S))

# TODO allow evaluating derivatives at point `x` (should be much cheaper than
# constructing a new Spline for the derivative)
@inline function (S::Spline)(x)
    B = basis(S)
    if has_parent_basis(B)
        parent_spline(S)(x)
    else
        _evaluate(S, x)
    end
end

function _evaluate(S::Spline, x)
    T = eltype(S)
    t = knots(S)
    n, zone = find_knot_interval(t, x)
    iszero(zone) || return zero(T)  # x is outside of knot domain
    k = order(S)
    spline_kernel(coefficients(S), t, n, x, BSplineOrder(k))
end

function spline_kernel(
        c::AbstractVector{T}, t, n, x, ::BSplineOrder{k},
    ) where {T,k}
    # Algorithm adapted from https://en.wikipedia.org/wiki/De_Boor's_algorithm
    if @generated
        ex = quote
            @nexprs $k j -> d_j = @inbounds c[j + n - $k]
        end
        for r = 2:k, j = k:-1:r
            d_j = Symbol(:d_, j)
            d_p = Symbol(:d_, j - 1)
            jk = j - k
            jr = j - r
            ex = quote
                $ex
                @inbounds ti = t[$jk + n]
                @inbounds tj = t[$jr + n + 1]
                α = (x - ti) / (tj - ti)
                $d_j = $T((1 - α) * $d_p + α * $d_j)
            end
        end
        d_k = Symbol(:d_, k)
        quote
            $ex
            return $d_k
        end
    else
        # Similar using MVector (a bit slower than @generated version).
        spline_kernel_alt(c, t, n, x, BSplineOrder(k))
    end
end

function spline_kernel_alt(
        c::AbstractVector{T}, t, n, x, ::BSplineOrder{k},
    ) where {T, k}
    d = MVector(ntuple(j -> @inbounds(c[j + n - k]), Val(k)))
    @inbounds for r = 2:k
        dprev = d[r - 1]
        for j = r:k
            jn = j + n
            ti = t[jn - k]
            tj = t[jn - r + 1]
            α = (x - ti) / (tj - ti)
            dtmp = dprev
            dprev = d[j]
            d[j] = (1 - α) * dtmp + α * dprev
        end
    end
    @inbounds d[k]
end

"""
    *(op::Derivative, S::Spline) -> Spline

Returns `N`-th derivative of spline `S` as a new spline.

See also [`diff`](@ref).

# Examples

```jldoctest; filter = r"coefficients: \\[.*\\]"
julia> B = BSplineBasis(BSplineOrder(4), -1:0.2:1);

julia> S = Spline(B, rand(length(B)))
13-element Spline{Float64}:
 basis: 13-element BSplineBasis of order 4, domain [-1.0, 1.0]
 order: 4
 knots: [-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
 coefficients: [0.461501, 0.619799, 0.654451, 0.667213, 0.334672, 0.618022, 0.967496, 0.900014, 0.611195, 0.469467, 0.221618, 0.80084, 0.269533]

julia> Derivative(0) * S === S
true

julia> Derivative(1) * S
12-element Spline{Float64}:
 basis: 12-element BSplineBasis of order 3, domain [-1.0, 1.0]
 order: 3
 knots: [-1.0, -1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
 coefficients: [2.37448, 0.259885, 0.0638088, -1.6627, 1.41675, 1.74737, -0.33741, -1.44409, -0.708643, -1.23925, 4.34416, -7.9696]

julia> Derivative(2) * S
11-element Spline{Float64}:
 basis: 11-element BSplineBasis of order 2, domain [-1.0, 1.0]
 order: 2
 knots: [-1.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0]
 coefficients: [-21.146, -0.98038, -8.63255, 15.3972, 1.65313, -10.4239, -5.53341, 3.67724, -2.65301, 27.917, -123.138]
```
"""
Base.:*(op::Derivative, S::Spline) = _diff(basis(S), S, op)

"""
    diff(S::Spline, [op::Derivative = Derivative(1)]) -> Spline

Same as `op * S`.

Returns `N`-th derivative of spline `S` as a new spline.
"""
Base.diff(S::Spline, op = Derivative(1)) = op * S

_diff(::AbstractBSplineBasis, S, etc...) = diff(parent_spline(S), etc...)

_diff(::BSplineBasis, S::Spline, ::Derivative{0}) = S

function _diff(
        ::BSplineBasis, S::Spline, ::Derivative{Ndiff} = Derivative(1),
    ) where {Ndiff}
    Ndiff :: Integer
    @assert Ndiff >= 1

    u = coefficients(S)
    t = knots(S)
    k = order(S)

    if Ndiff >= k
        throw(ArgumentError(
            "cannot differentiate order $k spline $Ndiff times!"))
    end

    Base.require_one_based_indexing(u)
    du = similar(u)
    copy!(du, u)

    @inbounds for m = 1:Ndiff, i in Iterators.Reverse(eachindex(du))
        dt = t[i + k - m] - t[i]
        if iszero(dt) || i == 1
            # In this case, the B-spline that this coefficient is
            # multiplying is zero everywhere, so we can set this to zero.
            # From de Boor (2001, p. 117): "anything times zero is zero".
            du[i] = zero(eltype(du))
        else
            du[i] = (k - m) * (du[i] - du[i - 1]) / dt
        end
    end

    # Finally, create lower-order spline with the given coefficients.
    # Note that the spline has `2 * Ndiff` fewer knots, and `Ndiff` fewer
    # B-splines.
    N = length(u)
    Nt = length(t)
    t_new = view(t, (1 + Ndiff):(Nt - Ndiff))
    B = BSplineBasis(BSplineOrder(k - Ndiff), t_new; augment = Val(false))

    Spline(B, view(du, (1 + Ndiff):N))
end

# Zeroth derivative: return S itself.
Base.diff(S::Spline, ::Derivative{0}) = S

"""
    integral(S::Spline)

Returns an antiderivative of the given spline as a new spline.

The algorithm is described in de Boor 2001, p. 127.
"""
integral(S::Spline) = _integral(basis(S), S)

_integral(::AbstractBSplineBasis, S, etc...) = integral(parent_spline(S), etc...)

function _integral(::BSplineBasis, S::Spline)
    u = coefficients(S)
    t = knots(S)
    k = order(S)
    Base.require_one_based_indexing(u)

    Nt = length(t)
    N = length(u)

    # Note that the new spline has 2 more knots and 1 more B-spline.
    t_int = similar(t, Nt + 2)
    t_int[2:(end - 1)] .= t
    t_int[1] = t_int[2]
    t_int[end] = t_int[end - 1]

    β = similar(u, N + 1)
    β[1] = zero(eltype(β))

    @inbounds for i in eachindex(u)
        m = i + 1
        β[m] = zero(eltype(β))
        for j = 1:i
            β[m] += u[j] * (t[j + k] - t[j]) / k
        end
    end

    B = BSplineBasis(BSplineOrder(k + 1), t_int; augment = Val(false))
    Spline(B, β)
end
