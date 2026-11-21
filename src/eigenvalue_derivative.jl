# The derivative of the eigenvalue with respect to c, from the eigenvector (Hellmann-Feynman).

#=
dM/dc of the band λ-matrix of _spectral_band, in the same upper band storage. Every entry of M is
a polynomial of second degree in c, c^2 a2 + c a1 + a0, so its derivative 2c a2 + a1 is exact.
=#
function _spectral_band_derivative(c, s::Int, m::Int, N::Int)
    T = _float_type(c)
    lmin = max(abs(m), abs(s))
    lmax = lmin + N - 1
    band = zeros(typeof(c * one(T)), 3, N)
    @inbounds for l in lmin:lmax
        i = l - lmin + 1
        band[3, i] = 2c * (1 - Bslm(T, s, l, m)) + 2 * (s * Hslm(T, s, l, m) - m)
        if l + 1 <= lmax
            band[2, i + 1] = -2c * Eslm(T, s, l + 1, m) + 2s * Gslm(T, s, l + 1, m)
        end
        if l + 2 <= lmax
            band[1, i + 2] = -2c * Cslm(T, s, l + 2, m)
        end
    end
    return band
end

#=
dλ/dc = vᵀ (dM/dc) v / (vᵀ v) for the eigenvector v of the complex symmetric λ-matrix M: its left
eigenvector is vᵀ, so the product is unconjugated, and the result does not depend on the
normalization or phase of v.
=#
function _hellmann_feynman_derivative(c, s::Int, m::Int, coefficients::AbstractVector)
    band = _spectral_band_derivative(c, s, m, length(coefficients))
    numerator = transpose(coefficients) * _symmetric_band_mul(band, coefficients)
    return numerator / (transpose(coefficients) * coefficients)
end

@doc raw"""
    angular_eigenvalue_derivative(pair::AngularEigenpair)
    angular_eigenvalue_derivative(harmonic::SpinWeightedSpheroidalHarmonicFunction)
    angular_eigenvalue_derivative(s::Int, l::Int, m::Int, c; precision=nothing)

The derivative ``d\lambda/dc`` of the spin-weighted spheroidal eigenvalue with respect to the
spheroidicity `c`, in the convention of `spin_weighted_spheroidal_eigenvalue`
(``\lambda = A + c^2 - 2mc``), on the same mode (for complex `c`, the same branch) as the eigenpair or
the harmonic.

It is computed from the eigenvector, without a second eigen-solve, by the Hellmann-Feynman theorem
for the complex symmetric spectral matrix ``M(c)``: ``d\lambda/dc = v^T M'(c) v / (v^T v)``, with the
unconjugated product and ``M'(c)`` exact (``M`` is quadratic in `c`). The result is in the
floating-point type of the pair or harmonic and accurate to about ``10^2`` eps of that type,
in `BigFloat` too.

The harmonic must come from the spectral decomposition (any method other than `"leaver"`). The
third form solves for the eigenpair at `c` first (with `continue_angular_mode` for complex `c`), in
the floating-point type of `c` or in `BigFloat` at `precision` bits; it returns a real number for
real `c` and a complex one for complex `c`.

# Example
```julia
pair = continue_angular_mode(-2, 2, 2, 0.5 - 0.3im)
angular_eigenvalue_derivative(pair)
angular_eigenvalue_derivative(-2, 2, 2, 3.0)
```
"""
function angular_eigenvalue_derivative(pair::AngularEigenpair{T}) where {T}
    return _with_precision(T, _input_precision(pair.c)) do
        _hellmann_feynman_derivative(pair.c, pair.s, pair.m, pair.coefficients)::Complex{T}
    end
end

function angular_eigenvalue_derivative(harmonic::SpinWeightedSpheroidalHarmonicFunction{T}) where {T}
    harmonic.method == :leaver && throw(ArgumentError(
        "angular_eigenvalue_derivative needs the spherical-harmonic coefficients of the spectral " *
        "decomposition, which a harmonic built with method=\"leaver\" does not have."))
    params = harmonic.params
    return _with_precision(T, harmonic.precision) do
        c = _to_float(T, params.c)
        derivative = _hellmann_feynman_derivative(c, params.s, params.m, harmonic.coeffs)
        c isa Real ? T(real(derivative)) : Complex{T}(derivative)
    end
end

function angular_eigenvalue_derivative(s::Int, l::Int, m::Int, c; precision::Union{Nothing, Integer}=nothing)
    _resolve_spheroidal_method("auto", s, l, m)
    _check_precision_keyword(precision)
    c = _with_precision_keyword(c, precision)
    T = _float_type(c)
    bits = precision === nothing ? _input_precision(c) : precision
    return _with_precision(T, bits) do
        if c isa Complex && !iszero(imag(c))
            return angular_eigenvalue_derivative(continue_angular_mode(s, l, m, Complex{T}(c)))
        end
        x = T(real(c))
        pair = _adaptive_real_eigenpair(x, s, l, m)
        T(real(_hellmann_feynman_derivative(x, s, m, pair.coefficients)))
    end
end
