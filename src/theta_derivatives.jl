# The value of a harmonic and its first and second theta-derivatives in one pass.

#=
S, dS/dθ and d²S/dθ² of a sum Σ_l w_l sYlm(θ, φ) over consecutive l of one (s, m), in one pass, as
_jacobi_value_and_theta_derivative does for the first two. Each core is written with the identity of
_jacobi_term already applied, f = r(n) S^a C^b P_N^(α',β')(cos θ), with S = sin(θ/2), C = cos(θ/2),
nonnegative a, b, α', β' and r(n) the constant ratio of the identity. With
dP_N^(α',β')/dθ = -K1 S C P_{N-1}^(α'+1,β'+1), K1 = N + α' + β' + 1, and K2 = K1 + 1 for the next one,
    f'  = (a/2) S^(a-1) C^(b+1) P - (b/2) S^(a+1) C^(b-1) P - K1 S^(a+1) C^(b+1) P1,
    f'' = a(a-1)/4 S^(a-2) C^(b+2) P - (2ab + a + b)/4 S^a C^b P + b(b-1)/4 S^(a+2) C^(b-2) P
          - K1 S^a C^b ((a + 1/2) C² - (b + 1/2) S²) P1 + K1 K2 S^(a+2) C^(b+2) P2,
every power with a negative exponent having a vanishing coefficient, so all three stay accurate up to
the poles. The three Jacobi families (P, P1, P2) come from three recurrences run along the sum.
`weight(l)` is w_l, or `nothing` for a harmonic left out of the sum; theta must lie in (0, π) (see
_near_pole). Nothing is allocated in Float64.
=#
function _jacobi_value_and_two_theta_derivatives(weight, s::Int, m::Int, lmin::Int, lmax::Int, theta, phi)
    sc, mc = m < 0 ? (-s, -m) : (s, m)
    α, β = mc + sc, mc - sc
    x = cos(theta)
    S = sin(theta / 2)
    C = cos(theta / 2)
    T = typeof(x)
    αp, βp, da, db, k = _jacobi_shift(α, β)
    a, b = α + da, β + db
    S2, C2 = S^2, C^2
    p00 = S^a * C^b
    pm = a >= 2 ? S^(a - 2) * C^(b + 2) : zero(T)
    pp = b >= 2 ? S^(a + 2) * C^(b - 2) : zero(T)
    q_a = a >= 1 ? S^(a - 1) * C^(b + 1) : zero(T)
    q_b = b >= 1 ? S^(a + 1) * C^(b - 1) : zero(T)
    q_1 = S^(a + 1) * C^(b + 1)
    q_2 = S^(a + 2) * C^(b + 2)
    # the l-independent combinations of the powers
    value_power = p00
    d1_power = (T(a) / 2) * q_a - (T(b) / 2) * q_b
    d2_power = (T(a * (a - 1)) / 4) * pm - (T(2a * b + a + b) / 4) * p00 + (T(b * (b - 1)) / 4) * pp
    d2_power1 = p00 * ((a + T(1) / 2) * C2 - (b + T(1) / 2) * S2)
    # P_j of each family, with its predecessor
    j0, p0, r0 = 0, one(T), zero(T)
    j1, p1, r1 = 0, one(T), zero(T)
    j2, p2, r2 = 0, one(T), zero(T)
    zero_value = zero(T) * zero(weight(lmin) === nothing ? zero(T) : weight(lmin))
    value = first_derivative = second_derivative = zero_value
    for l in lmin:lmax
        w = weight(l)
        w === nothing && continue
        n = l - mc
        N = n - k
        while j0 < N
            j0, p0, r0 = j0 + 1, _jacobi_next(j0, p0, r0, αp, βp, x), p0
        end
        K1 = T(N + αp + βp + 1)
        f = p0
        f1 = d1_power * p0
        f2 = d2_power * p0
        if N >= 1
            while j1 < N - 1
                j1, p1, r1 = j1 + 1, _jacobi_next(j1, p1, r1, αp + 1, βp + 1, x), p1
            end
            f1 -= K1 * q_1 * p1
            f2 -= K1 * d2_power1 * p1
            if N >= 2
                while j2 < N - 2
                    j2, p2, r2 = j2 + 1, _jacobi_next(j2, p2, r2, αp + 2, βp + 2, x), p2
                end
                f2 += K1 * (K1 + 1) * q_2 * p2
            end
        end
        factor = _swsh_prefactor(T, sc, l, mc) * _jacobi_shift_ratio(n, α, β, x)
        value += w * (factor * (value_power * f))
        first_derivative += w * (factor * f1)
        second_derivative += w * (factor * f2)
    end
    phase = (m < 0 ? (-1)^(s - m) : 1) * cis(m * phi)
    return value * phase, first_derivative * phase, second_derivative * phase
end

@doc raw"""
    value_and_theta_derivatives(harmonic, theta, phi=0)
    value_and_theta_derivatives(harmonic, thetas::AbstractVector, phi=0)

Evaluate a spin-weighted spheroidal or spherical harmonic and its first and second derivatives with
respect to `theta` together, and return them as a tuple `(S, dS/dtheta, d2S/dtheta2)`; for a vector of
angles, return the tuple of the three vectors. The results are those of `harmonic(theta, phi;
theta_derivative=k)` for `k = 0, 1, 2`, in the same floating-point type and precision. See
[`value_and_theta_derivatives!`](@ref) to fill preallocated vectors.

For a harmonic built with the Jacobi recurrence (the default, `method="auto"` or `"jacobi"`, of
`spin_weighted_spheroidal_harmonic` and `spin_weighted_spherical_harmonic`) the three are computed in
one pass: the Jacobi polynomials of all the spherical harmonics in the expansion come from one
recurrence for each of the three. The second derivative is taken exactly, not from the angular
differential equation, so it is accurate up to the poles, and at `theta = 0` and `π` themselves the
exact pole values are returned. For the other methods this is the same as the three separate calls.
A Float64 evaluation at one angle allocates nothing.

# Example
```julia
S = spin_weighted_spheroidal_harmonic(-2, 2, 2, 0.35)
value, first_derivative, second_derivative = value_and_theta_derivatives(S, 1.1, 0.3)
values, first_derivatives, second_derivatives = value_and_theta_derivatives(S, range(0, π; length=101))
```
"""
function value_and_theta_derivatives(harmonic::SpinWeightedSpheroidalHarmonicFunction{T}, theta, phi=0) where {T}
    # As _evaluate_at, written out so that a Float64 evaluation does not allocate
    R = _float_type(zero(T), theta, phi)
    return _with_precision(R, max(harmonic.precision, _input_precision(theta, phi))) do
        _value_and_theta_derivatives(harmonic, R(theta), R(phi))::NTuple{3, Complex{R}}
    end
end

function value_and_theta_derivatives(harmonic::SpinWeightedSphericalHarmonicFunction{T}, theta, phi=0) where {T}
    R = _float_type(zero(T), theta, phi)
    return _with_precision(R, max(harmonic.precision, _input_precision(theta, phi))) do
        _value_and_theta_derivatives(harmonic, R(theta), R(phi))::NTuple{3, Complex{R}}
    end
end

const _AnyHarmonic = Union{SpinWeightedSpheroidalHarmonicFunction, SpinWeightedSphericalHarmonicFunction}

function value_and_theta_derivatives(harmonic::_AnyHarmonic, thetas::AbstractVector, phi=0)
    T = _harmonic_float_type(harmonic)
    R = _float_type(zero(T), zero(eltype(thetas)), phi)
    return _with_precision(R, max(harmonic.precision, _input_precision(thetas, phi))) do
        values = Vector{Complex{R}}(undef, length(thetas))
        first_derivatives, second_derivatives = similar(values), similar(values)
        _fill_theta_derivatives!(values, first_derivatives, second_derivatives, harmonic, thetas, R(phi))
        values, first_derivatives, second_derivatives
    end
end

_harmonic_float_type(::SpinWeightedSpheroidalHarmonicFunction{T}) where {T} = T
_harmonic_float_type(::SpinWeightedSphericalHarmonicFunction{T}) where {T} = T

"""
    value_and_theta_derivatives!(S, dS, d2S, harmonic, thetas::AbstractVector, phi=0)

Fill the vectors `S`, `dS` and `d2S` (of the length of `thetas`) with the values of `harmonic` at
`(thetas[i], phi)` and their first and second `theta`-derivatives, as
[`value_and_theta_derivatives`](@ref) returns them, and return the three vectors. The vectors must have
a complex element type that holds the results (`ComplexF64` for a Float64 harmonic and angles); a
Float64 evaluation into `Vector{ComplexF64}`s allocates nothing.
"""
function value_and_theta_derivatives!(values::AbstractVector, first_derivatives::AbstractVector,
        second_derivatives::AbstractVector, harmonic::_AnyHarmonic, thetas::AbstractVector, phi=0)
    n = length(thetas)
    length(values) == length(first_derivatives) == length(second_derivatives) == n || throw(DimensionMismatch(
        "the output vectors have lengths $(length(values)), $(length(first_derivatives)) and " *
        "$(length(second_derivatives)), but there are $n angles."))
    T = _harmonic_float_type(harmonic)
    R = real(eltype(values))
    _with_precision(R, max(harmonic.precision, _input_precision(thetas, phi))) do
        _fill_theta_derivatives!(values, first_derivatives, second_derivatives, harmonic, thetas, R(phi))
    end
    return values, first_derivatives, second_derivatives
end

function _three_calls(harmonic, theta, phi)
    C = Complex{typeof(theta)}
    return (convert(C, harmonic(theta, phi))::C, convert(C, harmonic(theta, phi; theta_derivative=1))::C,
        convert(C, harmonic(theta, phi; theta_derivative=2))::C)
end

function _value_and_theta_derivatives(harmonic::SpinWeightedSpheroidalHarmonicFunction{T}, theta, phi) where {T}
    harmonics = _one_pass_harmonics(harmonic)
    harmonics === nothing && return _three_calls(harmonic, theta, phi)
    params = harmonic.params
    lmin = max(abs(params.s), abs(params.m))
    coefficients, normalization = harmonic.coeffs, harmonic.normalization_const
    # The types a spectral harmonic stores, spelled out so that the call below is resolved statically
    if coefficients isa Vector{Complex{T}} && normalization isa Int
        return _two_derivatives_at(harmonic, params.s, params.m, lmin, lmin + params.N - 1,
            coefficients, harmonics, normalization, theta, phi)
    elseif coefficients isa Vector{Complex{T}} && normalization isa T
        return _two_derivatives_at(harmonic, params.s, params.m, lmin, lmin + params.N - 1,
            coefficients, harmonics, normalization, theta, phi)
    end
    return _two_derivatives_at(harmonic, params.s, params.m, lmin, lmin + params.N - 1,
        coefficients, harmonics, normalization, theta, phi)::NTuple{3, Complex{typeof(theta)}}
end

_value_and_theta_derivatives(harmonic::SpinWeightedSphericalHarmonicFunction, theta, phi) =
    harmonic.method == :jacobi ?
        _two_derivatives_at(harmonic, harmonic.s, harmonic.m, harmonic.l, harmonic.l, nothing, nothing, 1, theta, phi) :
        _three_calls(harmonic, theta, phi)

function _fill_theta_derivatives!(values, first_derivatives, second_derivatives,
        harmonic::SpinWeightedSpheroidalHarmonicFunction{T}, thetas, phi) where {T}
    harmonics = _one_pass_harmonics(harmonic)
    if harmonics === nothing
        _fill_three_calls!(values, first_derivatives, second_derivatives, harmonic, thetas, phi)
        return nothing
    end
    params = harmonic.params
    lmin = max(abs(params.s), abs(params.m))
    lmax = lmin + params.N - 1
    coefficients, normalization = harmonic.coeffs, harmonic.normalization_const
    # The types a spectral harmonic stores, spelled out so that the loop is compiled for them and the
    # call is resolved statically (a dynamic call would allocate)
    if coefficients isa Vector{Complex{T}} && normalization isa Int
        _fill_two_derivatives!(values, first_derivatives, second_derivatives, harmonic, params.s, params.m,
            lmin, lmax, coefficients, harmonics, normalization, thetas, phi)
    elseif coefficients isa Vector{Complex{T}} && normalization isa T
        _fill_two_derivatives!(values, first_derivatives, second_derivatives, harmonic, params.s, params.m,
            lmin, lmax, coefficients, harmonics, normalization, thetas, phi)
    else
        _fill_two_derivatives!(values, first_derivatives, second_derivatives, harmonic, params.s, params.m,
            lmin, lmax, coefficients, harmonics, normalization, thetas, phi)
    end
    return nothing
end

function _fill_theta_derivatives!(values, first_derivatives, second_derivatives,
        harmonic::SpinWeightedSphericalHarmonicFunction, thetas, phi)
    if harmonic.method == :jacobi
        _fill_two_derivatives!(values, first_derivatives, second_derivatives, harmonic, harmonic.s, harmonic.m,
            harmonic.l, harmonic.l, nothing, nothing, 1, thetas, phi)
    else
        _fill_three_calls!(values, first_derivatives, second_derivatives, harmonic, thetas, phi)
    end
    return nothing
end

function _fill_three_calls!(values, first_derivatives, second_derivatives, harmonic, thetas, phi)
    R = real(eltype(values))
    for (i, theta) in enumerate(thetas)
        values[i], first_derivatives[i], second_derivatives[i] = _three_calls(harmonic, R(theta), phi)
    end
end

function _fill_two_derivatives!(values, first_derivatives, second_derivatives, harmonic, s, m, lmin, lmax,
        coefficients, harmonics, normalization, thetas, phi)
    R = real(eltype(values))
    for (i, theta) in enumerate(thetas)
        values[i], first_derivatives[i], second_derivatives[i] = _two_derivatives_at(harmonic, s, m, lmin, lmax,
            coefficients, harmonics, normalization, R(theta), phi)
    end
end

function _two_derivatives_at(harmonic, s, m, lmin, lmax, coefficients, harmonics, normalization, theta, phi)
    _theta, _phi = _fold_theta_phi(theta, phi)
    # At the poles the three evaluations take their exact pole values
    _near_pole(_theta) && return _three_calls(harmonic, theta, phi)
    weight = coefficients === nothing ? (l -> 1) :
        (l -> (i = l - lmin + 1; harmonics[i] === nothing ? nothing : coefficients[i]))
    value, first_derivative, second_derivative =
        _jacobi_value_and_two_theta_derivatives(weight, s, m, lmin, lmax, _theta, _phi)
    C = Complex{typeof(theta)}
    return convert(C, value / normalization), convert(C, first_derivative / normalization),
        convert(C, second_derivative / normalization)
end
