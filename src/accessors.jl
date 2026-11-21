# Typed accessors of a harmonic and of an eigenpair: the eigenvalue and the mode parameters.

# λ in the floating-point type T of a harmonic: T for real c, Complex{T} for complex c
_eigenvalue_in(::Type{T}, lambda::Complex) where {T} = Complex{T}(lambda)
_eigenvalue_in(::Type{T}, lambda::Real) where {T} = T(lambda)

"""
    spin_weighted_spheroidal_eigenvalue(harmonic::SpinWeightedSpheroidalHarmonicFunction{T})
    spin_weighted_spheroidal_eigenvalue(pair::AngularEigenpair{T})

The eigenvalue ``λ`` (in the convention of `spin_weighted_spheroidal_eigenvalue(s, l, m, c)`) of a
harmonic or a continued eigenpair, without solving for it again. It is in the floating-point type `T`
the harmonic was built in: a `T` for real `c` (`c = 0` included) and a `Complex{T}` for complex `c`;
for an eigenpair it is always a `Complex{T}`. Use it in place of the field `lambda`, which is not
typed.
"""
function spin_weighted_spheroidal_eigenvalue(harmonic::SpinWeightedSpheroidalHarmonicFunction{T}) where {T}
    return _with_precision(T, harmonic.precision) do
        _eigenvalue_in(T, harmonic.lambda)
    end
end

spin_weighted_spheroidal_eigenvalue(pair::AngularEigenpair{T}) where {T} = pair.lambda

"""
    spin_weighted_spherical_eigenvalue(harmonic::SpinWeightedSphericalHarmonicFunction{T})
    spin_weighted_spherical_eigenvalue(::Type{T}, s::Int, l::Int, m::Int=0)

The eigenvalue ``λ = l(l+1) - s(s+1)`` of a spin-weighted spherical harmonic, in the floating-point type
`T` the harmonic was built in, or in the type `T` given (`spin_weighted_spherical_eigenvalue(s, l, m)`
returns it as an `Int`).
"""
spin_weighted_spherical_eigenvalue(harmonic::SpinWeightedSphericalHarmonicFunction{T}) where {T} =
    _with_precision(() -> T(harmonic.lambda), T, harmonic.precision)

spin_weighted_spherical_eigenvalue(::Type{T}, s::Int, l::Int, m::Int=0) where {T<:Number} =
    T(spin_weighted_spherical_eigenvalue(s, l, m))

"""
    angular_parameters(harmonic::SpinWeightedSpheroidalHarmonicFunction{T})
    angular_parameters(harmonic::SpinWeightedSphericalHarmonicFunction{T})
    angular_parameters(pair::AngularEigenpair{T})

The mode parameters as a named tuple `(s, l, m, c)`: the spin weight `s`, harmonic index `l` and
azimuthal index `m` as `Int`s, and the spheroidicity `c` in the floating-point type `T` of the harmonic
(a `T` for real `c`, a `Complex{T}` for complex `c`; `zero(T)` for a spherical harmonic; always a
`Complex{T}` for an eigenpair). Use it in place of the fields `params.s`, `params.l`, `params.m` and
`params.c`, which are not all typed.
"""
function angular_parameters(harmonic::SpinWeightedSpheroidalHarmonicFunction{T}) where {T}
    params = harmonic.params
    c = _with_precision(T, harmonic.precision) do
        _eigenvalue_in(T, params.c)
    end
    return (s=params.s, l=Int(params.l), m=params.m, c=c)
end

angular_parameters(harmonic::SpinWeightedSphericalHarmonicFunction{T}) where {T} =
    (s=harmonic.s, l=harmonic.l, m=harmonic.m, c=zero(T))

angular_parameters(pair::AngularEigenpair) = (s=pair.s, l=pair.l, m=pair.m, c=pair.c)
