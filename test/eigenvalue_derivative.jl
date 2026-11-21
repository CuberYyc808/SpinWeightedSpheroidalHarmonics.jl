# dλ/dc from the eigenvector (angular_eigenvalue_derivative) against a finite difference of the
# eigenvalue at 512 bits (five-point stencil, step 1e-20: truncation 1e-80, roundoff 1e-134).
using SpinWeightedSpheroidalHarmonics, Test

const EIGENVALUE_DERIVATIVE_CS = (3//10, 3//1, 1//2 - 3im//10, 6//5 - 2im)

function _reference_eigenvalue_derivative(s, l, m, c)
    return setprecision(BigFloat, 512) do
        x = c isa Complex ? Complex{BigFloat}(big(real(c)), big(imag(c))) : big(c)
        h = big(10)^-20
        λ(t) = spin_weighted_spheroidal_eigenvalue(s, l, m, x + t * h)
        (8 * (λ(1) - λ(-1)) - (λ(2) - λ(-2))) / (12h)
    end
end

# The largest relative error (to max(1, |dλ/dc|)) in Float64 and at 256 bits over the modes given
function eigenvalue_derivative_errors(cases)
    worst64, worst256 = 0.0, 0.0
    for (s, l, m, c) in cases
        reference = _reference_eigenvalue_derivative(s, l, m, c)
        scale = max(1, abs(reference))
        d64 = angular_eigenvalue_derivative(s, l, m, c isa Complex ? ComplexF64(c) : Float64(c))
        d256 = angular_eigenvalue_derivative(s, l, m, c; precision=256)
        worst64 = max(worst64, Float64(abs(d64 - reference) / scale))
        worst256 = max(worst256, Float64(abs(d256 - reference) / scale))
    end
    return worst64, worst256
end

function eigenvalue_derivative_cases(; ss=(-2, 0, 2), ls=nothing, cs=EIGENVALUE_DERIVATIVE_CS)
    cases = []
    for s in ss, c in cs, l in something(ls, abs(s):12), m in unique((-l, -2, 0, 1, 2, l))
        l >= max(abs(s), abs(m)) && push!(cases, (s, l, m, c))
    end
    return cases
end

@testset "Derivative of the eigenvalue with respect to c (sweep)" begin
    for s in (-2, 0, 2), c in EIGENVALUE_DERIVATIVE_CS
        worst64, worst256 = eigenvalue_derivative_errors(eigenvalue_derivative_cases(; ss=(s,), cs=(c,)))
        @test worst64 <= 1e-13
        @test worst256 <= 1e-60
    end
end

@testset "Derivative of the eigenvalue with respect to c" begin
    # The forms agree: the pair, the harmonic built from it (either normalization) and the parameters
    c = 1.2 - 2.0im
    pair = continue_angular_mode(-2, 3, 1, c)
    d = angular_eigenvalue_derivative(pair)
    @test d isa ComplexF64
    @test angular_eigenvalue_derivative(-2, 3, 1, c) == d
    for normalization in (:modulus, :analytic)
        @test angular_eigenvalue_derivative(spin_weighted_spheroidal_harmonic(pair; normalization)) ≈ d rtol=1e-14
    end
    @test angular_eigenvalue_derivative(spin_weighted_spheroidal_harmonic(-2, 3, 1, 3.0)) ≈
        angular_eigenvalue_derivative(-2, 3, 1, 3.0) rtol=1e-14
    @test angular_eigenvalue_derivative(-2, 3, 1, 3.0) isa Float64
    # c = 0: dλ/dc = 2 s H - 2m, H = -m s / (l (l + 1))
    @test angular_eigenvalue_derivative(-2, 3, 1, 0.0) ≈ 2 * (-2) * (2 / 12) - 2 atol=1e-15
    big_derivative = angular_eigenvalue_derivative(-2, 3, 1, 3//1; precision=256)
    @test big_derivative isa BigFloat && precision(big_derivative) == 256
    @test_throws ArgumentError angular_eigenvalue_derivative(spin_weighted_spheroidal_harmonic(-2, 3, 1, 0.7; method="leaver"))
end
