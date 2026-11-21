# The default tests check that the package loads and that its main functions can be called. The full
# test suite (test/full.jl, which also includes precision.jl and threads.jl) runs with
#     SWSH_FULL_TESTS=true julia --project -e 'using Pkg; Pkg.test()'
# or with Pkg.test(test_args=["full"]); run it with several threads (JULIA_NUM_THREADS) to exercise
# concurrent use.
using SpinWeightedSpheroidalHarmonics
using Test

@testset "SpinWeightedSpheroidalHarmonics.jl" begin
    @test spin_weighted_spherical_eigenvalue(-2, 2, 2) == 4

    Y = spin_weighted_spherical_harmonic(-2, 2, 2)
    @test Y(1.1, 0.3) ≈ sqrt(5 / π) / 8 * (1 + cos(1.1))^2 * cis(0.6) rtol=1e-12   # (1/2)√(5/π) cos⁴(θ/2) e^{2iφ}
    @test isfinite(Y(1.1, 0.3; theta_derivative=2, phi_derivative=1))

    @test spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 0.0) ≈ 4 atol=1e-12
    λ = spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 0.35)
    S = spin_weighted_spheroidal_harmonic(-2, 2, 2, 0.35)
    @test S.lambda ≈ λ rtol=1e-12
    @test isfinite(S(1.1, 0.3)) && isfinite(S(1.1, 0.3; theta_derivative=2))
    @test all(value_and_theta_derivative(S, 1.1, 0.3) .≈ (S(1.1, 0.3), S(1.1, 0.3; theta_derivative=1)))
    @test spin_weighted_spheroidal_harmonic(-2, 2, 2, 0.0)(1.1, 0.3) ≈ Y(1.1, 0.3) rtol=1e-12

    c = 0.5 - 0.5im
    Sc = spin_weighted_spheroidal_harmonic(-2, 2, 2, c)
    @test Sc.lambda ≈ spin_weighted_spheroidal_eigenvalue(-2, 2, 2, c) rtol=1e-12
    @test isfinite(Sc(1.1, 0.3))
    Sa = spin_weighted_spheroidal_harmonic(-2, 2, 2, c; normalization=:analytic)
    @test Sa.lambda == Sc.lambda
    @test sum(b^2 for b in Sa.coeffs) ≈ 1 rtol=1e-13

    # The interfaces added in 1.5.0
    @test angular_eigenvalue_derivative(-2, 2, 2, 0.35) ≈
        (spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 0.35 + 1e-6) - spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 0.35 - 1e-6)) / 2e-6 rtol=1e-8
    @test angular_eigenvalue_derivative(continue_angular_mode(-2, 2, 2, c)) isa ComplexF64
    three = value_and_theta_derivatives(S, 1.1, 0.3)
    @test all(three .≈ (S(1.1, 0.3), S(1.1, 0.3; theta_derivative=1), S(1.1, 0.3; theta_derivative=2)))
    @test spin_weighted_spheroidal_eigenvalue(S) === Float64(S.lambda)
    @test angular_parameters(Sc) === (s=-2, l=2, m=2, c=c)

    λbig = spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 7//20; precision=128)
    @test λbig isa BigFloat && precision(λbig) == 128
    @test Float64(λbig) ≈ spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 0.35) rtol=1e-13
end

if lowercase(get(ENV, "SWSH_FULL_TESTS", "false")) in ("true", "1") || "full" in ARGS
    include("full.jl")
end
