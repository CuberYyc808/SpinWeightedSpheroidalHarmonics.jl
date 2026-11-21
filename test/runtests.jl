# The default tests check that the package loads and that its main functions can be called. The full
# test suite (test/full.jl, which also includes precision.jl and threads.jl) runs with
#     SWSH_FULL_TESTS=true julia --project -e 'using Pkg; Pkg.test()'
# or with Pkg.test(test_args=["full"]); run it with several threads (JULIA_NUM_THREADS) to exercise
# concurrent use.
using SpinWeightedSpheroidalHarmonics
using LinearAlgebra
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
    basis = spin_weighted_spheroidal_basis(-2, 2, 0.35; lmax=6)
    @test spin_weighted_spheroidal_eigenvalue(basis, 2) ≈ λ rtol=1e-14
    @test spin_weighted_spheroidal_harmonic(basis, 3)(1.1, 0.3) ≈ spin_weighted_spheroidal_harmonic(-2, 3, 2, 0.35)(1.1, 0.3) rtol=1e-13

    # The angular spectral matrix M(c) = M0 + c M1 + c^2 M2: its eigenvalue nearest the spherical one is the
    # separation constant A = λ - c^2 + 2mc, and the polynomial is the matrix built internally
    for s in -2:2, m in -3:3, N in (20, 40), c in (0.7, 0.3 - 0.2im)
        M0, M1, M2 = angular_spectral_matrix_coefficients(s, m, N)
        M = M0 + c * M1 + c^2 * M2
        @test M ≈ SpinWeightedSpheroidalHarmonics.construct_spectral_matrix(c, s, m, N) rtol=0 atol=4eps(maximum(abs, M))
        values = eigvals(M)
        for l in (max(abs(s), abs(m)), max(abs(s), abs(m)) + 2)
            A = spin_weighted_spheroidal_eigenvalue(s, l, m, c) - c^2 + 2m * c
            nearest = values[argmin(abs.(values .- (l * (l + 1) - s * (s + 1))))]
            @test abs(nearest - A) <= 1e-12 * abs(A)
        end
    end
    @test eltype(first(setprecision(() -> angular_spectral_matrix_coefficients(BigFloat, -2, 2, 6), BigFloat, 256))) == BigFloat

    λbig = spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 7//20; precision=128)
    @test λbig isa BigFloat && precision(λbig) == 128
    @test Float64(λbig) ≈ spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 0.35) rtol=1e-13
end

if lowercase(get(ENV, "SWSH_FULL_TESTS", "false")) in ("true", "1") || "full" in ARGS
    include("full.jl")
end
