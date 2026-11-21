# The typed accessors of a harmonic and an eigenpair, and the documented guarantee of a BigFloat eigenpair.
using SpinWeightedSpheroidalHarmonics, LinearAlgebra, Test

@testset "Typed accessors" begin
    S = spin_weighted_spheroidal_harmonic(-2, 3, 1, 0.7)
    @test spin_weighted_spheroidal_eigenvalue(S) === Float64(S.lambda)
    @test spin_weighted_spheroidal_eigenvalue(S) ≈ spin_weighted_spheroidal_eigenvalue(-2, 3, 1, 0.7) rtol=1e-13
    @test angular_parameters(S) === (s=-2, l=3, m=1, c=0.7)
    S0 = spin_weighted_spheroidal_harmonic(-2, 3, 1, 0)
    @test spin_weighted_spheroidal_eigenvalue(S0) isa Float64
    @test spin_weighted_spheroidal_eigenvalue(S0) == 10
    @test angular_parameters(S0).c === 0.0
    Sc = spin_weighted_spheroidal_harmonic(-2, 3, 1, 0.7 - 0.2im)
    @test spin_weighted_spheroidal_eigenvalue(Sc) isa ComplexF64
    @test angular_parameters(Sc).c === 0.7 - 0.2im
    @test spin_weighted_spheroidal_eigenvalue(spin_weighted_spheroidal_harmonic(-2, 3, 1, 0.7; method="leaver")) isa Float64
    B = spin_weighted_spheroidal_harmonic(-2, 3, 1, 7//10; precision=256)
    λ = spin_weighted_spheroidal_eigenvalue(B)
    @test λ isa BigFloat && precision(λ) == 256
    @test angular_parameters(B).c isa BigFloat
    pair = continue_angular_mode(-2, 3, 1, 0.7 - 0.2im)
    @test spin_weighted_spheroidal_eigenvalue(pair) === pair.lambda
    @test angular_parameters(pair) === (s=-2, l=3, m=1, c=pair.c)
    Y = spin_weighted_spherical_harmonic(-2, 3, 1)
    @test spin_weighted_spherical_eigenvalue(Y) === 10.0
    @test spin_weighted_spherical_eigenvalue(Float64, -2, 3) === 10.0
    @test angular_parameters(Y) === (s=-2, l=3, m=1, c=0.0)
    # inferred
    @test (@inferred spin_weighted_spheroidal_eigenvalue(pair)) isa ComplexF64
    @test (@inferred angular_parameters(Y)) isa NamedTuple
end

@testset "The guarantee of a BigFloat eigenpair" begin
    # coefficients of unit norm, residual at the roundoff level (far below the documented bound), an accepted status
    for (s, l, m, c) in ((-2, 2, 2, 1 - 4im), (-2, 5, -3, 3//10 - 2im//5), (0, 4, 1, 6 - 1im), (2, 7, 0, 12 - 3im))
        pair = continue_angular_mode(s, l, m, c; precision=256)
        T = BigFloat
        setprecision(BigFloat, 256) do
            @test abs(sum(abs2, pair.coefficients) - 1) <= 10eps(T)
            @test pair.residual <= 16 * sqrt(pair.matrix_size) * eps(T)
            @test pair.residual <= SpinWeightedSpheroidalHarmonics._tol(T, 5e-12)
        end
        @test pair.status in (:spherical_anchor, :predictor_corrected, :continued, :banded, :precision_tracked, :near_collision)
    end
end
