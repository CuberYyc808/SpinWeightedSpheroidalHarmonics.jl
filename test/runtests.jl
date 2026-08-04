using SpinWeightedSpheroidalHarmonics
using LinearAlgebra
using Test

function _theta_integral_abs2(swsh; n::Int=2001, phi=0.0)
    thetas = range(0.0, π; length=n)
    h = step(thetas)
    vals = [abs2(swsh(θ, phi)) * sin(θ) for θ in thetas]
    # Trapezoidal rule on [0, π]
    return h * (sum(vals) - 0.5 * (vals[1] + vals[end]))
end

@testset "SpinWeightedSpheroidalHarmonics.jl" begin
    @testset "Jacobi spherical harmonic evaluation" begin
        test_spins = (-2, -1, 0, 1, 2)
        test_angles = [
            (0.37, 0.91),
            (1.10, 2.30),
            (2.40, -0.20),
        ]

        for s in test_spins
            for l in abs(s):20
                for m in -l:l
                    y_direct = spin_weighted_spherical_harmonic(s, l, m; method="direct")
                    y_jacobi = spin_weighted_spherical_harmonic(s, l, m; method="jacobi")

                    for (theta, phi) in test_angles
                        @test y_jacobi(theta, phi) ≈ y_direct(theta, phi) rtol=2e-8 atol=1e-10
                        @test y_jacobi(theta, phi; phi_derivative=2) ≈ y_direct(theta, phi; phi_derivative=2) rtol=2e-8 atol=1e-10
                        @test y_jacobi(theta, phi; theta_derivative=1) ≈ y_direct(theta, phi; theta_derivative=1) rtol=1e-8 atol=1e-10
                        @test y_jacobi(theta, phi; theta_derivative=2) ≈ y_direct(theta, phi; theta_derivative=2) rtol=1e-8 atol=1e-9
                        @test y_jacobi(theta, phi; theta_derivative=1, phi_derivative=1) ≈ y_direct(theta, phi; theta_derivative=1, phi_derivative=1) rtol=1e-8 atol=1e-10
                    end
                end
            end
        end
    end

    @testset "Auto method selection" begin
        @test spin_weighted_spherical_harmonic(-2, 10, 2; method="auto").method == :direct
        @test spin_weighted_spherical_harmonic(-2, 40, 2; method="auto").method == :jacobi
    end

    @testset "Jacobi symmetry tricks: m reflection" begin
        # Jacobi uses a symmetry relation for m < 0. Validate it against
        # direct evaluation, which is treated as the golden standard.
        symmetry_pairs = [
            (-2, 10, 3),
            (-1, 9, 2),
            (1, 8, 2),
            (2, 11, 1),
        ]
        symmetry_angles = [
            (0.42, 0.30),
            (1.70, -0.80),
            (2.60, 2.20),
        ]

        for (s, l, mpos) in symmetry_pairs
            mneg = -mpos
            y_jacobi_neg = spin_weighted_spherical_harmonic(s, l, mneg; method="jacobi")
            y_direct_neg = spin_weighted_spherical_harmonic(s, l, mneg; method="direct")
            y_direct_ref = spin_weighted_spherical_harmonic(-s, l, mpos; method="direct")
            phase = (-1)^(s - mneg)

            for (theta, phi) in symmetry_angles
                # Golden-standard agreement for the negative-m mode.
                @test y_jacobi_neg(theta, phi) ≈ y_direct_neg(theta, phi) rtol=1e-9 atol=1e-12
                @test y_jacobi_neg(theta, phi; theta_derivative=1) ≈ y_direct_neg(theta, phi; theta_derivative=1) rtol=1e-9 atol=1e-10
                @test y_jacobi_neg(theta, phi; theta_derivative=2) ≈ y_direct_neg(theta, phi; theta_derivative=2) rtol=1e-9 atol=1e-9

                # Explicitly check the symmetry trick against direct.
                @test y_jacobi_neg(theta, phi) ≈ phase * conj(y_direct_ref(theta, phi)) rtol=1e-9 atol=1e-12
                @test y_jacobi_neg(theta, phi; theta_derivative=1) ≈ phase * conj(y_direct_ref(theta, phi; theta_derivative=1)) rtol=1e-9 atol=1e-10
                @test y_jacobi_neg(theta, phi; theta_derivative=1, phi_derivative=1) ≈ phase * conj(y_direct_ref(theta, phi; theta_derivative=1, phi_derivative=1)) rtol=1e-9 atol=1e-10
            end
        end
    end

    @testset "Jacobi symmetry tricks: theta folding" begin
        # Jacobi maps theta to [0, pi] using (theta, phi) -> (2pi-theta, phi+pi)
        # when needed. Validate this mapping with direct evaluation.
        fold_modes = [
            (-2, 8, 2),
            (-2, 8, -2),
            (0, 9, 0),
            (1, 7, -1),
        ]
        principal_angles = [
            (0.70, 0.40),
            (1.40, -1.10),
            (2.20, 2.20),
        ]

        for (s, l, m) in fold_modes
            y_direct = spin_weighted_spherical_harmonic(s, l, m; method="direct")
            y_jacobi = spin_weighted_spherical_harmonic(s, l, m; method="jacobi")

            for (theta, phi) in principal_angles
                theta_folded = 2π - theta
                theta_negative = -theta

                @test y_jacobi(theta_folded, phi) ≈ y_direct(theta, phi + π) rtol=1e-9 atol=1e-12
                @test y_jacobi(theta_negative, phi) ≈ y_direct(theta, phi + π) rtol=1e-9 atol=1e-12
                @test y_jacobi(theta_folded, phi; phi_derivative=2) ≈ y_direct(theta, phi + π; phi_derivative=2) rtol=1e-9 atol=1e-10
                @test y_jacobi(theta_negative, phi; phi_derivative=2) ≈ y_direct(theta, phi + π; phi_derivative=2) rtol=1e-9 atol=1e-10
                @test y_jacobi(theta_folded, phi; theta_derivative=1) ≈ y_direct(theta, phi + π; theta_derivative=1) rtol=1e-9 atol=1e-10
                @test y_jacobi(theta_negative, phi; theta_derivative=1) ≈ y_direct(theta, phi + π; theta_derivative=1) rtol=1e-9 atol=1e-10
            end
        end
    end

    @testset "Case-insensitive method options" begin
        @test spin_weighted_spherical_harmonic(-2, 40, 2; method="JaCoBi").method == :jacobi
        @test spin_weighted_spherical_harmonic(-2, 40, 2; method=:CHEBYSHEV).method == :chebyshev
        @test spin_weighted_spherical_harmonic(-2, 10, 2; method="DiReCt").method == :direct
        @test spin_weighted_spherical_harmonic(-2, 10, 2; method=:AUTO).method == :direct
    end

    @testset "High-l Jacobi is finite" begin
        y = spin_weighted_spherical_harmonic(-2, 200, 2; method="jacobi")
        val = y(1.1, 0.4)
        @test isfinite(real(val))
        @test isfinite(imag(val))
    end

    @testset "Jacobi normalization convention" begin
        modes = [
            (-2, 6, 2),
            (-2, 12, 2),
            (-1, 15, 1),
            (0, 20, 0),
            (2, 18, -1),
        ]

        for (s, l, m) in modes
            y_direct = spin_weighted_spherical_harmonic(s, l, m; method="direct")
            y_jacobi = spin_weighted_spherical_harmonic(s, l, m; method="jacobi")

            norm_direct_phi0 = _theta_integral_abs2(y_direct; phi=0.0)
            norm_jacobi_phi0 = _theta_integral_abs2(y_jacobi; phi=0.0)
            norm_direct_phi1 = _theta_integral_abs2(y_direct; phi=1.3)
            norm_jacobi_phi1 = _theta_integral_abs2(y_jacobi; phi=1.3)

            # Jacobi should match direct normalization for the same mode.
            @test norm_jacobi_phi0 ≈ norm_direct_phi0 rtol=1e-9 atol=1e-12
            @test norm_jacobi_phi1 ≈ norm_direct_phi1 rtol=1e-9 atol=1e-12

            # Direct evaluation method follows the package convention:
            # ∫_0^π |sY_lm(θ, ϕ)|^2 sinθ dθ = 1/(2π), independent of ϕ.
            @test norm_direct_phi0 ≈ 1 / (2π) rtol=5e-5 atol=5e-7
            @test norm_direct_phi1 ≈ 1 / (2π) rtol=5e-5 atol=5e-7
        end
    end

    @testset "Spheroidal lambda consistency" begin
        s = -2
        l = 8
        m = 2
        c = 0.3
        swsh = spin_weighted_spheroidal_harmonic(s, l, m, c; method="JaCoBi")
        @test swsh.lambda ≈ spin_weighted_spheroidal_eigenvalue(s, l, m, c)
    end

    @testset "Real eigenvalue fast path" begin
        cases = [
            (-2, 2, 2, 0.45),
            (2, 2, -2, -1.0e-7),
            (-2, 8, 0, 0.5),
            (-2, 32, 0, 0.5),
            (-2, 50, 0, 10.0),
            (2, 50, -25, -0.99999),
        ]

        for (s, l, m, c) in cases
            N = SpinWeightedSpheroidalHarmonics._determine_matrix_size_N(s, l, m)
            reference, _ = SpinWeightedSpheroidalHarmonics._spectral_decomposition(c, s, l, m, N)
            candidate = SpinWeightedSpheroidalHarmonics._angular_eigenvalue(c, s, l, m, N)
            @test candidate ≈ reference rtol=5e-14 atol=2e-12
        end
    end

    @testset "Adaptive high-c eigenvalue" begin
        cases = [
            (1, 2, -2, -0.99999, -0.03510321969441066),
            (0, 2, 0, -9.9999, 54.50971251592067),
            (1, 2, 0, -9.0, 47.5455735438706),
            (2, 2, 0, -45.0, 87.01704710570348),
            (1, 50, -50, -99.999, -143.41923491519816),
        ]

        for (s, l, m, c, reference) in cases
            candidate = spin_weighted_spheroidal_eigenvalue(s, l, m, c)
            @test isapprox(candidate, reference; rtol=3e-14, atol=5e-13)
        end
    end

    @testset "Small-c eigenvalue" begin
        cases = [
            (-2, 2, 0, -1.0e-7, 4.0000000000000047619047619047605),
            (2, 2, -2, 1.0e-6, 6.66666693121699e-6),
            (1, 50, 0, -9.9999e-8, 2548.0),
            (-1, 50, 0, -9.9999e-8, 2550.0),
        ]

        for (s, l, m, c, reference) in cases
            candidate = spin_weighted_spheroidal_eigenvalue(s, l, m, c)
            @test isapprox(candidate, reference; rtol=0, atol=1e-18)
        end

        inside = SpinWeightedSpheroidalHarmonics._adaptive_real_lambda(
            1.0e-6, -2, 2, 0)
        outside = SpinWeightedSpheroidalHarmonics._adaptive_real_lambda(
            nextfloat(1.0e-6), -2, 2, 0)
        @test inside.refinement == 0
        @test outside.refinement > 0
    end

    @testset "Spherical limit (c = 0)" begin
        s = -2
        l = 6
        m = 2
        spheroidal = spin_weighted_spheroidal_harmonic(s, l, m, 0.0; method="direct")
        spherical = spin_weighted_spherical_harmonic(s, l, m; method="direct")
        @test spheroidal(1.1, 0.7) ≈ spherical(1.1, 0.7) rtol=1e-12 atol=1e-12
    end

    @testset "Signed-frequency angular cache" begin
        cache = AngularCache(max_entries=8)
        positive = continue_angular_mode(
            -2, 3, 2, 0.4 + 0.15im; cache)
        negative = continue_angular_mode(
            -2, 3, 2, -0.4 + 0.15im; cache)
        @test positive.c != negative.c
        @test positive.lambda != negative.lambda
        @test length(cache.values) == 2
        repeated = continue_angular_mode(
            -2, 3, 2, 0.4 + 0.15im; cache)
        @test repeated === positive
    end

    @testset "Complex angular continuation" begin
        path = ComplexF64[
            0.0,
            0.15 - 0.05im,
            0.30 - 0.10im,
            0.45 - 0.20im,
        ]
        result = track_angular_mode(
            -2, 4, 2, path;
            sheet_id=:test_path, truncation_order=24)
        @test result.status == :open
        @test last(result.states).c == last(path)
        @test all(state.residual <= 5e-12 for state in result.states)
        @test all(state.previous_overlap >= 0.65
            for state in Iterators.drop(result.states, 1))
        @test all(abs(imag(dot(
                result.states[index - 1].coefficients,
                result.states[index].coefficients))) <= 5e-13
            for index in 2:length(result.states))
    end

    @testset "Independent lateral sheets and mirror" begin
        lateral = continue_angular_lateral_pair(
            -2, 3, 1, 0.7, 0.4, 1e-4;
            truncation_order=24)
        right = last(lateral.right.states)
        left = last(lateral.left.states)
        @test right.sheet_id == :right_lateral
        @test left.sheet_id == :left_lateral
        @test real(right.c) == -real(left.c)
        @test imag(right.c) == imag(left.c)

        pair = continue_angular_mode(
            -2, 3, 1, 0.35 - 0.2im;
            sheet_id=:mirror_source, truncation_order=24,
            cache=nothing)
        mirrored_parameters = mirror_angular_parameters(
            pair.s, pair.l, pair.m, pair.c)
        mirrored = continue_angular_mode(
            mirrored_parameters.s, mirrored_parameters.l,
            mirrored_parameters.m, mirrored_parameters.c;
            sheet_id=:mirror_target, truncation_order=24,
            cache=nothing)
        @test angular_mirror_residual(pair, mirrored) <= 2e-12
    end

    @testset "Angular monodromy and observables" begin
        loop = ComplexF64[
            0.0,
            0.15,
            0.15 + 0.1im,
            0.0 + 0.1im,
            0.0,
        ]
        result = track_angular_mode(
            -2, 2, 2, loop;
            sheet_id=:closed_test, truncation_order=24)
        @test result.status == :closed
        @test result.lambda_closure_error <= 1e-12
        @test abs(abs(result.monodromy_overlap) - 1) <= 1e-10

        pair = continue_angular_mode(
            -2, 3, 2, 0.25 - 0.15im;
            sheet_id=:observable_test, truncation_order=24,
            cache=nothing)
        public_lambda = spin_weighted_spheroidal_eigenvalue(
            pair.s, pair.l, pair.m, pair.c;
            N=pair.matrix_size)
        public_harmonic = spin_weighted_spheroidal_harmonic(
            pair.s, pair.l, pair.m, pair.c;
            N=pair.matrix_size, method="auto")
        @test public_lambda ≈ pair.lambda rtol=2e-13 atol=2e-13
        @test public_harmonic.lambda ≈ pair.lambda rtol=2e-13 atol=2e-13
        observables = angular_observables(
            pair, 1.1, 0.3; derivative_order=2)
        @test public_harmonic(1.1, 0.3) ≈ observables.value
        @test isfinite(observables.value)
        @test isfinite(observables.first_derivative)
        @test isfinite(observables.second_derivative)
        @test observables.lambda == pair.lambda
    end

    @testset "Angular precision and truncation certificate" begin
        certificate = angular_precision_certificate(
            -2, 2, 2, 0.2 - 0.1im;
            truncation_orders=(24, 32),
            precision_bits=(128, 160))
        @test certificate.accepted
        @test certificate.truncation_drift <= 1e-12
        @test certificate.precision_drift <= 1e-14
        @test certificate.eigenvector_overlap >= 1 - 1e-10
    end
end
