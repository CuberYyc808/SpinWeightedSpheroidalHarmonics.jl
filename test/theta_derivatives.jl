# value_and_theta_derivatives: the value and the first and second theta-derivatives in one pass.
using SpinWeightedSpheroidalHarmonics, Test

const ONE_PASS_CS = (0.0, 0.5, 2.5, 19.6, 0.5 - 0.2im, 2.5 - 1.0im, 19.6 - 4.0im)
const ONE_PASS_BASE = (1e-8, 1e-5, 1e-3, 0.3, π / 2)

#=
Against the three separate calls, for the modes of spin s and spheroidicity c with l in ls: the
largest difference of each of the three, relative to its largest size over the angles, away from the
poles (theta >= 1e-3 from them) and near them; near them the separate calls of the second derivative
go through the direct sum, so the one-pass result is also compared with the same harmonic evaluated
at 256 bits there. Also the number of modes Float64 refuses (nearly degenerate at |c| = 19.6).
=#
function one_pass_errors(s, c; ls=abs(s):20)
    far = (0.3, π / 2, π - 0.3, 1e-3, π - 1e-3)
    near = (1e-8, 1e-5, π - 1e-8, π - 1e-5)
    worst_far = zeros(3)
    worst_near_big = zeros(3)
    refused = 0
    for l in ls, m in -l:l
        S = try
            spin_weighted_spheroidal_harmonic(s, l, m, c)
        catch err
            message = err isa AngularContinuationError ? err.message : err isa ErrorException ? err.msg : ""
            occursin("cannot be resolved", message) || rethrow()
            refused += 1
            continue
        end
        grid = range(0.02, π - 0.02; length=24)
        scale = [maximum(abs(S(θ, 0.7; theta_derivative=k)) for θ in grid) for k in 0:2]
        for θ in far
            one_pass = value_and_theta_derivatives(S, θ, 0.7)
            for k in 0:2
                worst_far[k + 1] = max(worst_far[k + 1], abs(one_pass[k + 1] - S(θ, 0.7; theta_derivative=k)) / scale[k + 1])
            end
        end
        for θ in near
            one_pass = value_and_theta_derivatives(S, θ, 0.7)
            reference = setprecision(BigFloat, 256) do
                [S(big(θ), big(0.7); theta_derivative=k) for k in 0:2]
            end
            for k in 0:2
                worst_near_big[k + 1] = max(worst_near_big[k + 1], Float64(abs(one_pass[k + 1] - reference[k + 1]) / scale[k + 1]))
            end
        end
    end
    return (; worst_far, worst_near_big, refused)
end

@testset "The harmonic and two θ-derivatives in one pass (sweep)" begin
    refused = 0
    for s in (-2, 2), c in ONE_PASS_CS
        errors = one_pass_errors(s, c)
        @test all(errors.worst_far .<= 1e-13)
        @test all(errors.worst_near_big .<= 1e-13)
        refused += errors.refused
    end
    @test refused < 100
end

@testset "The harmonic and two θ-derivatives in one pass" begin
    # At the poles: the exact pole values of the separate calls
    S = spin_weighted_spheroidal_harmonic(-2, 7, -3, 1.5 - 0.5im)
    for θ in (0.0, π, 2π)
        @test isequal(value_and_theta_derivatives(S, θ, 0.3),
            (S(θ, 0.3), S(θ, 0.3; theta_derivative=1), S(θ, 0.3; theta_derivative=2)))
    end
    # m = ±s: the harmonic does not vanish at a pole, nor does its second derivative
    S = spin_weighted_spheroidal_harmonic(-2, 5, 2, 2.5)
    reference = setprecision(BigFloat, 256) do
        S(big(1e-9), big(0.0); theta_derivative=2)
    end
    @test abs(value_and_theta_derivatives(S, 1e-9)[3] - reference) <= 1e-13 * abs(reference)

    # The first two agree with value_and_theta_derivative, the vector and in-place forms with the scalar one
    S = spin_weighted_spheroidal_harmonic(-2, 7, -3, 1.5 - 0.5im)
    angles = [0.0, 1e-9, 0.4, 2.0, π, 4.0]
    for θ in angles
        three = value_and_theta_derivatives(S, θ, 0.3)
        two = value_and_theta_derivative(S, θ, 0.3)
        @test three[1] ≈ two[1] rtol=1e-13 atol=1e-15
        @test three[2] ≈ two[2] rtol=1e-12 atol=1e-14
    end
    values, firsts, seconds = value_and_theta_derivatives(S, angles, 0.3)
    @test isequal(values, [value_and_theta_derivatives(S, θ, 0.3)[1] for θ in angles])
    @test isequal(firsts, [value_and_theta_derivatives(S, θ, 0.3)[2] for θ in angles])
    @test isequal(seconds, [value_and_theta_derivatives(S, θ, 0.3)[3] for θ in angles])
    buffers = (similar(values), similar(values), similar(values))
    @test value_and_theta_derivatives!(buffers..., S, angles, 0.3) === buffers
    @test isequal(buffers, (values, firsts, seconds))
    @test_throws DimensionMismatch value_and_theta_derivatives!(similar(values, 2), firsts, seconds, S, angles)

    # The other methods give the three separate calls, the spherical harmonics too
    for method in ("direct", "chebyshev", "leaver")
        h = spin_weighted_spheroidal_harmonic(-2, 3, 1, 0.7; method)
        @test isequal(value_and_theta_derivatives(h, 1.1, 0.3),
            (h(1.1, 0.3), h(1.1, 0.3; theta_derivative=1), h(1.1, 0.3; theta_derivative=2)))
    end
    for method in ("jacobi", "direct"), θ in (1e-6, 0.8, π - 1e-6)
        Y = spin_weighted_spherical_harmonic(2, 9, -4; method)
        reference = setprecision(BigFloat, 256) do
            [Y(big(θ), big(0.3); theta_derivative=k) for k in 0:2]
        end
        three = value_and_theta_derivatives(Y, θ, 0.3)
        for k in 0:2
            @test abs(three[k + 1] - reference[k + 1]) <= 1e-12 * max(1, abs(reference[k + 1]))
        end
    end

    # BigFloat: the one-pass result keeps the precision of the harmonic
    B = spin_weighted_spheroidal_harmonic(-2, 5, -3, 7//20 - 1im//10; precision=256)
    three = value_and_theta_derivatives(B, 1.1, 0.3)
    @test three[3] isa Complex{BigFloat} && precision(real(three[3])) == 256
    for k in 0:2
        @test abs(three[k + 1] - B(1.1, 0.3; theta_derivative=k)) <= 1e-70
    end

    # No allocation per Float64 call, scalar and in place
    S = spin_weighted_spheroidal_harmonic(-2, 9, 2, 2.5)
    evaluate(S) = value_and_theta_derivatives(S, 1.1, 0.0)
    evaluate(S)
    @test (@allocated evaluate(S)) == 0
    angles = collect(range(0.1, 3.0; length=50))
    buffers = (zeros(ComplexF64, 50), zeros(ComplexF64, 50), zeros(ComplexF64, 50))
    fill_buffers(S, buffers, angles) = value_and_theta_derivatives!(buffers..., S, angles, 0.0)
    fill_buffers(S, buffers, angles)
    @test (@allocated fill_buffers(S, buffers, angles)) == 0
end
