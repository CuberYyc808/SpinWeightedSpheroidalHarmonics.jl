# All the modes l of one (s, m, c) from one basis, against the single-l calls.
using SpinWeightedSpheroidalHarmonics, Test

const BASIS_CS = (0.0, 1e-8, 0.5, 3.0, 10.0, 30.0, -3.0)
const BASIS_THETAS = (1e-6, 0.2, 0.9, π / 2, 2.3, π - 1e-6)

#=
For each mode up to lmin + extra, the relative difference of the eigenvalue between the basis and the
single-l call (and of each against the 256-bit eigenvalue), and for the harmonic (value and two θ-derivatives, relative to the largest size of the
value or the derivative over a grid) the larger of the errors of the basis and of the single-l
harmonic against the single-l harmonic at 256 bits, and the difference between the two (the eigenvector
of a mode close to a neighbour is accurate only to about eps |M| / gap, in either). Also the number of
modes both refuse (unresolvable eigenvector in Float64).
=#
function basis_errors(s, m, c; extra=60)
    lmin = max(abs(s), abs(m))
    basis = spin_weighted_spheroidal_basis(s, m, c; lmax=lmin + extra)
    worst_lambda, worst_lambda_basis, worst_lambda_single = 0.0, 0.0, 0.0
    worst_basis, worst_single, worst_difference, refused = 0.0, 0.0, 0.0, 0
    for l in lmin:lmin + extra
        λ, λbasis = spin_weighted_spheroidal_eigenvalue(s, l, m, c), spin_weighted_spheroidal_eigenvalue(basis, l)
        λ256 = spin_weighted_spheroidal_eigenvalue(s, l, m, c; precision=256)
        scale = max(1, abs(λ))
        worst_lambda = max(worst_lambda, abs(λbasis - λ) / scale)
        worst_lambda_basis = max(worst_lambda_basis, Float64(abs(λbasis - λ256)) / scale)
        worst_lambda_single = max(worst_lambda_single, Float64(abs(λ - λ256)) / scale)
        single = try
            spin_weighted_spheroidal_harmonic(s, l, m, c)
        catch err
            occursin("cannot be resolved", err.msg) || rethrow()
            @test_throws ErrorException spin_weighted_spheroidal_harmonic(basis, l)
            refused += 1
            continue
        end
        from_basis = spin_weighted_spheroidal_harmonic(basis, l)
        reference = spin_weighted_spheroidal_harmonic(s, l, m, c; precision=256)
        # value_and_theta_derivatives is accurate up to the poles, where the separate calls of a
        # θ-derivative at large l are not
        grid = range(0.01, π - 0.01; length=64)
        scale = [maximum(abs(value_and_theta_derivatives(single, θ, 0.4)[k]) for θ in grid) for k in 1:3]
        scale = max.(scale, scale[1])
        for θ in BASIS_THETAS
            a, b = value_and_theta_derivatives(from_basis, θ, 0.4), value_and_theta_derivatives(single, θ, 0.4)
            r = setprecision(BigFloat, 256) do
                value_and_theta_derivatives(reference, big(θ), big(0.4))
            end
            worst_basis = max(worst_basis, maximum(Float64(abs(a[k] - r[k])) / scale[k] for k in 1:3))
            worst_single = max(worst_single, maximum(Float64(abs(b[k] - r[k])) / scale[k] for k in 1:3))
            worst_difference = max(worst_difference, maximum(abs(a[k] - b[k]) / scale[k] for k in 1:3))
        end
    end
    return (; worst_lambda, worst_lambda_basis, worst_lambda_single, worst_basis, worst_single, worst_difference, refused)
end

@testset "All modes of one (s, m, c) from one basis (sweep)" begin
    for s in (-2, 0, 2), m in (-3, 0, 2), c in BASIS_CS
        errors = basis_errors(s, m, c)
        # the eigenvalue of the basis is that of the single-l call to 4e-15, or closer to the 256-bit one
        @test errors.worst_lambda <= 4e-15 || errors.worst_lambda_basis <= errors.worst_lambda_single
        # the basis gives the single-l harmonic to 4e-15, or it is the more accurate of the two against 256
        # bits (up to a factor 3). Measured: the coefficients of the basis agree with those at 256 bits to
        # 1e-16 - 1e-14, those of the single-l call (truncated at a smaller matrix) to 1e-14 - 1e-13, so the
        # two harmonics differ by 2e-14 - 3e-13 for well separated modes, and by up to 4e-8 for modes close
        # to a neighbour at |c| = 30, where the eigenvector of either is accurate only to about eps |M| / gap
        @test errors.worst_difference <= 4e-15 || errors.worst_basis <= 3 * errors.worst_single
    end
end

@testset "All modes of one (s, m, c) from one basis" begin
    # Raising lmax keeps the modes already there and gives those of a basis built at the larger lmax
    small = spin_weighted_spheroidal_basis(-2, 2, 3.0; lmax=6)
    λ6 = spin_weighted_spheroidal_eigenvalue(small, 6)
    λ30 = spin_weighted_spheroidal_eigenvalue(small, 30)
    @test spin_weighted_spheroidal_eigenvalue(small, 6) === λ6
    large = spin_weighted_spheroidal_basis(-2, 2, 3.0; lmax=30)
    for l in 2:30
        @test spin_weighted_spheroidal_eigenvalue(small, l) ≈ spin_weighted_spheroidal_eigenvalue(large, l) rtol=4e-15
        @test spin_weighted_spheroidal_harmonic(small, l)(0.7, 0.2) ≈ spin_weighted_spheroidal_harmonic(large, l)(0.7, 0.2) rtol=1e-13 atol=1e-15
    end
    @test λ30 ≈ spin_weighted_spheroidal_eigenvalue(-2, 30, 2, 3.0) rtol=4e-15

    # BigFloat
    B = spin_weighted_spheroidal_basis(-2, 2, 3//1; lmax=8, precision=256)
    for l in 2:8
        λ = spin_weighted_spheroidal_eigenvalue(B, l)
        @test λ isa BigFloat && precision(λ) == 256
        @test abs(λ - spin_weighted_spheroidal_eigenvalue(-2, l, 2, 3//1; precision=256)) <= 1e-70 * abs(λ)
        H = spin_weighted_spheroidal_harmonic(B, l)
        @test abs(H(0.7, 0.2) - spin_weighted_spheroidal_harmonic(-2, l, 2, 3//1; precision=256)(0.7, 0.2)) <= 1e-70
    end

    @test_throws ArgumentError spin_weighted_spheroidal_basis(-2, 2, 3.0; lmax=1)
    @test_throws ArgumentError spin_weighted_spheroidal_eigenvalue(small, 1)
end
