# All the modes l of one (s, m, c), real c, from one decomposition of the shared spectral matrix.

#=
The eigenpairs of the modes lmin + first - 1 .. lmin + last - 1 (positions first:last among the sorted
eigenvalues of the λ-matrix) are found together, in Float64 by one LAPACK call (all the eigenpairs); the matrix size starts
from the estimate for the highest mode and grows, as in _real_eigenpair_seed, until every eigenvector
has a small coefficient tail and residual. A wider working precision refines each eigenpair on the
band matrix of that size, as _adaptive_real_eigenpair does; in Float64 each eigenpair is refined when it
is first asked for (_refine_basis_mode). A mode whose eigenvector this larger
matrix cannot resolve is solved on its own, as by the single-l call; one that cannot be resolved at all
is recorded with a zero gap, and refused only when it is asked for.
=#
struct _BasisBlock{T}
    lambdas::Vector{T}
    coefficients::Vector{Vector{Complex{T}}}
    gaps::Vector{T}
    # what is left to do when a mode is first asked for: refine it on this band matrix (_refine_basis_mode),
    # :eigenvalue (solve for the eigenvalue alone, of a mode whose eigenvector cannot be resolved), or nothing
    pending::Vector{Union{Nothing, Symbol, Matrix{Float64}}}
end

function _basis_block(::Type{T}, c::T, s::Int, m::Int, first::Int, last::Int) where {T}
    lmin = max(abs(s), abs(m))
    c64 = Float64(c)
    size = max(_determine_matrix_size_N(s, lmin + last - 1, m, c64), last + 1)
    _seed_is_final(T) || (size = max(size, _determine_matrix_size_N(s, lmin + last - 1, m, c)))
    step = max(8, ceil(Int, abs(c64) / 8)) # as in _real_eigenpair_seed
    for _ in 0:SWSH_MAX_REFINEMENTS
        block = _basis_block_at_size(T, c, s, m, first, last, size)
        block === nothing || return block
        size += step
    end
    error("SWSH basis failed to converge after $(SWSH_MAX_REFINEMENTS) matrix refinements.")
end

function _basis_block_at_size(::Type{T}, c::T, s::Int, m::Int, first::Int, last::Int, size::Int) where {T}
    band = _real_lambda_band(Float64(c), s, m, size)
    matrix_norm = _symmetric_band_inf_norm(band)
    # all the eigenpairs at once: divide and conquer is several times faster than selecting the block
    # (with dsbevx, which bisects for each eigenvalue)
    values, vectors = _all_banded_eigenpairs!(copy(band))
    final = _seed_is_final(T)
    working_band = final ? band : _real_lambda_band(c, s, m, size)
    count = last - first + 1
    lambdas = Vector{T}(undef, count)
    coefficients = Vector{Vector{Complex{T}}}(undef, count)
    gaps = Vector{T}(undef, count)
    pending = Vector{Union{Nothing, Symbol, Matrix{Float64}}}(nothing, count)
    for (k, index) in enumerate(first:last)
        position = index
        gap = minimum((abs(values[j] - values[position]) for j in eachindex(values) if j != position);
            init=Inf) / max(matrix_norm, floatmin(Float64))
        vector = _sign_normalized(view(vectors, :, position), index)
        identified = gap >= _eigenvector_gap_tol(Float64)
        if identified
            _coefficient_tail(vector) <= _coefficient_tail_tol(Float64) || return nothing
            _band_relative_residual(band, values[position], vector) <= _eigenvector_residual_tol(Float64) ||
                return nothing
        end
        if final && identified
            # refined when first asked for (c = 0 is exact already: the matrix is diagonal)
            value = values[position]
            iszero(c) || (pending[k] = band)
        elseif final && gap * matrix_norm < _eigenvector_gap_tol(Float64) * abs(values[position])
            # Unresolvable on its own too (the matrix of the single mode has a norm of at least |λ|); the
            # eigenvalue is solved for alone when asked for, as by the single-l call
            value, gap = T(values[position]), zero(T)
            pending[k] = :eigenvalue
        elseif final
            #=
            The gap relative to the norm of this matrix, larger than the matrix of a single mode, is too
            small to resolve the eigenvector here; the mode is solved on its own as the single-l call
            does, which also decides whether it can be resolved at all
            =#
            value, vector, gap = try
                pair = _adaptive_real_eigenpair(c, s, _basis_lmin(s, m) + index - 1, m)
                pair.lambda, real.(pair.coefficients), T(Inf)
            catch err
                err isa ErrorException || rethrow()
                pending[k] = :eigenvalue
                T(values[position]), vector, zero(T) # refused when it is asked for
            end
        elseif identified || gap >= 10eps(Float64)
            # Unidentified in Float64 (but with distinct eigenvalues), as _adaptive_real_eigenpair does
            value, vector, gap = try
                _refined_real_eigenpair(working_band, index, values[position], vector, gap, identified,
                    s, _basis_lmin(s, m) + index - 1, m, c)
            catch
                T(values[position]), T.(vector), zero(T) # unresolvable in T too: refused when asked for
            end
            _coefficient_tail(vector) <= _coefficient_tail_tol(T) || return nothing
        else
            value = T(values[position])
        end
        lambdas[k] = T(value)
        coefficients[k] = Complex{T}.(vector)
        gaps[k] = T(gap)
    end
    return _BasisBlock{T}(lambdas, coefficients, gaps, pending)
end

#=
Divide and conquer leaves each eigenvector accurate to about eps |M| / gap for the whole matrix,
although its residual is already at roundoff relative to |M|; one step of inverse iteration on the band
(O(N)) makes it as accurate as the inverse iteration of the single-l solver, and the eigenvalue is then
its Rayleigh quotient, accurate to the roundoff of the entries the localized eigenvector covers rather
than to eps |M|. This is done when the mode is first asked for, so that preparing the basis costs one
decomposition.
=#
function _refine_basis_mode(band::Matrix{Float64}, value::Float64, vector::AbstractVector, index::Int)
    refined = _sign_normalized(_shifted_band_solve(_shifted_band_lu(band, value), real.(vector)), index)
    return transpose(refined) * _symmetric_band_mul(band, refined), ComplexF64.(refined)
end

_basis_lmin(s::Int, m::Int) = max(abs(s), abs(m))

# All the eigenvalues (ascending) and unit eigenvectors of the symmetric band matrix, by LAPACK's
# divide and conquer (dsbevd); band is overwritten
function _all_banded_eigenpairs!(band::Matrix{Float64})
    n = size(band, 2)
    kd = size(band, 1) - 1
    eigenvalues = zeros(Float64, n)
    eigenvectors = zeros(Float64, n, n)
    lwork = 1 + 5n + 2n^2
    liwork = 3 + 5n
    work = zeros(Float64, lwork)
    integer_work = zeros(BlasInt, liwork)
    info = Ref{BlasInt}()
    ccall((@blasfunc(dsbevd_), SWSH_LAPACK), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
            Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
            Clong, Clong),
        'V', 'U', n, kd, band, kd + 1, eigenvalues, eigenvectors, n, work, lwork, integer_work, liwork, info, 1, 1)
    info[] == 0 || error("LAPACK dsbevd failed with info=$(info[]).")
    return eigenvalues, eigenvectors
end

"""
    SpinWeightedSpheroidalBasis{T}

The modes of one `(s, m, c)`, real `c`, returned by [`spin_weighted_spheroidal_basis`](@ref). Pass it
to `spin_weighted_spheroidal_eigenvalue(basis, l)` and `spin_weighted_spheroidal_harmonic(basis, l)`.
"""
struct SpinWeightedSpheroidalBasis{T<:AbstractFloat}
    s::Int
    m::Int
    c::T
    precision::Int
    lambdas::Vector{T}
    coefficients::Vector{Vector{Complex{T}}}
    gaps::Vector{T}
    pending::Vector{Union{Nothing, Symbol, Matrix{Float64}}}
    # the Jacobi spherical harmonics of the expansion, built when first needed and shared by the modes
    spherical::Vector{Union{Nothing, SpinWeightedSphericalHarmonicFunction}}
    lock::ReentrantLock
end

function Base.show(io::IO, ::MIME"text/plain", basis::SpinWeightedSpheroidalBasis)
    print(io, "SpinWeightedSpheroidalBasis(s = $(basis.s), m = $(basis.m), c = $(basis.c), lmax = $(_basis_lmax(basis)))")
end

_basis_lmax(basis::SpinWeightedSpheroidalBasis) = _basis_lmin(basis.s, basis.m) + length(basis.lambdas) - 1

@doc raw"""
    spin_weighted_spheroidal_basis(s::Int, m::Int, c::Real; lmax::Int, precision=nothing)

Prepare the spin-weighted spheroidal modes `l = max(|s|, |m|), ..., lmax` of spin weight `s`,
azimuthal index `m` and real spheroidicity `c` together, from one decomposition of the spectral matrix
they share. Then
```julia
basis = spin_weighted_spheroidal_basis(-2, 2, 3.0; lmax=12)
spin_weighted_spheroidal_eigenvalue(basis, l)   # λ of mode l, in the floating-point type of c
spin_weighted_spheroidal_harmonic(basis, l)     # the harmonic of mode l
```
need no further eigen-solve. They give the eigenvalue and the harmonic (with its normalization, sign
convention and pole behaviour) of `spin_weighted_spheroidal_eigenvalue(s, l, m, c)` and
`spin_weighted_spheroidal_harmonic(s, l, m, c)`: the eigenvalues agree to a few eps, and the
spherical-harmonic coefficients are those of the single-l call to its accuracy (the basis solves a
larger matrix, so its coefficients are usually the more accurate of the two; for a mode close to a
neighbour both are accurate only to about eps times the norm of the matrix over the gap). The harmonics are evaluated with the
Jacobi recurrence (the default method), and the spherical harmonics of the expansion are built once
and shared by all of them.

Asking for an `l` above `lmax` extends the basis (to at least half as many modes again), with one
more decomposition for the new modes only; the modes already prepared are kept. A basis may be used
by several tasks at once.

The computation is in the floating-point type of `c` (at least Float64), or in `BigFloat` at
`precision` bits. A mode whose eigenvector cannot be resolved in that precision (nearly degenerate
with a neighbour) raises the same error as `spin_weighted_spheroidal_harmonic` when it is asked for.
"""
function spin_weighted_spheroidal_basis(s::Int, m::Int, c::Real; lmax::Int,
        precision::Union{Nothing, Integer}=nothing)
    lmin = _basis_lmin(s, m)
    lmax >= lmin || throw(ArgumentError("lmax = $lmax is below the lowest mode l = max(|s|, |m|) = $lmin."))
    _check_precision_keyword(precision)
    c = _with_precision_keyword(c, precision)
    T = _float_type(c)
    bits = precision === nothing ? _input_precision(c) : precision
    return _with_precision(T, bits) do
        x = T(c)
        block = _basis_block(T, x, s, m, 1, lmax - lmin + 1)
        SpinWeightedSpheroidalBasis{T}(s, m, x, bits, block.lambdas, block.coefficients, block.gaps, block.pending,
            Union{Nothing, SpinWeightedSphericalHarmonicFunction}[], ReentrantLock())
    end
end

# The eigenvalue, coefficients and gap of mode l, extending the basis first if needed
function _basis_mode(basis::SpinWeightedSpheroidalBasis{T}, l::Int) where {T}
    s, m = basis.s, basis.m
    lmin = _basis_lmin(s, m)
    l >= lmin || throw(ArgumentError("l = $l is below the lowest mode l = max(|s|, |m|) = $lmin for s = $s, m = $m."))
    index = l - lmin + 1
    return lock(basis.lock) do
        count = length(basis.lambdas)
        if index > count
            last = max(index, count + max(8, count ÷ 2))
            block = _with_precision(() -> _basis_block(T, basis.c, s, m, count + 1, last), T, basis.precision)
            append!(basis.lambdas, block.lambdas)
            append!(basis.coefficients, block.coefficients)
            append!(basis.gaps, block.gaps)
            append!(basis.pending, block.pending)
        end
        task = basis.pending[index]
        if task isa Matrix{Float64}
            basis.lambdas[index], basis.coefficients[index] =
                _refine_basis_mode(task, basis.lambdas[index], basis.coefficients[index], index)
        elseif task === :eigenvalue
            basis.lambdas[index] = Teukolsky_lambda_const(basis.c, s, l, m)
        end
        basis.pending[index] = nothing
        basis.lambdas[index], basis.coefficients[index], basis.gaps[index]
    end
end

"""
    spin_weighted_spheroidal_eigenvalue(basis::SpinWeightedSpheroidalBasis, l::Int)

The eigenvalue of mode `l` of a basis (see [`spin_weighted_spheroidal_basis`](@ref)), in its
floating-point type.
"""
function spin_weighted_spheroidal_eigenvalue(basis::SpinWeightedSpheroidalBasis{T}, l::Int) where {T}
    return first(_basis_mode(basis, l))
end

"""
    spin_weighted_spheroidal_harmonic(basis::SpinWeightedSpheroidalBasis, l::Int)

The harmonic of mode `l` of a basis (see [`spin_weighted_spheroidal_basis`](@ref)), the same function
as `spin_weighted_spheroidal_harmonic(s, l, m, c)` builds.
"""
function spin_weighted_spheroidal_harmonic(basis::SpinWeightedSpheroidalBasis{T}, l::Int) where {T}
    lambda, coefficients, gap = _basis_mode(basis, l)
    s, m, c = basis.s, basis.m, basis.c
    gap >= _eigenvector_gap_tol(T) || error(_unresolvable_eigenvector_message(s, l, m, c, Float64(gap)))
    N = length(coefficients)
    keep = [abs(b) >= _spherical_coefficient_tol(T) for b in coefficients]
    spherical_harmonics_l = lock(basis.lock) do
        spherical = basis.spherical
        length(spherical) < N && append!(spherical, fill(nothing, N - length(spherical)))
        lmin = _basis_lmin(s, m)
        for n in 1:N
            keep[n] && spherical[n] === nothing && (spherical[n] = spin_weighted_spherical_harmonic(
                s, lmin + n - 1, m; method="jacobi", precision=_precision_keyword(basis.precision)))
        end
        Union{SpinWeightedSphericalHarmonicFunction, Nothing}[keep[n] ? spherical[n] : nothing for n in 1:N]
    end
    params = SpectralDecompositionInputParams(s, l, m, c, N)
    return SpinWeightedSpheroidalHarmonicFunction{T}(params, coefficients, spherical_harmonics_l, 1, lambda,
        :spectral, nothing, nothing, basis.precision)
end
