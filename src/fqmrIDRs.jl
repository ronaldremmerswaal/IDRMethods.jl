include("factorized.jl")
include("bandedHessenberg.jl")

type FQMRSolution{T} <: Solution{T}
  x::StridedVector{T}
  ρ
  rho0
  tol

end
FQMRSolution{T}(x::StridedVector{T}, ρ, tol) = FQMRSolution{T}(x, [ρ], ρ, tol)

type FQMRSpace{T} <: IDRSpace{T}
  s

  A
  P

  G::StridedMatrix{T}
  W::StridedMatrix{T}
  v::StridedVector{T}           # last projected orthogonal to R0
  vhat::StridedVector{T}

  latestIdx   # Index in G corresponding to latest g

  orthT

  r::StridedVector{T}
  hes

end
# TODO how many n-vectors do we need? (g, v, vhat)
FQMRSpace{T}(A, P, g::StridedVector{T}, s, rho0, orthT, hes) = FQMRSpace{T}(s, A, P, Matrix{T}(length(g), s + 1), Matrix{T}(length(g), s + 1), g, Vector{T}(length(g)), 1, orthT, zeros(T, s + 3), hes)

type FQMRProjector{T} <: Projector{T}
  n
  s
  j
  μ
  ω
  M
  m::StridedVector{T}
  α
  R0::StridedMatrix{T}
  u::StridedVector{T}
  κ
  orthSearch
  skewT

  latestIdx
  oldestIdx   # Index in G corresponding to oldest column in M
  gToMIdx     # Maps from G idx to M idx

  lu

end
FQMRProjector(n, s, R0, κ, orthSearch, skewT, T) = FQMRProjector{T}(n, s, 0, zero(T), zero(T), [], zeros(T, s), zeros(T, s), R0, zeros(T, s), κ, orthSearch, skewT, 0, 0, [], [])

# Iteratively construct the generalized Hessenberg decomposition of A:
#   A * G * U = G * H,
# and approximately solve A * x = b by minimizing the upperbound for
#   ||b - A * G * U * ϕ|| = ||G (e1 * r0 - H * ϕ)|| <= √(j + 1) * ||e1 * r0 - H * ϕ||
# as described in
#
#     Gijzen, Martin B., Gerard LG Sleijpen, and Jens‐Peter M. Zemke.
#     "Flexible and multi‐shift induced dimension reduction algorithms for solving large sparse linear systems."
#     Numerical Linear Algebra with Applications 22.1 (2015): 1-25.
#
function fqmrIDRs{T}(A, b::StridedVector{T}; s = 8, tol = sqrt(eps(real(T))), maxIt = size(b, 1), x0 = Vector{T}(), P = Identity(), R0 = Matrix{T}(0, 0), orthTol = eps(real(T)), orthSearch = false, kappa = 0.7, orth = :MGS, hesOrth = :HH, skewRepeat = 1, orthRepeat = 3, projDim = s)

  if length(R0) > 0 && size(R0) != (length(b), projDim)
    error("size(R0) != [", length(b), ", $s] (User provided shadow residuals are of incorrect size)")
  end
  if projDim > s
    error("Dimension of projector may not exceed that of the orthogonal basis")
  end

  if length(x0) == 0
    x0 = zeros(b)
    r0 = copy(b)
  else
    r0 = b - A * x0
  end

  orthOne = one(real(T)) / √2

  if orth == :RCGS
    orthT = RepeatedClassicalGS(orthOne, orthTol, orthRepeat)
  elseif orth == :CGS
    orthT = ClassicalGS()
  elseif orth == :MGS
    orthT = ModifiedGS()
  end

  if skewRepeat == 1
    skewT = SingleSkew()
  else
    skewT = RepeatSkew(orthOne, orthTol, skewRepeat)
  end
  rho0 = norm(r0)
  scale!(r0, one(T) / rho0)
  if hesOrth == :HH
    hes = HHBandedHessenberg(s + 1, rho0)
  elseif hesOrth == "Givens"
    hes = GivensBandedHessenberg(s + 1, rho0)
  end
  idrSpace = FQMRSpace(A, P, r0, s, rho0, orthT, hes)
  idrSpace.W[:, 1] = zero(T)
  idrSpace.G[:, 1] = r0
  solution = FQMRSolution(x0, rho0, tol)
  projector = FQMRProjector(size(b, 1), projDim, R0, kappa, orthSearch, skewT, T)

  return IDRMethod(solution, idrSpace, projector, maxIt)

end

# Maps v -> v - G * (R0' * G)^-1 * R0 * v
function apply!{T}(proj::FQMRProjector{T}, idr::FQMRSpace{T})
  if proj.j == 0 && idr.latestIdx <= idr.s
    return
  end

  # Columns of M correspond to G[:, j], where
  #   j = 1 : idr.latestIdx - 1, proj.oldestIdx : idr.s + 1
  # if proj.oldestIdx > idr.latestIdx, and otherwise
  #   j = proj.oldestIdx : idr.latestIdx - 1

  # NB if idr.s == proj.s we can always use
  #   j = 1 : idr.latestIdx - 1, idr.latestIdx + 1 : idr.s + 1

  proj.latestIdx = proj.latestIdx == proj.s ? 1 : proj.latestIdx + 1
  if idr.latestIdx == 1
    skewProject!(idr.v, unsafe_view(idr.G, :, proj.oldestIdx : idr.s + 1), proj.R0, proj.M, proj.α, proj.u, unsafe_view(proj.gToMIdx, proj.oldestIdx : idr.s + 1), proj.m, proj.skewT)
  elseif proj.oldestIdx < idr.latestIdx
    skewProject!(idr.v, unsafe_view(idr.G, :, proj.oldestIdx : idr.latestIdx - 1), proj.R0, proj.M, proj.α, proj.u, unsafe_view(proj.gToMIdx, proj.oldestIdx : idr.latestIdx - 1), proj.m, proj.skewT)
  else
    skewProject!(idr.v, unsafe_view(idr.G, :, proj.oldestIdx : idr.s + 1), unsafe_view(idr.G, :, 1 : idr.latestIdx - 1), proj.R0, proj.M, proj.α, proj.u, unsafe_view(proj.gToMIdx, proj.oldestIdx : idr.s + 1), unsafe_view(proj.gToMIdx, 1 : idr.latestIdx - 1), proj.m, proj.skewT)
  end

  # Update permutations
  proj.gToMIdx[idr.latestIdx] = proj.latestIdx

  proj.oldestIdx = proj.oldestIdx > idr.s ? 1 : proj.oldestIdx + 1

end

function update!{T}(proj::FQMRProjector{T}, idr::FQMRSpace{T})
  if proj.j == 0 && idr.latestIdx == idr.s
    initialize!(proj, idr)
  elseif proj.j > 0
    replaceColumn!(proj.M, proj.latestIdx, proj.m, proj.α)
  end
end

function initialize!{T}(proj::FQMRProjector{T}, idr::FQMRSpace{T})
  if length(proj.R0) == 0
    proj.R0, = qr(rand(proj.n, proj.s))
  end
  M = Matrix{T}(proj.s, proj.s)
  Ac_mul_B!(M, proj.R0, unsafe_view(idr.G, :, idr.s - proj.s + 1 : idr.s))
  proj.M = LUFactorized(M, proj.s - 1)   # NB allow for s - 1 column updates before recomputing lu factorization

  proj.latestIdx = proj.s
  proj.oldestIdx = idr.s - proj.s + 1

  proj.gToMIdx = zeros(Int64, idr.s + 1)
  proj.gToMIdx[idr.s - proj.s + 1 : idr.s] = 1 : proj.s
end

function expand!{T}(idr::FQMRSpace{T}, proj::FQMRProjector{T})
  idr.latestIdx = idr.latestIdx > idr.s ? 1 : idr.latestIdx + 1

  evalPrecon!(idr.vhat, idr.P, idr.v)
  if proj.orthSearch && proj.j == 0
    # First s steps we project orthogonal to R0 by using a flexible preconditioner
    α = zeros(T, proj.s)
    orthogonalize!(idr.vhat, proj.R0, α, idr.orthT)
  end
  A_mul_B!(unsafe_view(idr.G, :, idr.latestIdx), idr.A, idr.vhat)
end

function update!{T}(idr::FQMRSpace{T}, proj::FQMRProjector{T}, k, iter)
  updateG!(idr, k)
  updateHes!(idr, proj, k, iter)
  updateW!(idr, k, iter)
end

function updateG!{T}(idr::FQMRSpace{T}, k)

  idr.r[:] = zero(T)
  if k < idr.s + 1
    idr.r[end] = orthogonalize!(unsafe_view(idr.G, :, idr.latestIdx), unsafe_view(idr.G, :, 1 : k), unsafe_view(idr.r, idr.s + 3 - k : idr.s + 2), idr.orthT)
  else
    idr.r[end] = norm(unsafe_view(idr.G, :, idr.latestIdx))
  end

  scale!(unsafe_view(idr.G, :, idr.latestIdx), one(T) / idr.r[end])
  copy!(idr.v, unsafe_view(idr.G, :, idr.latestIdx))

end

# Updates the QR factorization of H
function updateHes!{T}(idr::FQMRSpace{T}, proj::FQMRProjector{T}, k, iter)

  if proj.j > 0
    # Add contribution due to projector
    axpy!(-proj.μ, proj.u, unsafe_view(idr.r, 2 + idr.s - proj.s : idr.s + 1))
    idr.r[end - 1] += proj.μ
  end

  addColumn!(idr.hes, idr.r)
end

@inline function mapToIDRSpace!{T}(idr::FQMRSpace{T}, proj::FQMRProjector{T})
  if proj.j > 0
    axpy!(-proj.μ, idr.v, unsafe_view(idr.G, :, idr.latestIdx));
  end
end

function updateW!{T}(idr::FQMRSpace{T}, k, iter)
  oneOverR = one(T) / idr.r[end - 1]
  if iter > idr.s
    gemv!('N', -oneOverR, idr.W, idr.r[[idr.s + 2 - k : idr.s + 1; 1 : idr.s + 1 - k]], oneOverR, idr.vhat)
  else
    gemv!('N', -oneOverR, unsafe_view(idr.W, :, 1 : k), unsafe_view(idr.r, idr.s + 2 - k : idr.s + 1), oneOverR, idr.vhat)
  end

  copy!(unsafe_view(idr.W, :, idr.latestIdx), idr.vhat)
end

function update!{T}(sol::FQMRSolution{T}, idr::FQMRSpace{T}, proj::FQMRProjector{T})
  axpy!(idr.hes.ϕ, unsafe_view(idr.W, :, idr.latestIdx), sol.x)
  push!(sol.ρ, abs(idr.hes.φ) * sqrt(proj.j + 1.))
end
