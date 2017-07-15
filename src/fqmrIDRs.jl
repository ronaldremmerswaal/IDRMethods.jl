type FQMRSolution <: Solution
  x
  ρ
  rho0
  tol

  FQMRSolution(x, ρ, tol) = new(x, [ρ], ρ, tol)
end

type FQMRSpace <: IDRSpace
  A
  P

  G
  W
  n
  s
  v           # last projected orthogonal to R0
  vhat

  latestIdx   # Index in G corresponding to latest g

  orthT

  r
  normG
  givensRot
  ϕ
  φ


  # TODO how many n-vectors do we need? (g, v, vhat)
  FQMRSpace(A, P, g, rho0, orthT, n, s, T) = new(A, P, Matrix{T}(n, s + 1), Matrix{T}(n, s + 1), n, s, g, Vector{T}(n), 1, orthT, zeros(T, s + 2), zero(T), Vector{LinAlg.Givens{T}}(s + 2), zero(T), rho0)
end

type FQMRProjector <: Projector
  n
  s
  j
  μ
  ω
  M
  m
  α
  R0
  u
  κ
  orthSearch
  skewT

  latestIdx
  oldestIdx   # Index in G corresponding to oldest column in M
  gToMIdx     # Maps from G idx to M idx

  lu

  FQMRProjector(n, s, R0, κ, orthSearch, skewT, T) = new(n, s, 0, zero(T), zero(T), [], zeros(T, s), zeros(T, s), R0, zeros(T, s), κ, orthSearch, skewT, 0, 0, [], [])
end

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
function fqmrIDRs(A, b; s = 8, tol = sqrt(eps(real(eltype(b)))), maxIt = size(b, 1), x0 = [], P = Identity(), R0 = [], orthTol = eps(real(eltype(b))), orthSearch = false, kappa = 0.7, orth = "MGS", skewRepeat = 1, orthRepeat = 3, projDim = s)

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

  orthOne = one(real(eltype(b))) / √2

  if orth == "RCGS"
    orthT = RepeatedClassicalGS(orthOne, orthTol, orthRepeat)
  elseif orth == "CGS"
    orthT = ClassicalGS()
  elseif orth == "MGS"
    orthT = ModifiedGS()
  end

  if skewRepeat == 1
    skewT = SingleSkew()
  else
    skewT = RepeatSkew(orthOne, orthTol, skewRepeat)
  end
  rho0 = vecnorm(r0)
  scale!(r0, 1.0 / rho0)
  idrSpace = FQMRSpace(A, P, r0, rho0, orthT, size(b, 1), s, eltype(b))
  idrSpace.W[:, 1] = 0.0
  idrSpace.G[:, 1] = r0
  solution = FQMRSolution(x0, rho0, tol)
  projector = FQMRProjector(size(b, 1), projDim, R0, kappa, orthSearch, skewT, eltype(b))

  return IDRMethod(solution, idrSpace, projector, maxIt)

end

# Maps v -> v - G * (R0' * G)^-1 * R0 * v
function apply!(proj::FQMRProjector, idr::FQMRSpace)
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
  proj.gToMIdx[proj.oldestIdx] = 0  # NB Only to find bugs easier...

  proj.oldestIdx = proj.oldestIdx > idr.s ? 1 : proj.oldestIdx + 1

end

function update!(proj::FQMRProjector, idr::FQMRSpace)
  if proj.j == 0 && idr.latestIdx == idr.s
    initialize!(proj, idr)
  elseif proj.j > 0
    replaceColumn!(proj.M, proj.latestIdx, proj.m, proj.α)
  end
end

function initialize!(proj::FQMRProjector, idr::FQMRSpace)
  if length(proj.R0) == 0
    proj.R0 = rand(proj.n, proj.s)
    proj.R0, = qr(proj.R0)
  end
  proj.M = Matrix{eltype(idr.v)}(proj.s, proj.s)
  Ac_mul_B!(proj.M, proj.R0, unsafe_view(idr.G, :, idr.s - proj.s + 1 : idr.s))
  proj.M = Factorized(proj.M, proj.s - 1)   # NB allow for s - 1 column updates before recomputing lu factorization

  proj.latestIdx = proj.s
  proj.oldestIdx = idr.s - proj.s + 1

  proj.gToMIdx = zeros(Int64, idr.s + 1)
  proj.gToMIdx[idr.s - proj.s + 1 : idr.s] = 1 : proj.s
end

function expand!(idr::FQMRSpace, proj::FQMRProjector)
  idr.latestIdx = idr.latestIdx > idr.s ? 1 : idr.latestIdx + 1

  evalPrecon!(idr.vhat, idr.P, idr.v)
  if proj.orthSearch && proj.j == 0
    # First s steps we project orthogonal to R0 by using a flexible preconditioner
    α = zeros(eltype(idr.vhat), proj.s)
    orthogonalize!(idr.vhat, proj.R0, α, idr.orthT)
  end
  A_mul_B!(unsafe_view(idr.G, :, idr.latestIdx), idr.A, idr.vhat)
end

function update!(idr::FQMRSpace, proj::FQMRProjector, k, iter)
  updateG!(idr, k)
  updateHes!(idr, proj, k, iter)
  updateW!(idr, k, iter)
end

function updateG!(idr::FQMRSpace, k)

  idr.r[:] = zero(eltype(idr.v))
  aIdx = idr.latestIdx
  if k < idr.s + 1
    idr.normG = orthogonalize!(unsafe_view(idr.G, :, aIdx), unsafe_view(idr.G, :, 1 : k), unsafe_view(idr.r, idr.s + 3 - k : idr.s + 2), idr.orthT)
  else
    idr.normG = vecnorm(unsafe_view(idr.G, :, aIdx))
  end

  scale!(unsafe_view(idr.G, :, aIdx), one(eltype(idr.v)) / idr.normG)
  copy!(idr.v, unsafe_view(idr.G, :, aIdx))

end

# Updates the QR factorization of H
function updateHes!(idr::FQMRSpace, proj::FQMRProjector, k, iter)
  idr.givensRot[1 : end - 1] = unsafe_view(idr.givensRot, 2 : idr.s + 2)

  if proj.j > 0
    axpy!(-proj.μ, proj.u, unsafe_view(idr.r, 2 + idr.s - proj.s : idr.s + 1))
    idr.r[end] += proj.μ
  end

  startIdx = max(1, idr.s + 3 - iter)
  for l = startIdx : idr.s + 1
    idr.r[l : l + 1] = idr.givensRot[l] * unsafe_view(idr.r, l : l + 1)
  end

  idr.givensRot[end], idr.r[end] = givens(idr.r[end], idr.normG, 1, 2)

  idr.ϕ = idr.givensRot[end].c * idr.φ
  idr.φ = -conj(idr.givensRot[end].s) * idr.φ
end

@inline function mapToIDRSpace!(idr::FQMRSpace, proj::FQMRProjector)
  if proj.j > 0
    axpy!(-proj.μ, idr.v, unsafe_view(idr.G, :, idr.latestIdx));
  end
end

function updateW!(idr::FQMRSpace, k, iter)
  oneOverR = one(eltype(idr.W)) / idr.r[end]
  if iter > idr.s
    gemv!('N', -oneOverR, idr.W, idr.r[[idr.s + 2 - k : idr.s + 1; 1 : idr.s + 1 - k]], oneOverR, idr.vhat)
  else
    gemv!('N', -oneOverR, unsafe_view(idr.W, :, 1 : k), unsafe_view(idr.r, idr.s + 2 - k : idr.s + 1), oneOverR, idr.vhat)
  end

  copy!(unsafe_view(idr.W, :, idr.latestIdx), idr.vhat)
end

function update!(sol::FQMRSolution, idr::FQMRSpace, proj::FQMRProjector)
  axpy!(idr.ϕ, unsafe_view(idr.W, :, idr.latestIdx), sol.x)
  push!(sol.ρ, abs(idr.φ) * sqrt(proj.j + 1.))
  @show idr.givensRot[end]
  @show sol.ρ[end] - vecnorm(idr.A * sol.x - ones(sol.x))
end
