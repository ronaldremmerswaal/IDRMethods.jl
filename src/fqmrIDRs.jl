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

  # TODO how many n-vectors do we need? (g, v, vhat)
  FQMRSpace(A, P, g, orthT, n, s, T) = new(A, P, Matrix{T}(n, s + 1), Matrix{T}(n, s + 1), n, s, g, Vector{T}(n), 1, orthT)
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
  hessenberg = Hessenberg(size(b, 1), s, eltype(b), rho0)
  idrSpace = FQMRSpace(A, P, r0, orthT, size(b, 1), s, eltype(b))
  idrSpace.W[:, 1] = 0.0
  idrSpace.G[:, 1] = r0
  solution = Solution(x0, rho0, tol)
  projector = Projector(size(b, 1), projDim, R0, kappa, orthSearch, skewT, eltype(b))

  return IDRMethod(solution, idrSpace, hessenberg, projector, maxIt)

end

# Maps v -> v - G * (R0' * G)^-1 * R0 * v
function apply!(proj::Projector, idrSpace::FQMRSpace)

  # Columns of M correspond to G[:, j], where
  #   j = 1 : idrSpace.latestIdx - 1, proj.oldestIdx : idrSpace.s + 1
  # if proj.oldestIdx > idrSpace.latestIdx, and otherwise
  #   j = proj.oldestIdx : idrSpace.latestIdx - 1

  # NB if idrSpace.s == proj.s we can always use
  #   j = 1 : idrSpace.latestIdx - 1, idrSpace.latestIdx + 1 : idrSpace.s + 1

  proj.latestIdx = proj.latestIdx == proj.s ? 1 : proj.latestIdx + 1
  if idrSpace.latestIdx == 1
    skewProject!(idrSpace.v, unsafe_view(idrSpace.G, :, proj.oldestIdx : idrSpace.s + 1), proj.R0, proj.M, proj.α, proj.u, unsafe_view(proj.gToMIdx, proj.oldestIdx : idrSpace.s + 1), proj.m, proj.skewT)
  elseif proj.oldestIdx < idrSpace.latestIdx
    skewProject!(idrSpace.v, unsafe_view(idrSpace.G, :, proj.oldestIdx : idrSpace.latestIdx - 1), proj.R0, proj.M, proj.α, proj.u, unsafe_view(proj.gToMIdx, proj.oldestIdx : idrSpace.latestIdx - 1), proj.m, proj.skewT)
  else
    skewProject!(idrSpace.v, unsafe_view(idrSpace.G, :, proj.oldestIdx : idrSpace.s + 1), unsafe_view(idrSpace.G, :, 1 : idrSpace.latestIdx - 1), proj.R0, proj.M, proj.α, proj.u, unsafe_view(proj.gToMIdx, proj.oldestIdx : idrSpace.s + 1), unsafe_view(proj.gToMIdx, 1 : idrSpace.latestIdx - 1), proj.m, proj.skewT)
  end

  # Update permutations
  proj.gToMIdx[idrSpace.latestIdx] = proj.latestIdx
  proj.gToMIdx[proj.oldestIdx] = 0  # NB Only to find bugs easier...

  proj.oldestIdx = proj.oldestIdx > idrSpace.s ? 1 : proj.oldestIdx + 1

end

function update!(proj::Projector, idrSpace::FQMRSpace)
  if proj.j == 0
    initialize!(proj, idrSpace)
  else
    # proj.M[:, proj.latestIdx] = proj.m
    replaceColumn!(proj.M, proj.latestIdx, proj.m, proj.α)
  end

  # proj.lu = lufact(proj.M)
end

function initialize!(proj::Projector, idrSpace::FQMRSpace)
  if length(proj.R0) == 0
    proj.R0 = rand(proj.n, proj.s)
    proj.R0, = qr(proj.R0)
  end
  proj.M = Matrix{eltype(idrSpace.v)}(proj.s, proj.s)
  Ac_mul_B!(proj.M, proj.R0, unsafe_view(idrSpace.G, :, idrSpace.s - proj.s + 1 : idrSpace.s))
  proj.M = Factorized(proj.M, proj.s - 1)   # NB allow for s - 1 column updates before recomputing lu factorization

  proj.latestIdx = proj.s
  proj.oldestIdx = idrSpace.s - proj.s + 1

  proj.gToMIdx = zeros(Int64, idrSpace.s + 1)
  proj.gToMIdx[idrSpace.s - proj.s + 1 : idrSpace.s] = 1 : proj.s
end

function nextIDRSpace!(proj::Projector, idrSpace::FQMRSpace)
  proj.j += 1

  # Compute residual minimizing μ
  ν = vecdot(unsafe_view(idrSpace.G, :, idrSpace.latestIdx), idrSpace.v)
  τ = vecdot(unsafe_view(idrSpace.G, :, idrSpace.latestIdx), unsafe_view(idrSpace.G, :, idrSpace.latestIdx))

  ω = ν / τ
  η = ν / (sqrt(τ) * norm(idrSpace.v))
  if abs(η) < proj.κ
    ω *= proj.κ / abs(η)
  end
  # TODO condest(A)? instead of 1.
  proj.μ = abs(ω) > eps(real(eltype(idrSpace.v))) ? 1. / ω : 1.

end

function cycle!(hes::Hessenberg)
  hes.cosine[1 : end - 1] = unsafe_view(hes.cosine, 2 : hes.s + 2)
  hes.sine[1 : end - 1] = unsafe_view(hes.sine, 2 : hes.s + 2)
end

# Updates the QR factorization of H
function update!(hes::Hessenberg, proj::Projector, iter)
  cycle!(hes)

  axpy!(-proj.μ, proj.u, unsafe_view(hes.r, 2 + hes.s - proj.s : hes.s + 1))
  hes.r[end - 1] += proj.μ

  startIdx = max(1, hes.s + 3 - iter)
  applyGivens!(unsafe_view(hes.r, startIdx : hes.s + 2), unsafe_view(hes.sine, startIdx : hes.s + 1), unsafe_view(hes.cosine, startIdx : hes.s + 1))

  updateGivens!(hes.r, hes.sine, hes.cosine)

  hes.ϕ = hes.cosine[end] * hes.φ
  hes.φ = -conj(hes.sine[end]) * hes.φ
end

function expand!(idrSpace::FQMRSpace, proj::Projector)
  idrSpace.latestIdx = idrSpace.latestIdx > idrSpace.s ? 1 : idrSpace.latestIdx + 1
  evalPrecon!(idrSpace.vhat, idrSpace.P, idrSpace.v)
  if proj.orthSearch && proj.j == 0
    # First s steps we project orthogonal to R0 by using a flexible preconditioner
    α = zeros(eltype(idrSpace.vhat), proj.s)
    orthogonalize!(idrSpace.vhat, proj.R0, α, idrSpace.orthT)
  end
  A_mul_B!(unsafe_view(idrSpace.G, :, idrSpace.latestIdx), idrSpace.A, idrSpace.vhat)
end

function updateW!(idrSpace::FQMRSpace, hes::Hessenberg, k, iter)
  if iter > idrSpace.s
    gemv!('N', -1.0, idrSpace.W, hes.r[[idrSpace.s + 2 - k : idrSpace.s + 1; 1 : idrSpace.s + 1 - k]], 1.0, idrSpace.vhat)
  else
    gemv!('N', -1.0, unsafe_view(idrSpace.W, :, 1 : k), unsafe_view(hes.r, idrSpace.s + 2 - k : idrSpace.s + 1), 1.0, idrSpace.vhat)
  end

  copy!(unsafe_view(idrSpace.W, :, idrSpace.latestIdx), idrSpace.vhat)
  scale!(unsafe_view(idrSpace.W, :, idrSpace.latestIdx), 1 / hes.r[end - 1])
end

function updateG!(idrSpace::FQMRSpace, hes::Hessenberg, k)

  hes.r[:] = 0.
  aIdx = idrSpace.latestIdx

  if k < idrSpace.s + 1
    hes.r[end] = orthogonalize!(unsafe_view(idrSpace.G, :, aIdx), unsafe_view(idrSpace.G, :, 1 : k), unsafe_view(hes.r, idrSpace.s + 3 - k : idrSpace.s + 2), idrSpace.orthT)
  else
    hes.r[end] = vecnorm(unsafe_view(idrSpace.G, :, aIdx))
  end

  scale!(unsafe_view(idrSpace.G, :, aIdx), 1 / hes.r[end])
  copy!(idrSpace.v, unsafe_view(idrSpace.G, :, aIdx))

end


@inline function mapToIDRSpace!(idrSpace::FQMRSpace, proj::Projector)
  if proj.j > 0
    axpy!(-proj.μ, idrSpace.v, unsafe_view(idrSpace.G, :, idrSpace.latestIdx));
  end
end


function update!(sol::Solution, idrSpace::FQMRSpace, hes::Hessenberg, proj::Projector, k)
  axpy!(hes.ϕ, unsafe_view(idrSpace.W, :, idrSpace.latestIdx), sol.x)
  push!(sol.ρ, abs(hes.φ) * sqrt(proj.j + 1.))
end
