module IDRMethods

export fqmrIDRs

using Base.BLAS
using Base.LinAlg

include("harmenView.jl")

type Identity end
type Preconditioner end

abstract type OrthType end
type ClassicalGS <: OrthType
end
type RepeatedClassicalGS <: OrthType
  one
  tol
  maxRepeat
end
type ModifiedGS <: OrthType
end

abstract type SkewType end
type RepeatSkew <: SkewType
  one
  tol
  maxRepeat
end
type SingleSkew <: SkewType
end

type Projector
  j
  μ
  M
  R0
  u
  m
  κ
  orthSearch
  skewT

  Projector(n, s, R0, κ, orthSearch, skewT, T) = new(0, zero(T), zeros(T, s, s), R0, zeros(T, s), Vector{T}(s), κ, orthSearch, skewT)
end

type Hessenberg
  n
  s
  r
  cosine
  sine
  ϕ
  φ

  Hessenberg(n, s, T, rho0) = new(n, s, zeros(T, s + 3), zeros(T, s + 2), zeros(T, s + 2), zero(T), rho0)
end

type Arnoldi
  A
  P

  permG
  G
  W
  n
  s
  v           # last projected orthogonal to R0
  vhat

  α
  lastIdx

  orthT

  # TODO how many n-vectors do we need? (g, v, vhat)
  Arnoldi(A, P, g, orthT, n, s, T) = new(A, P, [1 : s...], Matrix{T}(n, s + 1), Matrix{T}(n, s + 1), n, s, g, Vector{T}(n), Vector{T}(s), 1, orthT)
end

type Solution
  x
  ρ
  rho0
  tol

  Solution(x, ρ, tol) = new(x, [ρ], ρ, tol)
end


function fqmrIDRs(A, b; s = 8, tol = sqrt(eps(real(eltype(b)))), maxIt = size(b, 1), x0 = [], P = Identity(), R0 = [], orthTol = eps(real(eltype(b))), orthSearch = false, kappa = 0.7, orth = "MGS", skewRepeat = 1, orthRepeat = 3)

  iter = 0
  solution, arnoldi, hessenberg, projector = initMethod(A, b, s, tol, maxIt, x0, P, R0, orthTol, orthSearch, kappa, orth, skewRepeat, orthRepeat)
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
  while true
    for k in 1 : s + 1
      iter += 1

      if iter == s + 1
        initialize!(projector, arnoldi)
      end

      if iter > s
        # Compute u, v: the skew-projection of g along G orthogonal to R0 (for which v = (I - G * inv(M) * R0) * g)
        apply!(projector, arnoldi, k)
      end

      cycle!(arnoldi)
      cycle!(hessenberg)

      # Compute g = A * v
      expand!(arnoldi, projector, k)

      if k == s + 1
        nextIDRSpace!(projector, arnoldi)
      end
      # Compute t = (A - μ * I) * g
      mapToIDRSpace!(arnoldi, projector, k)

      # Compute g = t - G * α, such that g orthogonal w.r.t. G(:, 1 : k)
      updateG!(arnoldi, hessenberg, k)

      # Compute the new column r of R, and element of Q(1, :)  (for which H = Q * R)
      update!(hessenberg, projector, iter)

      # Compute w = (P \ v - W * r) / r(end) (for which W * R = G * U)
      updateW!(arnoldi, hessenberg, k, iter)

      # Update x <- x + Q(1, end) * w
      update!(solution, arnoldi, hessenberg, projector, k)
      if isConverged(solution) || iter == maxIt
        return solution.x, solution.ρ
      end
    end
  end

end

function initMethod(A, b, s, tol, maxIt, x0, P, R0, orthTol, orthSearch, kappa, orth, skewRepeat, orthRepeat)
  if length(R0) > 0 && size(R0) != (length(b), s)
    error("size(R0) != [", length(b), ", $s] (User provided shadow residuals are of incorrect size)")
  end

  if length(x0) == 0
    x0 = zeros(b)
    r0 = b
  else
    r0 = b - A * x
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
  hessenberg = Hessenberg(size(b, 1), s, eltype(b), rho0)
  arnoldi = Arnoldi(A, P, r0 / rho0, orthT, size(b, 1), s, eltype(b))
  arnoldi.W[:, 1] = 0.
  arnoldi.G[:, 1] = r0 / rho0
  solution = Solution(x0, rho0, tol)
  projector = Projector(size(b, 1), s, R0, kappa, orthSearch, skewT, eltype(b))

  return solution, arnoldi, hessenberg, projector
end

# Maps v -> v - G * (R0' * G)^-1 * R0 * v
function apply!(proj::Projector, arnold::Arnoldi, k)
  lu = lufact(proj.M)

  skewProject!(arnold.v, unsafe_view(arnold.G, :, 1 : arnold.lastIdx - 1), unsafe_view(arnold.G, :, arnold.lastIdx + 1 : arnold.s + 1), proj.R0, lu, proj.u, arnold.s - k + 2 : arnold.s, 1 : arnold.s - k + 1, arnold.permG, proj.m, proj.skewT)

  proj.M[:, arnold.permG[1]] = proj.m

end

function skewProject!(v, G1, G2, R0, lu, u, idx1, idx2, perm, m, skewT::RepeatSkew)
  Ac_mul_B!(m, R0, v)
  A_ldiv_B!(u, lu, m)
  u[:] = u[perm]

  gemv!('N', -1.0, G1, unsafe_view(u, idx1), 1.0, v)
  gemv!('N', -1.0, G2, unsafe_view(u, idx2), 1.0, v)

  happy = vecnorm(v) < skewT.one * vecnorm(u)

  if happy return end

  mUpdate = zeros(m)
  uUpdate = zeros(u)
  for idx = 2 : skewT.maxRepeat
    # Repeat projection
    Ac_mul_B!(mUpdate, R0, v)
    A_ldiv_B!(uUpdate, lu, mUpdate)
    uUpdate[:] = uUpdate[perm]

    gemv!('N', -1.0, G1, unsafe_view(uUpdate, idx1), 1.0, v)
    gemv!('N', -1.0, G2, unsafe_view(uUpdate, idx2), 1.0, v)

    axpy!(1.0, mUpdate, m)
    axpy!(1.0, uUpdate, u)

    happy = vecnorm(v) > skewT.one * vecnorm(uUpdate)
    if happy break end
  end
end

function skewProject!(v, G1, G2, R0, lu, u, idx1, idx2, perm, m, skewT::SingleSkew)
  Ac_mul_B!(m, R0, v)
  A_ldiv_B!(u, lu, m)
  u[:] = u[perm]

  gemv!('N', -1.0, G1, unsafe_view(u, idx1), 1.0, v)
  gemv!('N', -1.0, G2, unsafe_view(u, idx2), 1.0, v)
end

function initialize!(proj::Projector, arnold::Arnoldi)
  if length(proj.R0) == 0
    # NB if user provided R0, then we assume it is orthogonalized already!
    proj.R0 = rand(arnold.n, arnold.s)
    proj.R0, = qr(proj.R0)
  end
  Ac_mul_B!(proj.M, proj.R0, unsafe_view(arnold.G, :, 1 : arnold.s))
end

function nextIDRSpace!(proj::Projector, arnold::Arnoldi)
  proj.j += 1

  # Compute residual minimizing μ
  ν = vecdot(unsafe_view(arnold.G, :, arnold.lastIdx), arnold.v)
  τ = vecdot(unsafe_view(arnold.G, :, arnold.lastIdx), unsafe_view(arnold.G, :, arnold.lastIdx))

  ω = ν / τ
  η = ν / (sqrt(τ) * norm(arnold.v))
  if abs(η) < proj.κ
    ω *= proj.κ / abs(η)
  end
  proj.μ = abs(ω) > eps() ? 1. / ω : 1.
end

function cycle!(hes::Hessenberg)
  hes.cosine[1 : end - 1] = unsafe_view(hes.cosine, 2 : hes.s + 2)
  hes.sine[1 : end - 1] = unsafe_view(hes.sine, 2 : hes.s + 2)
end

# Updates the QR factorization of H
function update!(hes::Hessenberg, proj::Projector, iter)
  axpy!(-proj.μ, proj.u, unsafe_view(hes.r, 2 : hes.s + 1))
  hes.r[end - 1] += proj.μ

  startIdx = max(1, hes.s + 3 - iter)
  applyGivens!(unsafe_view(hes.r, startIdx : hes.s + 2), unsafe_view(hes.sine, startIdx : hes.s + 1), unsafe_view(hes.cosine, startIdx : hes.s + 1))

  updateGivens!(hes.r, hes.sine, hes.cosine)

  hes.ϕ = hes.cosine[end] * hes.φ
  hes.φ = -conj(hes.sine[end]) * hes.φ
end

function applyGivens!(r, sine, cosine)
  for l = 1 : length(r) - 1
    oldRl = r[l]
    r[l] = cosine[l] * oldRl + sine[l] * r[l + 1]
    r[l + 1] = -conj(sine[l]) * oldRl + cosine[l] * r[l + 1]
  end
end

function updateGivens!(r, sine, cosine)
  α = r[end - 1]
  β = r[end]
  if abs(α) < eps()
    sine[end] = 1.
    cosine[end] = 0.
    r[end - 1] = β
  else
    t = abs(α) + abs(β)
    ρ = t * sqrt(abs(α / t) ^ 2 + abs(β / t) ^ 2)
    Θ = α / abs(α)
    sine[end] = Θ * conj(β) / ρ

    cosine[end] = abs(α) / ρ
    r[end - 1] = Θ * ρ
  end
end

function cycle!(arnold::Arnoldi)
  pGEnd = arnold.permG[1]
  arnold.permG[1 : end - 1] = unsafe_view(arnold.permG, 2 : arnold.s)
  arnold.permG[end] = pGEnd
end

@inline evalPrecon!(vhat, P::Identity, v) = copy!(vhat, v)
@inline function evalPrecon!(vhat, P::Preconditioner, v)
  A_ldiv_B!(vhat, P, v)
end
@inline function evalPrecon!(vhat, P::Function, v)
  P(vhat, v)
end

function expand!(arnold::Arnoldi, proj::Projector, k)
  arnold.lastIdx = k > arnold.s ? 1 : k + 1
  evalPrecon!(arnold.vhat, arnold.P, arnold.v)
  if proj.orthSearch && proj.j == 0
    # First s steps we project orthogonal to R0 by using a flexible preconditioner
    orthogonalize!(arnold.vhat, proj.R0, arnold.α, arnold.orthT)
  end
  A_mul_B!(unsafe_view(arnold.G, :, arnold.lastIdx), arnold.A, arnold.vhat)
end

function updateW!(arnold::Arnoldi, hes::Hessenberg, k, iter)
  if iter > arnold.s
    gemv!('N', -1.0, arnold.W, hes.r[[arnold.s + 2 - k : arnold.s + 1; 1 : arnold.s + 1 - k]], 1.0, arnold.vhat)
  else
    gemv!('N', -1.0, unsafe_view(arnold.W, :, 1 : k), unsafe_view(hes.r, arnold.s + 2 - k : arnold.s + 1), 1.0, arnold.vhat)
  end

  copy!(unsafe_view(arnold.W, :, arnold.lastIdx), arnold.vhat)
  scale!(unsafe_view(arnold.W, :, arnold.lastIdx), 1 / hes.r[end - 1])
end

function updateG!(arnold::Arnoldi, hes::Hessenberg, k)

  hes.r[:] = 0.
  aIdx = arnold.lastIdx

  if k < arnold.s + 1
    hes.r[end] = orthogonalize!(unsafe_view(arnold.G, :, aIdx), unsafe_view(arnold.G, :, 1 : k), unsafe_view(hes.r, arnold.s + 3 - k : arnold.s + 2), arnold.orthT)
  else
    hes.r[end] = vecnorm(unsafe_view(arnold.G, :, aIdx))
  end

  scale!(unsafe_view(arnold.G, :, aIdx), 1 / hes.r[end])
  copy!(arnold.v, unsafe_view(arnold.G, :, aIdx))

end

function orthogonalize!(g, G, h, orthT::ClassicalGS)
  Ac_mul_B!(h, G, g)
  gemv!('N', -1.0, G, h, 1.0, g)

  return vecnorm(g)
end

# Orthogonalize g w.r.t. G, and store coeffs in h (NB g is not normalized)
function orthogonalize!(g, G, h, orthT::RepeatedClassicalGS)
  Ac_mul_B!(h, G, g)
  # println(0, ", normG = ", vecnorm(g), ", normH = ", vecnorm(h))
  gemv!('N', -1.0, G, h, 1.0, g)

  normG = vecnorm(g)
  normH = vecnorm(h)

  happy = normG < orthT.one * normH || normH < orthT.tol * normG
  # println(1, ", normG = ", normG, ", normH = ", vecnorm(G' * g))
  if happy return normG end

  for idx = 2 : orthT.maxRepeat
    updateH = Vector(h)

    Ac_mul_B!(updateH, G, g)
    gemv!('N', -1.0, G, updateH, 1.0, g)

    axpy!(1.0, updateH, h)

    normG = vecnorm(g)
    normH = vecnorm(updateH)
    # println(idx, ", normG = ", normG, ", normH = ", normH)
    happy = normG < orthT.one * normH || normH < orthT.tol * normG
    if happy break end
  end

  return normG
end

function orthogonalize!(g, G, h, orthT::ModifiedGS)
  for l in 1 : length(h)
    h[l] = vecdot(unsafe_view(G, :, l), g)
    axpy!(-h[l], unsafe_view(G, :, l), g)
  end
  return vecnorm(g)
end

@inline function mapToIDRSpace!(arnold::Arnoldi, proj::Projector, k)
  if proj.j > 0
    axpy!(-proj.μ, arnold.v, unsafe_view(arnold.G, :, arnold.lastIdx));
  end
end

@inline function isConverged(sol::Solution)
  return sol.ρ[end] < sol.tol * sol.rho0
end

function update!(sol::Solution, arnold::Arnoldi, hes::Hessenberg, proj::Projector, k)
  axpy!(hes.ϕ, unsafe_view(arnold.W, :, arnold.lastIdx), sol.x)
  push!(sol.ρ, abs(hes.φ) * sqrt(proj.j + 1.))
end

end
