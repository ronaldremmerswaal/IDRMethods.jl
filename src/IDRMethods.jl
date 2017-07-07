module IDRMethods

export fqmrIDRs

using Base.BLAS
using Base.LinAlg

include("harmenView.jl")

type Identity
end
type Preconditioner
end

type Projector
  j
  mu
  M
  R0
  u
  m

  Projector(n, s, R0, T) = new(0, zero(T), zeros(T, s, s), R0, zeros(T, s), Vector{T}(s))
end

type Hessenberg
  n
  s
  r
  cosine
  sine
  phi
  phihat

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

  alpha
  lastIdx

  # TODO how many n-vectors do we need? (g, v, vhat)
  Arnoldi(A, P, g, n, s, T) = new(A, P, [1 : s...], Matrix{T}(n, s + 1), Matrix{T}(n, s + 1), n, s, g, Vector{T}(n), Vector{T}(s), 1)
end

type Solution
  x
  rho
  rho0
  tol

  Solution(x, rho, tol) = new(x, rho, rho, tol)
end


function fqmrIDRs(A, b; s = 8, tol = sqrt(eps(real(eltype(b)))), maxIt = size(b, 1), x0 = [], P = Identity(), R0 = [])

  if length(x0) == 0
    x0 = zeros(b)
    r0 = b
  else
    r0 = b - A * x0
  end
  rho0 = vecnorm(r0)
  hessenberg = Hessenberg(size(b, 1), s, eltype(b), rho0)
  arnoldi = Arnoldi(A, P, r0 / rho0, size(b, 1), s, eltype(b))
  arnoldi.W[:, 1] = 0.          # TODO put inside arnoldi constructor
  arnoldi.G[:, 1] = r0 / rho0   # TODO put inside arnoldi constructor
  solution = Solution(x0, rho0, tol)
  projector = Projector(size(b, 1), s, R0, eltype(b))

  iter = 0
  stopped = false

  while !stopped
    for k in 1 : s + 1
      iter += 1

      if iter == s + 1
        initialize!(projector, arnoldi)
      end

      if iter > s
        apply!(projector, arnoldi, k)
      end

      cycle!(arnoldi)
      cycle!(hessenberg)

      expand!(arnoldi, k)

      if k == s + 1
        nextIDRSpace!(projector, arnoldi)
      end
      mapToIDRSpace!(arnoldi, projector, k)

      updateG!(arnoldi, hessenberg, k)
      update!(hessenberg, projector, iter)
      updateW!(arnoldi, hessenberg, k, iter)

      update!(solution, arnoldi, hessenberg, projector, k)
      if isConverged(solution) || iter == maxIt
        stopped = true
        break
      end
    end
  end

  return solution.x, solution.rho
end

# Maps v -> v - G * (R0' * G)^-1 * R0 * v
function apply!(proj::Projector, arnold::Arnoldi, k)
  gemv!('C', 1.0, proj.R0, arnold.v, 0.0, proj.m)
  lu = lufact(proj.M)
  A_ldiv_B!(proj.u, lu, proj.m)
  proj.u[:] = proj.u[arnold.permG]
  if arnold.lastIdx > 1
    gemv!('N', -1.0, unsafe_view(arnold.G, :, 1 : arnold.lastIdx - 1), unsafe_view(proj.u, arnold.s - k + 2 : arnold.s), 1.0, arnold.v)
  end
  if arnold.lastIdx <= arnold.s
    gemv!('N', -1.0, unsafe_view(arnold.G, :, arnold.lastIdx + 1 : arnold.s + 1), unsafe_view(proj.u, 1 : arnold.s - k + 1), 1.0, arnold.v)
  end
  proj.M[:, arnold.permG[1]] = proj.m
end


@inline function initialize!(proj::Projector, arnold::Arnoldi)
  if length(proj.R0) == 0
    # NB if user provided R0, then we assume it is orthogonalized already!
    proj.R0 = rand(arnold.n, arnold.s)
    qrfact!(proj.R0)
  end
  gemm!('C', 'N', 1.0, proj.R0, unsafe_view(arnold.G, :, 1 : arnold.s), 1.0, proj.M)
end

function nextIDRSpace!(proj::Projector, arnold::Arnoldi)
  proj.j += 1

  # Compute residual minimizing mu
  tv = vecdot(unsafe_view(arnold.G, :, arnold.lastIdx), arnold.v)
  tt = vecdot(unsafe_view(arnold.G, :, arnold.lastIdx), unsafe_view(arnold.G, :, arnold.lastIdx))

  omega = tv / tt
  rho = tv / (sqrt(tt) * norm(arnold.v))
  if abs(rho) < 0.7
    omega *= 0.7 / abs(rho)
  end
  proj.mu = abs(omega) > eps() ? 1. / omega : 1.
end

@inline function cycle!(hes::Hessenberg)
  hes.cosine[1 : end - 1] = unsafe_view(hes.cosine, 2 : hes.s + 2)
  hes.sine[1 : end - 1] = unsafe_view(hes.sine, 2 : hes.s + 2)
end

# Updates the QR factorization of H
function update!(hes::Hessenberg, proj::Projector, iter)
  axpy!(-proj.mu, proj.u, unsafe_view(hes.r, 2 : hes.s + 1))
  hes.r[end - 1] += proj.mu

  startIdx = max(1, hes.s + 3 - iter)
  applyGivens!(unsafe_view(hes.r, startIdx : hes.s + 2), unsafe_view(hes.sine, startIdx : hes.s + 1), unsafe_view(hes.cosine, startIdx : hes.s + 1))

  updateGivens!(hes.r, hes.sine, hes.cosine)

  hes.phi = hes.cosine[end] * hes.phihat
  hes.phihat = -conj(hes.sine[end]) * hes.phihat
end

@inline function applyGivens!(r, sine, cosine)
  for l = 1 : length(r) - 1
    oldRl = r[l]
    r[l] = cosine[l] * oldRl + sine[l] * r[l + 1]
    r[l + 1] = -conj(sine[l]) * oldRl + cosine[l] * r[l + 1]
  end
end

function updateGivens!(r, sine, cosine)
  a = r[end - 1]
  b = r[end]
  if abs(a) < eps()
    sine[end] = 1.
    cosine[end] = 0.
    r[end - 1] = b
  else
    t = abs(a) + abs(b)
    rho = t * sqrt(abs(a / t) ^ 2 + abs(b / t) ^ 2)
    alpha = a / abs(a)

    sine[end] = alpha * conj(b) / rho
    cosine[end] = abs(a) / rho
    r[end - 1] = alpha * rho
  end
end

@inline function cycle!(arnold::Arnoldi)
  pGEnd = arnold.permG[1]
  arnold.permG[1 : end - 1] = unsafe_view(arnold.permG, 2 : arnold.s)
  arnold.permG[end] = pGEnd
end

@inline evalPrecon!(vhat, P::Identity, v) = copy!(vhat, v)
@inline function evalPrecon!(vhat, P::Preconditioner, v)
  A_ldiv_B!(vhat, P, v)
end

@inline function expand!(arnold::Arnoldi, k)
  arnold.lastIdx = k > arnold.s ? 1 : k + 1
  evalPrecon!(arnold.vhat, arnold.P, arnold.v)
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
    orthogonalize!(unsafe_view(arnold.G, :, aIdx), unsafe_view(arnold.G, :, 1 : k), unsafe_view(arnold.alpha, 1 : k))

    hes.r[arnold.s + 3 - k : arnold.s + 2] = unsafe_view(arnold.alpha, 1 : k)
  end

  hes.r[end] = vecnorm(unsafe_view(arnold.G, :, aIdx))
  scale!(unsafe_view(arnold.G, :, aIdx), 1 / hes.r[end])
  copy!(arnold.v, unsafe_view(arnold.G, :, aIdx))

end

# Orthogonalize g w.r.t. G, and store coeffs in h (NB g is not normalized)
function orthogonalize!(g, G, h)
  # for l in 1 : length(h)
  #   h[l] = vecdot(unsafe_view(G, :, l), g)
  #   axpy!(-h[l], unsafe_view(G, :, l), g)
  # end
  gemv!('C', 1.0, G, g, 0.0, h)
  gemv!('N', -1.0, G, h, 1.0, g)
end

@inline function mapToIDRSpace!(arnold::Arnoldi, proj::Projector, k)
  if proj.j > 0
    axpy!(-proj.mu, arnold.v, unsafe_view(arnold.G, :, arnold.lastIdx));
  end
end

@inline function isConverged(sol::Solution)
  return sol.rho < sol.tol * sol.rho0
end

function update!(sol::Solution, arnold::Arnoldi, hes::Hessenberg, proj::Projector, k)
  axpy!(hes.phi, unsafe_view(arnold.W, :, arnold.lastIdx), sol.x)
  sol.rho = abs(hes.phihat) * sqrt(proj.j + 1.)
end

end
