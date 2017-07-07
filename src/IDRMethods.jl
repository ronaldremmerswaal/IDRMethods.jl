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

  Projector(n, s, T) = new(0, zero(T), zeros(T, s, s), Matrix{T}(n, s), zeros(T, s), Vector{T}(s))
end

type Hessenberg
  n
  s
  r
  h
  cosine
  sine
  phi
  phihat

  Hessenberg(n, s, T, rho0) = new(n, s, zeros(T, s + 3), zeros(T, s + 2), zeros(T, s + 2), zeros(T, s + 2), zero(T), rho0)
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


function fqmrIDRs(A, b; s = 8, tol = sqrt(eps(real(eltype(b)))), maxIt = size(b, 1), x0 = [], P = Identity())

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
  projector = Projector(size(b, 1), s, eltype(b))

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
  if arnold.lastIdx > 1
    gemv!('N', -1.0, unsafe_view(arnold.G, :, 1 : arnold.lastIdx - 1), proj.u[arnold.permG[end - k + 2 : end]], 1.0, arnold.v)
  end
  if arnold.lastIdx <= arnold.s
    gemv!('N', -1.0, unsafe_view(arnold.G, :, arnold.lastIdx + 1 : arnold.s + 1), proj.u[arnold.permG[1 : end - k + 1]], 1.0, arnold.v)
  end
  proj.u[:] = proj.u[arnold.permG]
  proj.M[:, arnold.permG[1]] = proj.m
end


@inline function initialize!(proj::Projector, arnold::Arnoldi)
  # TODO replace by in-place orth?
  rand!(proj.R0)
  qrfact!(proj.R0)
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
  axpy!(-proj.mu, proj.u, unsafe_view(hes.h, 1 : hes.s))
  hes.h[end - 1] += proj.mu
  hes.r[1] = 0.
  hes.r[2 : end] = hes.h

  # Apply previous Givens rotations to new column of h
  for l = max(1, hes.s + 3 - iter) : hes.s + 1
    oldRl = hes.r[l]
    hes.r[l] = hes.cosine[l] * oldRl + hes.sine[l] * hes.r[l + 1]
    hes.r[l + 1] = -conj(hes.sine[l]) * oldRl + hes.cosine[l] * hes.r[l + 1]
  end

  # Compute new Givens rotation
  a = hes.r[end - 1]
  b = hes.r[end]
  if abs(a) < eps()
    hes.sine[end] = 1.
    hes.cosine[end] = 0.
    hes.r[end - 1] = b
  else
    t = abs(a) + abs(b)
    rho = t * sqrt(abs(a / t) ^ 2 + abs(b / t) ^ 2)
    alpha = a / abs(a)

    hes.sine[end] = alpha * conj(b) / rho
    hes.cosine[end] = abs(a) / rho
    hes.r[end - 1] = alpha * rho
  end

  hes.phi = hes.cosine[end] * hes.phihat
  hes.phihat = -conj(hes.sine[end]) * hes.phihat
end

@inline function cycle!(arnold::Arnoldi)
  pGEnd = arnold.permG[1]
  arnold.permG[1 : end - 1] = unsafe_view(arnold.permG, 2 : arnold.s)
  arnold.permG[end] = pGEnd
end

@inline evalPrecon!(P::Identity, v) =
@inline function evalPrecon!(P::Preconditioner, v)
  v = P \ v
end

@inline function expand!(arnold::Arnoldi, k)
  arnold.lastIdx = k > arnold.s ? 1 : k + 1
  copy!(arnold.vhat, arnold.v)
  evalPrecon!(arnold.P, arnold.vhat)
  A_mul_B!(unsafe_view(arnold.G, :, arnold.lastIdx), arnold.A, arnold.vhat)
end

function updateW!(arnold::Arnoldi, hes::Hessenberg, k, iter)
  if iter > arnold.s
    # TODO make periodic iterator such that view can be used here on hes.r
    gemv!('N', -1.0, arnold.W, hes.r[[arnold.s + 2 - k : arnold.s + 1; 1 : arnold.s + 1 - k]], 1.0, arnold.vhat)
  else
    gemv!('N', -1.0, unsafe_view(arnold.W, :, 1 : k), hes.r[arnold.s + 2 - k : arnold.s + 1], 1.0, arnold.vhat)
  end
  wIdx = k > arnold.s ? 1 : k + 1
  copy!(unsafe_view(arnold.W, :, wIdx), arnold.vhat)
  scale!(unsafe_view(arnold.W, :, wIdx), 1 / hes.r[end - 1])
end

function updateG!(arnold::Arnoldi, hes::Hessenberg, k)
  # TODO (repeated) CGS?
  hes.h[:] = 0.
  aIdx = arnold.lastIdx
  if k < arnold.s + 1
    # for l in 1 : k
    #   arnold.alpha[l] = vecdot(unsafe_view(arnold.G, :, l), unsafe_view(arnold.G, :, aIdx))
    #   axpy!(-arnold.alpha[l], unsafe_view(arnold.G, :, l), unsafe_view(arnold.G, :, aIdx))
    # end

    gemv!('C', 1.0, unsafe_view(arnold.G, :, 1 : k), unsafe_view(arnold.G, :, aIdx), 0.0, unsafe_view(arnold.alpha, 1 : k))
    gemv!('N', -1.0, unsafe_view(arnold.G, :, 1 : k), arnold.alpha[1 : k], 1.0, unsafe_view(arnold.G, :, aIdx))

    hes.h[arnold.s + 2 - k : arnold.s + 1] = unsafe_view(arnold.alpha, 1 : k)
  end

  hes.h[end] = vecnorm(unsafe_view(arnold.G, :, aIdx))
  scale!(unsafe_view(arnold.G, :, aIdx), 1 / hes.h[end])
  copy!(arnold.v, unsafe_view(arnold.G, :, aIdx))

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
  wIdx = k > arnold.s ? 1 : k + 1
  axpy!(hes.phi, unsafe_view(arnold.W, :, wIdx), sol.x)

  sol.rho = abs(hes.phihat) * sqrt(proj.j + 1.)
end

end
