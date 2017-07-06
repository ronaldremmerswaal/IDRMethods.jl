module IDRMethods

export fqmrIDRs

using Base.BLAS
using Base.LinAlg

type Identity
end

type Projector
  j::Integer
  mu
  M::Matrix
  R0::Matrix
  u
  m

  Projector(n, s, T) = new(0, zero(T), Matrix{T}(s, s), Matrix{T}(n, s), zeros(T, s), Vector{T}(s))
end

type Hessenberg
  n::Integer
  s::Integer
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
  G::Matrix
  W::Matrix
  g
  n::Integer
  s::Integer
  v           # last projected orthogonal to R0
  vhat

  alpha

  # TODO how many n-vectors do we need? (g, v, vhat)
  Arnoldi(A, P, g, n, s) = new(A, P, [1 : s...], Matrix{eltype(g)}(n, s), Matrix{eltype(g)}(n, s + 1), g, n, s, g, Vector{eltype(g)}(n), Vector{eltype(g)}(s))
  # TODO norm of r0 is computed 3 times now (also in Solution...)
end

type Solution
  x
  rho
  rho0
  tol

  Solution(x, rho, tol) = new(x, rho, rho, tol)
end


function fqmrIDRs(A, b; s::Integer = 8, tol = 1E-6, maxIt::Integer = size(b, 1), x0 = zeros(b), P = Identity())

  # TODO skip if x0 = 0
  r0 = b - A * x0
  rho0 = vecnorm(r0)
  hessenberg = Hessenberg(size(b, 1), s, eltype(b), rho0)
  arnoldi = Arnoldi(A, P, r0 / rho0, size(b, 1), s)
  arnoldi.W[:, 1] = zeros(b) # TODO put inside arnoldi constructor
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
        apply!(projector, arnoldi)
      end

      cycle!(arnoldi)
      cycle!(hessenberg)

      expand!(arnoldi)

      mapToIDRSpace(arnoldi, hessenberg, projector, k)

      orthogonalize!(arnoldi, hessenberg, k)
      update!(hessenberg, projector, iter)
      update!(arnoldi, hessenberg, k, iter)

      update!(solution, arnoldi, hessenberg, projector, k)

      if isConverged(solution) || iter == maxIt
        stopped = true
        break
      end
    end
  end

  return solution.x, solution.rho
end


function apply!(proj::Projector, arnold::Arnoldi)
  proj.m = gemv('C', 1.0, proj.R0, arnold.v)
  proj.u = proj.M \ proj.m
  gemv!('N', -1.0, arnold.G, proj.u, 1.0, arnold.v)
  proj.u = -proj.u[arnold.permG]
  proj.M[:, arnold.permG[1]] = proj.m
end


@inline function initialize!(proj::Projector, arnold::Arnoldi)
  # TODO replace by in-place orth?
  proj.R0, = qr(rand(eltype(arnold.G), arnold.n, arnold.s))
  proj.M = gemm('C', 'N', 1.0, proj.R0, arnold.G)
end

function nextIDRSpace!(proj::Projector, arnold::Arnoldi)
  proj.j += 1

  # Compute residual minimizing mu
  tv = vecdot(last(arnold), arnold.v)
  tt = vecdot(last(arnold), last(arnold))

  omega = tv / tt
  rho = tv / (sqrt(tt) * norm(arnold.v))
  if abs(rho) < 0.7
    omega *= 0.7 / abs(rho)
  end
  proj.mu = abs(omega) > eps() ? 1. / omega : 1.
end

@inline function cycle!(hes::Hessenberg)
  hes.cosine[1 : end - 1] = hes.cosine[2 : end]
  hes.sine[1 : end - 1] = hes.sine[2 : end]
end

# Updates the QR factorization of H
function update!(hes::Hessenberg, projector, iter)
  axpy!(projector.mu, projector.u, hes.h[1 : end - 2])
  hes.h[end - 1] += projector.mu

  hes.r[1] = 0.
  hes.r[2 : end] = hes.h

  # Apply previous Givens rotations to new column of h
  @inbounds for l = max(1, hes.s + 3 - iter) : hes.s + 1
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

# TODO see if we can include g in G
@inline last(arnold::Arnoldi) = arnold.g

@inline function cycle!(arnold::Arnoldi)
  pGEnd = arnold.permG[1]
  arnold.permG[1 : end - 1] = arnold.permG[2 : end]
  arnold.permG[end] = pGEnd

  arnold.G[:, pGEnd] = arnold.g
end

@inline evalPrecon!(P::Identity, v) = copy(v)

@inline function expand!(arnold::Arnoldi)

  arnold.vhat = evalPrecon!(arnold.P, arnold.v)
  A_mul_B!(arnold.g, arnold.A, arnold.vhat)
end

# function expand!(arnold::Arnoldi, P::Identity)
#
# end

function update!(arnold::Arnoldi, hes::Hessenberg, k, iter)
  if iter > arnold.s
    # TODO make periodic iterator such that view can be used here on hes.r
    gemv!('N', -1.0, arnold.W, hes.r[[arnold.s + 2 - k : arnold.s + 1; 1 : arnold.s + 1 - k]], 1.0, arnold.vhat)
  else
    gemv!('N', -1.0, view(arnold.W, :, 1 : k), view(hes.r, arnold.s + 2 - k : arnold.s + 1), 1.0, arnold.vhat)
  end
  wIdx = k > arnold.s ? 1 : k + 1
  arnold.W[:, wIdx] = arnold.vhat / hes.r[end - 1]
end

function orthogonalize!(arnold::Arnoldi, hes::Hessenberg, k)
  # TODO (repeated) CGS?
  hes.h[1 : arnold.s + 1 - k] = 0.
  if k < arnold.s + 1
    @inbounds for l in 1 : k
      arnold.alpha[l] = vecdot(view(arnold.G, :, arnold.permG[arnold.s - k + l]), arnold.g)
      axpy!(-arnold.alpha[l], view(arnold.G, :, arnold.permG[arnold.s - k + l]), arnold.g)
    end
    hes.h[arnold.s + 2 - k : arnold.s + 1] = view(arnold.alpha, 1 : k)
  end
  hes.h[end] = vecnorm(arnold.g)
  scale!(arnold.g, 1 / hes.h[end])
  arnold.v = copy(last(arnold)) # TODO is this needed?
end

function mapToIDRSpace(arnoldi::Arnoldi, hes::Hessenberg, projector::Projector, k)
  if k == arnoldi.s + 1
    nextIDRSpace!(projector, arnoldi)
  end
  if projector.j > 0
    axpy!(-projector.mu, arnoldi.v, arnoldi.g);
  end
end

@inline function isConverged(sol::Solution)
  return sol.rho < sol.tol * sol.rho0
end

function update!(sol::Solution, arnold::Arnoldi, hes::Hessenberg, proj::Projector, k)
  wIdx = k > arnold.s ? 1 : k + 1
  axpy!(hes.phi, view(arnold.W, :, wIdx), sol.x)

  sol.rho = abs(hes.phihat) * sqrt(proj.j + 1.)
end

end
