module IDRMethods

export fqmrIDRs

using SugarBLAS
import Base.*

type Projector
  j::Integer
  mu
  M::Matrix
  R0::Matrix
  u
  m

  Projector(n, s, T) = new(0, zero(T), Matrix{T}(s, s), Matrix{T}(n, s), Vector{T}(s), Vector{T}(n))
end

type Hessenberg
  n::Integer
  s::Integer
  r
  h
  cosine
  sine

  Hessenberg(n, s, T) = new(n, s, zeros(T, s + 3), zeros(T, s + 2), zeros(T, s + 2), zeros(T, s + 2))
end

type Arnoldi
  A
  permG
  G::Matrix
  W::Matrix
  g
  n::Integer
  s::Integer
  last::Integer
  v           # last projected orthogonal to R0
  vhat

  alpha

  # TODO how many n-vectors do we need? (g, v, vhat)
  Arnoldi(A, r0, n, s) = new(A, [1 : s...], Matrix{eltype(r0)}(n, s), Matrix{eltype(r0)}(n, s + 1), Vector{eltype(r0)}(n), n, s, 0, Vector{eltype(r0)}(n), Vector{eltype(r0)}(n), Vector{eltype(r0)}(s))
end

type Solution
  x
  phihat
  rho
  rho0
  tol

  Solution(x, rho, tol) = new(x, rho, rho, rho, tol)
end


function fqmrIDRs(A, b; s::Integer = 8, tol = 1E-6, maxIt::Integer = size(b, 1), x0 = zeros(b))

  # TODO skip if x0 = 0
  r0 = b - A * x0
  hessenberg = Hessenberg(size(b, 1), s, eltype(b))
  arnoldi = Arnoldi(A, r0, size(b, 1), s)
  solution = Solution(x0, vecnorm(r0), tol)
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

      expand!(arnoldi)

      mapToIDRSpace(arnoldi, projector, k)

      orthogonalize!(arnoldi, hessenberg, k)

      update!(hessenberg, iter)

      update!(solution, arnoldi, hessenberg, projector, k, iter)

      if isConverged(solution) || iter > maxIt
        stopped = true
        break
      end
    end
  end

  return solution.x, solution.rho
end


function apply!(p::Projector, a::Arnoldi)
  a.v = copy(last(a))

  p.m = BLAS.gemv('C', 1.0, p.R0, a.v)
  p.u[1 : a.s] = p.M \ p.m
  a.v -= a.G * p.u[1 : a.s]
  p.u[1 : a.s] = -p.u[a.permG]
  p.M[:, a.permG[end]] = p.m
end


@inline function initialize!(p::Projector, a::Arnoldi)
  p.R0, = qr(rand(eltype(a.G), a.n, a.s))
  p.M = BLAS.gemm('C', 'N', 1.0, p.R0, a.G)
end

function nextIDRSpace!(p::Projector, a::Arnoldi)
  p.j += 1

  # Compute residual minimizing mu
  tv = vecdot(last(a), a.v)
  tt = vecdot(last(a), last(a))

  p.mu = tv / tt
  rho = tv / (sqrt(tt) * norm(a.v))
  if abs(rho) < 0.7
    p.mu *= 0.7 / abs(rho)
  end
end


# Updates the QR factorization of H
function update!(hes::Hessenberg, iter)
  hes.r[1] = 0.
  hes.r[2 : end] = hes.h

  @inbounds for l = max(1, hes.s + 3 - iter) : hes.s + 1
    oldRl = hes.r[l]
    hes.r[l] = hes.cosine[l] * oldRl + hes.sine[l] * hes.r[l + 1]
    hes.r[l + 1] = -conj(hes.sine[l]) * oldRl + hes.cosine[l] * hes.r[l + 1]
  end

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
end

# TODO see if we can include g in G
@inline last(a::Arnoldi) = a.g

@inline function expand!(a::Arnoldi)
  a.vhat = a.v  # TODO optional preconditioning
  a.g = a.A * a.v
end

function orthogonalize!(a::Arnoldi, hes::Hessenberg, k)
  if k < a.s + 1
    @inbounds for l in 1 : k
      a.alpha[l] = vecdot(view(a.G, :, a.permG[a.s - k + l]), a.g)
      @blas! a.g -= a.alpha[l] * view(a.G, :, a.permG[a.s - k + l])
    end
    hes.h[a.s + 1 - k + 1 : a.s + 1] += view(a.alpha, 1 : k)
  end
  hes.h[end] = vecnorm(a.g)
  @blas! a.g *= 1 / hes.h[end]
end

function mapToIDRSpace(arnoldi::Arnoldi, projector::Projector, k)
  if k == arnoldi.s + 1
    nextIDRSpace!(projector, arnoldi)
  end
  if projector.j > 0
    arnoldi.g -= projector.mu * arnoldi.v
  end
end

@inline function isConverged(sol::Solution)
  return sol.rho < sol.tol * sol.rho0
end

function update!(sol::Solution, a::Arnoldi, hes::Hessenberg, p::Projector, k, iter)
  phi = hes.cosine[end] * sol.phihat
  sol.phihat = -conj(hes.sine[end]) * sol.phihat
  wIdx = k > a.s ? 1 : k + 1
  if iter > a.s
    a.W[:, wIdx] = a.vhat - a.W * view(hes.r, [a.s + 2 - k : a.s + 1; 1 : a.s + 1 - k])
  else
    a.W[:, wIdx] = a.vhat - view(a.W, :, 1 : k) * view(hes.r, a.s + 2 - k : a.s + 1)
  end
  @blas! a.W[:, wIdx] *= 1. / hes.r[end - 1]
  @blas! sol.x += phi * view(a.W, :, wIdx)

  sol.rho = abs(sol.phihat) * sqrt(p.j + 1.)
end

end
