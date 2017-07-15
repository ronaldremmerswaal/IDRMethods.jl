type BiOSpace <: IDRSpace
  n
  s

  A
  P

  G
  W

  v

  β

  latestIdx

  # TODO how many n-vectors do we need? (g, v, vhat)
  BiOSpace(n, s, A, P, r0, T) = new(n, s, A, P, Matrix{T}(n, s + 1), Matrix{T}(n, s + 1), copy(r0), zero(T), 0)
end

type BiOProjector <: Projector
  n
  s
  j
  μ
  ω
  M

  α
  m

  R0

  κ

  BiOProjector(n, s, R0, κ, T) = new(n, s, 0, zero(T), zero(T), eye(T, s), zeros(T, s), zeros(T, s), R0, κ)
end

function biIDRs(A, b; s = 8, tol = sqrt(eps(real(eltype(b)))), maxIt = size(b, 1), x0 = [], P = Identity(), R0 = qr(rand(size(b, 1), min(size(b, 1), s)))[1], kappa = 0.7, smoothing = "MR")

  s = min(s, size(b, 1))
  if length(R0) > 0 && size(R0) != (length(b), s)
    error("size(R0) != [", length(b), ", $s] (User provided shadow residuals are of incorrect size)")
  end

  if length(x0) == 0
    x0 = zeros(b)
    r0 = copy(b)
  else
    r0 = b - A * x0
  end

  rho0 = vecnorm(r0)
  if smoothing == "none"
    solution = NormalSolution(x0, rho0, tol, r0)
  elseif smoothing == "MR"
    solution = MRSmoothedSolution(x0, rho0, tol, r0)
  elseif smoothing == "QMR"
    solution = QMRSmoothedSolution(x0, rho0, tol, r0)
  end

  idrSpace = BiOSpace(size(b, 1), s, A, P, r0, eltype(b))
  projector = BiOProjector(size(b, 1), s, R0, kappa, eltype(b))

  return IDRMethod(solution, idrSpace, projector, maxIt)
end

# Maps v -> v - G * (R0' * G)^-1 * R0 * v
function apply!(proj::BiOProjector, idr::BiOSpace)
  k = idr.latestIdx == idr.s + 1 ? 1 : idr.latestIdx + 1

  if k <= idr.s
    proj.α[k : proj.s] = unsafe_view(proj.m, k : proj.s)
    if proj.j == 0
      proj.α[k] = proj.α[k] / proj.M[k, k]
      axpy!(proj.α[k], unsafe_view(proj.M, k + 1 : proj.s, k), unsafe_view(proj.α, k + 1 : proj.s))
    else
      for i = k : proj.s
        for j = k : i - 1
          proj.α[i] -= proj.M[i, j] * proj.α[j]
        end
      end
    end
    if proj.j > 0
      gemv!('N', -one(eltype(idr.v)), unsafe_view(idr.G, :, k : idr.s), unsafe_view(proj.α, k : idr.s), one(eltype(idr.v)), idr.v)
    end
  end
end

function expand!(idr::BiOSpace, proj::BiOProjector)
  idr.latestIdx = idr.latestIdx > idr.s ? 1 : idr.latestIdx + 1

  evalPrecon!(idr.v, idr.P, idr.v) # TODO is this safe?

  if proj.j > 0 && idr.latestIdx < idr.s + 1
    gemv!('N', one(eltype(idr.v)), unsafe_view(idr.W, :, idr.latestIdx : idr.s), unsafe_view(proj.α, idr.latestIdx : idr.s), proj.ω, idr.v)
  end

  idr.W[:, idr.latestIdx] = idr.v
  A_mul_B!(unsafe_view(idr.G, :, idr.latestIdx), idr.A, idr.v)

end

@inline function mapToIDRSpace!(idr::BiOSpace, proj::BiOProjector)
  # Do nothing, this is done inside expand!
end

function update!(idr::BiOSpace, proj::BiOProjector, k, iter)
  if k == idr.s + 1
    idr.β = proj.ω
  else
    # Biorthogonalise the pair R0, G
    α = Vector{eltype(idr.v)}(k - 1)
    for j = 1 : k - 1
      α[j] = vecdot(unsafe_view(proj.R0, :, j), unsafe_view(idr.G, :, k))
      axpy!(-α[j], unsafe_view(idr.G, :, j), unsafe_view(idr.G, :, k))
    end

    # And update W accordingly
    gemv!('N', -one(eltype(idr.W)), unsafe_view(idr.W, :, 1 : k - 1), α, one(eltype(idr.W)), unsafe_view(idr.W, :, k))

    # NB Scale G such that diag(proj.M) = eye(s)
    # TODO check if inner product nonzero..
    tmp = vecdot(unsafe_view(proj.R0, :, k), unsafe_view(idr.G, :, k))
    scale!(unsafe_view(idr.G, :, k), 1. / tmp)
    scale!(unsafe_view(idr.W, :, k), 1 ./ tmp)

    idr.β = proj.m[k]
  end
end

function update!(proj::BiOProjector, idr::BiOSpace)
  k = idr.latestIdx == idr.s + 1 ? 1 : idr.latestIdx + 1

  if k == 1
    Ac_mul_B!(proj.m, proj.R0, idr.v)
  end

  if k > 1

    # Update it
    Ac_mul_B!(unsafe_view(proj.M, k : idr.s, k - 1), unsafe_view(proj.R0, :, k : idr.s), unsafe_view(idr.G, :, k - 1))
    if k <= idr.s
      axpy!(-idr.β, unsafe_view(proj.M, k : idr.s, k - 1), unsafe_view(proj.m, k : idr.s))
    end
  end

end

function update!(sol::NormalSolution, idr::BiOSpace, proj::BiOProjector)
  axpy!(-idr.β, unsafe_view(idr.G, :, idr.latestIdx), sol.r)
  axpy!(idr.β, unsafe_view(idr.W, :, idr.latestIdx), sol.x)
  push!(sol.ρ, vecnorm(sol.r))
  copy!(idr.v, sol.r)
end

function update!(sol::QMRSmoothedSolution, idr::BiOSpace, proj::BiOProjector)
  axpy!(-idr.β, unsafe_view(idr.G, :, idr.latestIdx), sol.r)
  copy!(idr.v, sol.r)

  # Residual smoothing
  axpy!(idr.β, unsafe_view(idr.G, :, idr.latestIdx), sol.u)
  axpy!(idr.β, unsafe_view(idr.W, :, idr.latestIdx), sol.v)

  sol.η = vecnorm(sol.s - sol.u) ^ 2
  sol.τ = 1. / (1. / sol.τ + 1 ./ sol.η)

  ratio = sol.τ / sol.η

  axpy!(-ratio, sol.u, sol.s)
  axpy!(ratio, sol.v, sol.x)

  scale!(sol.u, 1. - ratio)
  scale!(sol.v, 1. - ratio)

  push!(sol.ρ, vecnorm(sol.s))
end

function update!(sol::MRSmoothedSolution, idr::BiOSpace, proj::BiOProjector)
  axpy!(-idr.β, unsafe_view(idr.G, :, idr.latestIdx), sol.r)
  copy!(idr.v, sol.r)

  # Residual smoothing
  axpy!(idr.β, unsafe_view(idr.G, :, idr.latestIdx), sol.u)
  axpy!(idr.β, unsafe_view(idr.W, :, idr.latestIdx), sol.v)

  ratio = vecdot(sol.s, sol.u) / vecdot(sol.u, sol.u)

  axpy!(-ratio, sol.u, sol.s)
  axpy!(ratio, sol.v, sol.x)

  scale!(sol.u, 1. - ratio)
  scale!(sol.v, 1. - ratio)

  push!(sol.ρ, vecnorm(sol.s))
end
