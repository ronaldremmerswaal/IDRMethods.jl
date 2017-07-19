type BiOSpace{T} <: IDRSpace{T}
  s

  A
  P

  G::DenseMatrix{T}
  W::DenseMatrix{T}

  v::DenseVector{T}

  β

  latestIdx

end
# TODO how many n-vectors do we need? (g, v, vhat)
BiOSpace{T}(A, P, g::DenseVector{T}, s) = BiOSpace{T}(s, A, P, Matrix{T}(length(g), s + 1), Matrix{T}(length(g), s + 1), copy(g), zero(T), 0)

type BiOProjector{T} <: Projector{T}
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

end
BiOProjector(n, s, R0, κ, T) = BiOProjector{T}(n, s, 0, zero(T), zero(T), eye(T, s), zeros(T, s), zeros(T, s), R0, κ)

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

  rho0 = norm(r0)
  if smoothing == "none"
    solution = NormalSolution(x0, rho0, tol, r0)
  elseif smoothing == "MR"
    solution = MRSmoothedSolution(x0, rho0, tol, r0)
  elseif smoothing == "QMR"
    solution = QMRSmoothedSolution(x0, rho0, tol, r0)
  end

  idrSpace = BiOSpace(A, P, r0, s)
  projector = BiOProjector(size(b, 1), s, R0, kappa, eltype(b))

  return IDRMethod(solution, idrSpace, projector, maxIt)
end

# Maps v -> v - G * (R0' * G)^-1 * R0 * v
function apply!{T}(proj::BiOProjector{T}, idr::BiOSpace{T})
  k = idr.latestIdx == idr.s + 1 ? 1 : idr.latestIdx + 1

  if k <= idr.s
    proj.α[k : proj.s] = unsafe_view(proj.m, k : proj.s)
    if proj.j == 0
      proj.α[k] = proj.α[k] / proj.M[k, k]
      axpy!(proj.α[k], unsafe_view(proj.M, k + 1 : proj.s, k), unsafe_view(proj.α, k + 1 : proj.s))
    else
      solveLowerTriangular!(proj.α, proj.M, k)
    end
    if proj.j > 0
      gemv!('N', -one(T), unsafe_view(idr.G, :, k : idr.s), unsafe_view(proj.α, k : idr.s), one(T), idr.v)
    end
  end
end

# NB assuming unit diagonal
function solveLowerTriangular!{T}(α::StridedVector{T}, M::StridedMatrix{T}, startIdx::Int)
  for i = startIdx : length(α)
    for j = startIdx : i - 1
      α[i] -= M[i, j] * α[j]
    end
  end
end

function expand!{T}(idr::BiOSpace{T}, proj::BiOProjector{T})
  idr.latestIdx = idr.latestIdx > idr.s ? 1 : idr.latestIdx + 1

  evalPrecon!(idr.v, idr.P, idr.v) # TODO is this safe?

  if proj.j > 0 && idr.latestIdx < idr.s + 1
    gemv!('N', one(T), unsafe_view(idr.W, :, idr.latestIdx : idr.s), unsafe_view(proj.α, idr.latestIdx : idr.s), proj.ω, idr.v)
  end

  copy!(unsafe_view(idr.W, :, idr.latestIdx), idr.v)
  A_mul_B!(unsafe_view(idr.G, :, idr.latestIdx), idr.A, idr.v)
end

@inline function mapToIDRSpace!{T}(idr::BiOSpace{T}, proj::BiOProjector{T})
  # Do nothing, this is done inside expand!
end

function update!{T}(idr::BiOSpace{T}, proj::BiOProjector{T}, k, iter)
  if k == idr.s + 1
    idr.β = proj.ω
  else
    # Biorthogonalise the pair R0, G
    α = biOrthogonalize!(unsafe_view(idr.G, :, k), idr.G, proj.R0, k - 1)

    # And update W accordingly
    gemv!('N', -one(T), unsafe_view(idr.W, :, 1 : k - 1), α, one(T), unsafe_view(idr.W, :, k))

    # NB Scale G such that diag(proj.M) = eye(s)
    # TODO check if inner product nonzero..
    tmp = dot(unsafe_view(proj.R0, :, k), unsafe_view(idr.G, :, k))
    scale!(unsafe_view(idr.G, :, k), one(T) / tmp)
    scale!(unsafe_view(idr.W, :, k), one(T) / tmp)

    idr.β = proj.m[k]
  end
end

function biOrthogonalize!{T}(g::StridedVector{T}, G::StridedMatrix{T}, R0::StridedMatrix{T}, endIdx::Int)
  α = Vector{T}(endIdx)
  for j = 1 : endIdx
    α[j] = dot(unsafe_view(R0, :, j), g)
    axpy!(-α[j], unsafe_view(G, :, j), g)
  end
  return α
end

function update!{T}(proj::BiOProjector{T}, idr::BiOSpace{T})
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

function update!{T}(sol::NormalSolution{T}, idr::BiOSpace{T}, proj::BiOProjector{T})
  axpy!(-idr.β, unsafe_view(idr.G, :, idr.latestIdx), sol.r)
  axpy!(idr.β, unsafe_view(idr.W, :, idr.latestIdx), sol.x)
  push!(sol.ρ, norm(sol.r))
  copy!(idr.v, sol.r)
end

function update!{T}(sol::QMRSmoothedSolution{T}, idr::BiOSpace{T}, proj::BiOProjector{T})
  axpy!(-idr.β, unsafe_view(idr.G, :, idr.latestIdx), sol.r)
  copy!(idr.v, sol.r)

  # Residual smoothing
  axpy!(idr.β, unsafe_view(idr.G, :, idr.latestIdx), sol.u)
  axpy!(idr.β, unsafe_view(idr.W, :, idr.latestIdx), sol.v)

  sol.η = norm(sol.s - sol.u) ^ 2
  sol.τ = one(T) / (one(T) / sol.τ + one(T) / sol.η)

  ratio = sol.τ / sol.η

  axpy!(-ratio, sol.u, sol.s)
  axpy!(ratio, sol.v, sol.x)

  scale!(sol.u, one(T) - ratio)
  scale!(sol.v, one(T) - ratio)

  push!(sol.ρ, norm(sol.s))
end

function update!{T}(sol::MRSmoothedSolution{T}, idr::BiOSpace{T}, proj::BiOProjector{T})
  axpy!(-idr.β, unsafe_view(idr.G, :, idr.latestIdx), sol.r)
  copy!(idr.v, sol.r)

  # Residual smoothing
  axpy!(idr.β, unsafe_view(idr.G, :, idr.latestIdx), sol.u)
  axpy!(idr.β, unsafe_view(idr.W, :, idr.latestIdx), sol.v)

  ratio = dot(sol.s, sol.u) / dot(sol.u, sol.u)

  axpy!(-ratio, sol.u, sol.s)
  axpy!(ratio, sol.v, sol.x)

  scale!(sol.u, one(T) - ratio)
  scale!(sol.v, one(T) - ratio)

  push!(sol.ρ, norm(sol.s))
end
