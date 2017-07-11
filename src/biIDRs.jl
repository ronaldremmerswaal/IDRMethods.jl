type BiOSpace <: IDRSpace
  n
  s

  A
  P

  G
  W

  v
  r

  β

  latestIdx

  # TODO how many n-vectors do we need? (g, v, vhat)
  BiOSpace(n, s, A, P, r0, T) = new(n, s, A, P, Matrix{T}(n, s + 1), Matrix{T}(n, s + 1), r0, copy(r0), zero(T), 0)
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
  diagM

  κ

  BiOProjector(n, s, R0, κ, T) = new(n, s, 0, zero(T), zero(T), eye(T, s), zeros(T, s), zeros(T, s), R0, ones(T, s), κ)
end

function biIDRs(A, b; s = 8, tol = sqrt(eps(real(eltype(b)))), maxIt = size(b, 1), x0 = [], P = Identity(), R0 = qr(rand(size(b, 1), min(size(b, 1), s)))[1], kappa = 0.7)

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
  solution = Solution(x0, rho0, tol)
  idrSpace = BiOSpace(size(b, 1), s, A, P, r0, eltype(b))
  projector = BiOProjector(size(b, 1), s, R0, kappa, eltype(b))

  return IDRMethod(solution, idrSpace, projector, maxIt)
end

# Maps v -> v - G * (R0' * G)^-1 * R0 * v
function apply!(proj::BiOProjector, idr::BiOSpace)
  k = idr.latestIdx == idr.s + 1 ? 1 : idr.latestIdx + 1
  # TODO More efficiently if j = 0 (then only one nontrivial column)
  if k <= idr.s
    proj.α[k : idr.s] = LowerTriangular(proj.M[k : proj.s, k : proj.s]) \ unsafe_view(proj.m, k : proj.s)
    if proj.j > 0
      gemv!('N', -1.0, unsafe_view(idr.G, :, k : idr.s), unsafe_view(proj.α, k : idr.s), 1.0, idr.v)
    end
  end
end

function expand!(idr::BiOSpace, proj::BiOProjector)
  idr.latestIdx = idr.latestIdx > idr.s ? 1 : idr.latestIdx + 1

  evalPrecon!(idr.v, idr.P, idr.v) # TODO is this safe?

  if proj.j > 0 && idr.latestIdx < idr.s + 1
    gemv!('N', 1.0, unsafe_view(idr.W, :, idr.latestIdx : idr.s), unsafe_view(proj.α, idr.latestIdx : idr.s), proj.ω, idr.v)
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
    for j = 1 : k - 1
      α = vecdot(unsafe_view(proj.R0, :, j), unsafe_view(idr.G, :, k)) / proj.diagM[j]
      axpy!(-α, unsafe_view(idr.G, :, j), unsafe_view(idr.G, :, k))
      axpy!(-α, unsafe_view(idr.W, :, j), unsafe_view(idr.W, :, k))
    end

    proj.diagM[k] = vecdot(unsafe_view(proj.R0, :, k), unsafe_view(idr.G, :, k))
    idr.β = proj.m[k] / proj.diagM[k];
  end
end

function update!(proj::BiOProjector, idr::BiOSpace)
  k = idr.latestIdx == idr.s + 1 ? 1 : idr.latestIdx + 1

  if k == 1
    gemv!('C', 1.0, proj.R0, idr.r, 0.0, proj.m)
  end

  if k > 1
    # Update it
    gemv!('C', 1.0, unsafe_view(proj.R0, :, k : idr.s), unsafe_view(idr.G, :, k - 1), 0.0, unsafe_view(proj.M, k : idr.s, k - 1))
    proj.M[k - 1, k - 1] = proj.diagM[k - 1]
    if k <= idr.s
      axpy!(-idr.β, unsafe_view(proj.M, k : idr.s, k - 1), unsafe_view(proj.m, k : idr.s))
    end
  end

end

function update!(sol::Solution, idr::BiOSpace, proj::BiOProjector)
  axpy!(-idr.β, unsafe_view(idr.G, :, idr.latestIdx), idr.r)
  axpy!(idr.β, unsafe_view(idr.W, :, idr.latestIdx), sol.x)
  push!(sol.ρ, vecnorm(idr.r))
  idr.v = copy(idr.r)
end
