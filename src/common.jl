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

type NormalSolution <: Solution
  x
  ρ
  rho0
  tol
  r

  NormalSolution(x, ρ, tol, r0) = new(x, [ρ], ρ, tol, r0)
end

abstract type SmoothedSolution <: Solution end


# With residual smoothing
type QMRSmoothedSolution <: SmoothedSolution
  ρ
  rho0
  tol
  r

  # Residual smoothing as proposed in
  #   Residual Smoothing Techniques for Iterative Methods
  #   Lu Zhou and Homer F. Walker
  # Algorithm 3.2.2
  η
  τ
  x
  s
  u
  v

  QMRSmoothedSolution(x, ρ, tol, r0) = new([ρ], ρ, tol, r0, ρ ^ 2, ρ ^ 2, x, copy(r0), zeros(eltype(r0), size(r0)), zeros(eltype(r0), size(r0)))
end

type MRSmoothedSolution <: SmoothedSolution
  ρ
  rho0
  tol
  r

  # Algorithm 2.2
  x
  s
  u
  v

  MRSmoothedSolution(x, ρ, tol, r0) = new([ρ], ρ, tol, r0, x, copy(r0), zeros(eltype(r0), size(r0)), zeros(eltype(r0), size(r0)))
end


function nextIDRSpace!(proj::Projector, idr::IDRSpace)
  proj.j += 1

  # Compute residual minimizing μ
  ν = vecdot(unsafe_view(idr.G, :, idr.latestIdx), idr.v)
  τ = vecdot(unsafe_view(idr.G, :, idr.latestIdx), unsafe_view(idr.G, :, idr.latestIdx))

  proj.ω = ν / τ
  η = ν / (sqrt(τ) * vecnorm(idr.v))
  if abs(η) < proj.κ
    proj.ω *= proj.κ / abs(η)
  end
  # TODO condest(A)? instead of 1.
  proj.μ = abs(proj.ω) > eps(real(eltype(idr.v))) ? 1. / proj.ω : 1.

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

@inline function isConverged(sol::Solution)
  return sol.ρ[end] < sol.tol * sol.rho0
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

@inline evalPrecon!(vhat, P::Identity, v) = copy!(vhat, v)
@inline function evalPrecon!(vhat, P::Preconditioner, v)
  A_ldiv_B!(vhat, P, v)
end
@inline function evalPrecon!(vhat, P::Function, v)
  P(vhat, v)
end



# To ensure contiguous memory, we often have to split the projections in 2 blocks
function skewProject!(v, G1, G2, R0, lu, α, u, uIdx1, uIdx2, m, skewT::SingleSkew)
  Ac_mul_B!(m, R0, v)
  A_ldiv_B!(α, lu, m)

  copy!(u, α[[uIdx1; uIdx2]])
  gemv!('N', -1.0, G1, u[1 : length(uIdx1)], 1.0, v)
  gemv!('N', -1.0, G2, u[length(uIdx1) + 1 : end], 1.0, v)
end

function skewProject!(v, G, R0, lu, α, u, uIdx, m, skewT::SingleSkew)
  Ac_mul_B!(m, R0, v)
  A_ldiv_B!(α, lu, m)

  copy!(u, α[uIdx])
  gemv!('N', -1.0, G, u, 1.0, v)
end


# function skewProject!(v, G1, G2, R0, lu, u, idx1, idx2, perm, m, skewT::RepeatSkew)
#   Ac_mul_B!(m, R0, v)
#   A_ldiv_B!(u, lu, m)
#   u[:] = u[perm]
#
#   gemv!('N', -1.0, G1, unsafe_view(u, idx1), 1.0, v)
#   gemv!('N', -1.0, G2, unsafe_view(u, idx2), 1.0, v)
#
#   happy = vecnorm(v) < skewT.one * vecnorm(u)
#
#   if happy return end
#
#   mUpdate = zeros(m)
#   uUpdate = zeros(u)
#   for idx = 2 : skewT.maxRepeat
#     # Repeat projection
#     Ac_mul_B!(mUpdate, R0, v)
#     A_ldiv_B!(uUpdate, lu, mUpdate)
#     uUpdate[:] = uUpdate[perm]
#
#     gemv!('N', -1.0, G1, unsafe_view(uUpdate, idx1), 1.0, v)
#     gemv!('N', -1.0, G2, unsafe_view(uUpdate, idx2), 1.0, v)
#
#     axpy!(1.0, mUpdate, m)
#     axpy!(1.0, uUpdate, u)
#
#     happy = vecnorm(v) > skewT.one * vecnorm(uUpdate)
#     if happy break end
#   end
# end

# Solves x(range) = M(range, range) \ b(range),
# for indexing range range. M is assumed lower triangular
function lowerBlockSolve!(x, M, b, range)
  for i = range
    x[i] = b[i]
    for j = range[1] : i - 1
      x[i] -= M[i, j] * x[j]
    end
    x[i] /= M[i, i]
  end
end
