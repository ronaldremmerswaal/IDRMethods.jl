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
  proj.μ = abs(proj.ω) > eps(real(eltype(idr.v))) ? one(eltype(idr.v)) / proj.ω : one(eltype(idr.v))

end

function orthogonalize!(g, G, h, orthT::ClassicalGS)
  Ac_mul_B!(h, G, g)
  gemv!('N', -one(eltype(G)), G, h, one(eltype(G)), g)

  return vecnorm(g)
end

# Orthogonalize g w.r.t. G, and store coeffs in h (NB g is not normalized)
function orthogonalize!(g, G, h, orthT::RepeatedClassicalGS)
  Ac_mul_B!(h, G, g)
  # println(0, ", normG = ", vecnorm(g), ", normH = ", vecnorm(h))
  gemv!('N', -one(eltype(G)), G, h, one(eltype(G)), g)

  normG = vecnorm(g)
  normH = vecnorm(h)

  happy = normG < orthT.one * normH || normH < orthT.tol * normG
  # println(1, ", normG = ", normG, ", normH = ", vecnorm(G' * g))
  if happy return normG end

  for idx = 2 : orthT.maxRepeat
    updateH = Vector(h)

    Ac_mul_B!(updateH, G, g)
    gemv!('N', -one(eltype(G)), G, updateH, one(eltype(G)), g)

    axpy!(one(eltype(G)), updateH, h)

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
  gemv!('N', -one(eltype(G1)), G1, unsafe_view(u, 1 : length(uIdx1)), one(eltype(G1)), v)
  gemv!('N', -one(eltype(G2)), G2, unsafe_view(u, length(uIdx1) + 1 : length(u)), one(eltype(G2)), v)
end

function skewProject!(v, G, R0, lu, α, u, uIdx, m, skewT::SingleSkew)
  Ac_mul_B!(m, R0, v)
  A_ldiv_B!(α, lu, m)

  copy!(u, α[uIdx])
  gemv!('N', -one(eltype(G)), G, u, one(eltype(G)), v)
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
