module ArnoldiBasis

export ArnoldiHH, expand!, getSolution, getSolution!, getBasis, gmresHH

include("harmenView.jl")

abstract type Arnoldi end

# Iteratively builds the Arnoldi basis using Householder reflections
type ArnoldiHH{T} <: Arnoldi
  A
  G::StridedMatrix{T}
  τ::StridedVector{T}
  latestIdx::Int

  Qe1::StridedVector{T}   # Equals Q' * e1 * ρ0
  reduce::Bool
  ρ::Vector{T}
  givensRot::Vector{LinAlg.Givens{T}}

  g::StridedVector{T}
end

function ArnoldiHH{T}(A, r0::StridedVector{T}; s::Int = size(A, 1), reduce::Bool = true)
  G = Matrix{T}(length(r0), s)
  G[:, 1] = r0

  τ = Vector{T}(s)
  τ[1] = LinAlg.reflector!(unsafe_view(G, :, 1))

  Qe1 = zeros(T, s)
  Qe1[1] = G[1, 1]

  ArnoldiHH{T}(A, G, τ, 1, Qe1, reduce, [abs(Qe1[1])], [], Vector{T}(length(r0)))
end

# Householder GMRes
function gmresHH(A, b; maxIt = length(b), x0::StridedVector = [], tol = sqrt(eps(real(eltype(b)))))
  if length(x0) == 0
    r0 = b
    x0 = zeros(b)
  else
    r0 = b - A * x
  end
  arnold = ArnoldiHH(A, r0, s = maxIt + 1)

  tol *= norm(b)

  for it = 1 : maxIt
    if arnold.ρ[end] < tol
      break
    end
    expand!(arnold)
  end
  return x0 + getSolution(arnold), arnold.ρ
end

function expand!{T}(arnold::ArnoldiHH{T})
  # MV
  g = zeros(T, size(arnold.G, 1))
  A_mul_B!(g, arnold.A, getBasisColumn!(arnold.g, arnold, arnold.latestIdx))
  expand!(arnold, g)
end

function expand!{T}(arnold::ArnoldiHH{T}, g::StridedVector{T})
  if arnold.latestIdx == size(arnold.G, 2)
    error("Maxiumum size reached")
  end
  arnold.G[:, arnold.latestIdx + 1] = g

  # Apply previous reflections
  reflectorApply!(arnold, unsafe_view(arnold.G, :, arnold.latestIdx + 1))
  arnold.latestIdx += 1

  # New reflection
  arnold.τ[arnold.latestIdx] = LinAlg.reflector!(unsafe_view(arnold.G, arnold.latestIdx : size(arnold.G, 1), arnold.latestIdx))

  if arnold.reduce
    # Apply previous Givens to new column
    for (idx, φ) ∈ enumerate(arnold.givensRot)
      arnold.G[idx : idx + 1, arnold.latestIdx] = φ * unsafe_view(arnold.G, idx : idx + 1, arnold.latestIdx)
    end

    if arnold.latestIdx ≤ size(arnold.G, 1)
      # New Givens for Hessenberg matrix
      φ, arnold.G[arnold.latestIdx - 1, arnold.latestIdx] = LinAlg.givens(unsafe_view(arnold.G, arnold.latestIdx - 1 : arnold.latestIdx, arnold.latestIdx), 1, 2)
      push!(arnold.givensRot, φ)

      # Apply Givens to rhs of small system (Qe1)
      arnold.Qe1[arnold.latestIdx - 1 : arnold.latestIdx] = φ * unsafe_view(arnold.Qe1, arnold.latestIdx - 1 : arnold.latestIdx)
      push!(arnold.ρ, abs(arnold.Qe1[arnold.latestIdx]))
    end
  end
end

@inline getSolution{T}(arnold::ArnoldiHH{T}) = getSolution!(Vector{T}(size(arnold.G, 1)), arnold)
function getSolution!{T}(sol::StridedVector{T}, arnold::ArnoldiHH{T})
  if arnold.reduce
    α = getReducedHessenberg(arnold) \ unsafe_view(arnold.Qe1, 1 : arnold.latestIdx - 1)
  else
    α = getHessenberg(arnold) \ arnold.Qe1[1 : min(arnold.latestIdx, size(arnold.G, 1))]
  end
  return getLinearComBasis(arnold, α)
end

@inline function getBasisColumn!{T}(col::StridedVector{T}, arnold::ArnoldiHH{T}, colIdx::Int)
  col[:] = zero(T)
  col[colIdx] = one(T)
  return reflectorReverseApply!(arnold, col, colIdx)
end
@inline getBasisColumn{T}(arnold::ArnoldiHH{T}, colIdx::Int) = getBasisColumn!(Vector{T}(size(arnold.G, 1)), arnold, colIdx)

function getHessenbergColumn{T}(arnold::ArnoldiHH{T}, colIdx)
  return arnold.G[1 : colIdx + 1, colIdx + 1]
end

# Returns β = Qn' * e1 * α1 + Qn' * e2 * α2 + ...
function getLinearComBasis!{T}(α1::StridedVector{T}, arnold::ArnoldiHH{T}, α::StridedVector{T})
  α1[1 : length(α)] = α
  α1[length(α) + 1 : end] = zero(T)

  return reflectorReverseApply!(arnold, α1)
end
@inline getLinearComBasis{T}(arnold::ArnoldiHH{T}, α::StridedVector{T}) = getLinearComBasis!(Vector{T}(size(arnold.G, 1)), arnold, α)

@inline function getBasis{T}(arnold::ArnoldiHH{T}, nrProj::Int = arnold.latestIdx)
  basis = eye(T, size(arnold.G, 1), nrProj)
  for col = 1 : nrProj
    reflectorReverseApply!(arnold, unsafe_view(basis, :, col), nrProj)
  end
  return basis
end

@inline getReducedHessenberg{T}(arnold::ArnoldiHH{T}) = UpperTriangular(view(arnold.G, 1 : arnold.latestIdx - 1, 2 : arnold.latestIdx))

@inline function getHessenberg{T}(arnold::ArnoldiHH{T})
  if arnold.reduce
    error("Hessenberg matrix has been reduced...")
  end
  rows = min(arnold.latestIdx, size(arnold.G, 1))
  return triu(arnold.G[1 : rows, 2 : arnold.latestIdx], -1)
end

@inline function reflectorApply!{T}(arnold::ArnoldiHH{T}, A::StridedVector, nrProj::Int = arnold.latestIdx)
  n = size(arnold.G, 1)
  for reflector = 1 : nrProj
    reflectorApply!(unsafe_view(arnold.G, reflector : n, reflector), arnold.τ[reflector], unsafe_view(A, reflector : n))
  end
  return A
end

@inline function reflectorReverseApply!{T}(arnold::ArnoldiHH{T}, A::StridedVector, nrProj::Int = arnold.latestIdx)
  n = size(arnold.G, 1)
  for reflector = nrProj : -1 : 1
    reflectorApply!(unsafe_view(arnold.G, reflector : n, reflector), arnold.τ[reflector], unsafe_view(A, reflector : n))
  end
  return A
end

@inline function reflectorApply!{T}(arnold::ArnoldiHH{T}, A::StridedMatrix, nrProj::Int = arnold.latestIdx)
  n = size(arnold.G, 1)
  for col = 1 : size(A, 2)
    reflectorApply!(arnold, unsafe_view(A, :, col), nrProj)
  end
  return A
end

# NB copy from julia base, but now for a vector
@inline function reflectorApply!(x::AbstractVector, τ::Number, A::StridedVector)
  m = length(A)

  # dot
  vAj = A[1] + dot(unsafe_view(x, 2 : m), unsafe_view(A, 2 : m))
  vAj = τ' * vAj

  # ger
  A[1] -= vAj
  LinAlg.axpy!(-vAj, unsafe_view(x, 2 : m), unsafe_view(A, 2 : m))

end

function testArnoldiRelation(n, s)
  A = rand(n, n)
  r0 = A * ones(n)

  arnold = ArnoldiHH(A, r0, reduce = false, s = s + 1)

  for it = 1 : s
    expand!(arnold)

    GL = getBasis(arnold, it)
    GR = getBasis(arnold, it + 1)

    H = getHessenberg(arnold)

    # Verify Arnoldi-type relation
    res = A * GL - GR * H
    println("Arnoldi relation residual norm = ", norm(res))

  end
end

function testSolution(n, s)
  A = rand(n, n)
  r0 = A * ones(n)

  arnoldRed = ArnoldiHH(A, r0, reduce = true, s = s + 1)
  arnoldURed = ArnoldiHH(A, r0, reduce = false, s = s + 1)

  for it = 1 : s
    expand!(arnoldRed)
    expand!(arnoldURed)

    println("Reduced:   ", norm(A * getSolution(arnoldRed) - r0))
    println("Red. estim:", arnoldRed.ρ[end])
    println("Unreduced: ", norm(A * getSolution(arnoldURed) - r0))
  end

end
end
