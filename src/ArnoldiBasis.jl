module ArnoldiBasis

export ArnoldiHH, expand!, getSolution, getSolution!, getBasis

include("harmenView.jl")

# Iteratively builds the Arnoldi basis using Householder reflections; without storing the Hessenberg matrix.
type ArnoldiHH{T}
  A
  G::StridedMatrix{T}
  τ::StridedVector{T}
  latestIdx::Int

  Qe1::StridedVector{T}   # Equals Q' * e1 * ρ0
  reduce::Bool
  x0::StridedVector{T}
  ρ::Vector{T}
  givensRot::Vector{LinAlg.Givens{T}}
end

function ArnoldiHH{T}(A, r0::StridedVector{T}; s::Int = size(A, 1), reduce::Bool = true, x0 = zeros(T, size(A, 1)))
  G = Matrix{T}(length(r0), s)
  G[:, 1] = r0

  τ = Vector{T}(s)
  τ[1] = LinAlg.reflector!(unsafe_view(G, :, 1))

  Qe1 = zeros(T, s)
  Qe1[1] = G[1, 1]

  ArnoldiHH{T}(A, G, τ, 1, Qe1, reduce, x0, [abs(Qe1[1])], [])
end

function expand!{T}(arnold::ArnoldiHH{T})
  if arnold.latestIdx == size(arnold.G, 2)
    error("Maxiumum size reached")
  end
  # MV
  A_mul_B!(unsafe_view(arnold.G, :, arnold.latestIdx + 1), arnold.A, getColumn(arnold, arnold.latestIdx))

  # Apply previous reflections
  reflectorApply!(arnold, unsafe_view(arnold.G, :, arnold.latestIdx + 1))
  arnold.latestIdx += 1

  # New reflection
  arnold.τ[arnold.latestIdx] = LinAlg.reflector!(unsafe_view(arnold.G, arnold.latestIdx : size(arnold.G, 1), arnold.latestIdx))

  if arnold.reduce
    # Apply previous Givens new column
    for φ ∈ arnold.givensRot
      arnold.G[:, arnold.latestIdx] = φ * unsafe_view(arnold.G, :, arnold.latestIdx)
    end

    if arnold.latestIdx <= size(arnold.G, 1)
      # New Givens for Hessenberg matrix
      φ, arnold.G[arnold.latestIdx - 1, arnold.latestIdx] = LinAlg.givens(arnold.G, arnold.latestIdx - 1, arnold.latestIdx, arnold.latestIdx)
      push!(arnold.givensRot, φ)

      # Apply Givens to rhs of small system (Qe1)
      arnold.Qe1 = φ * arnold.Qe1
      push!(arnold.ρ, arnold.Qe1[arnold.latestIdx])
    end

  end
end

@inline getSolution{T}(arnold::ArnoldiHH{T}) = getSolution!(Vector{T}(size(arnold.G, 1)), arnold)
function getSolution!{T}(sol::StridedVector{T}, arnold::ArnoldiHH{T})
  if arnold.reduce
    R = getReducedHessenberg(arnold)
    Qe1 = unsafe_view(arnold.Qe1, 1 : arnold.latestIdx - 1)
  else
    H = getHessenberg(arnold)
    Q, R = qr(H)
    Qe1 = Q' * arnold.Qe1[1 : min(arnold.latestIdx, size(arnold.G, 1))]
  end

  return arnold.x0 + getLinearComBasisLagged(arnold, R \ Qe1)
end

@inline function getColumn!{T}(col::StridedVector{T}, arnold::ArnoldiHH{T}, colIdx::Int)
  col[:] = zero(T)
  col[colIdx] = one(T)
  return reflectorApply!(arnold, col, colIdx)
end
@inline getColumn{T}(arnold::ArnoldiHH{T}, colIdx::Int) = getColumn!(Vector{T}(size(arnold.G, 1)), arnold, colIdx)

# Returns β = Qn * e1 * α1 + Qn * e2 * α2 + ...
function getLinearComBasis!{T}(α1::StridedVector{T}, arnold::ArnoldiHH{T}, α::StridedVector{T})
  α1[1 : length(α)] = α
  α1[length(α) + 1 : end] = zero(T)

  return reflectorApply!(arnold, α1)
end
@inline getLinearComBasis{T}(arnold::ArnoldiHH{T}, α::StridedVector{T}) = getLinearComBasis!(Vector{T}(size(arnold.G, 1)), arnold, α)

# Returns β = Q1 * e1 * α1 + Q2 * e2 * α2 + ...
function getLinearComBasisLagged!{T}(β::StridedVector{T}, arnold::ArnoldiHH{T}, α::StridedVector{T})
  LinAlg.gemv!('N', one(T), reflectorApplyLagged!(arnold, eye(T, size(arnold.G, 1), length(α))), α, zero(T), β)
  return β
end
@inline getLinearComBasisLagged{T}(arnold::ArnoldiHH{T}, α::StridedVector{T}) = getLinearComBasisLagged!(Vector{T}(size(arnold.G, 1)), arnold, α)

@inline function getBasis{T}(arnold::ArnoldiHH{T})
  basis = eye(T, size(arnold.G, 1), arnold.latestIdx)
  for col = 1 : arnold.latestIdx
    reflectorApply!(arnold, unsafe_view(basis, :, col), col)
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

@inline function reflectorApply!{T}(arnold::ArnoldiHH{T}, A::StridedMatrix, nrProj::Int = arnold.latestIdx)
  n = size(arnold.G, 1)
  for col = 1 : size(A, 2)
    reflectorApply!(arnold, unsafe_view(A, :, col), nrProj)
  end
  return A
end

# Returns [Q1 e1 Q2 e2, ..., ] * A
@inline function reflectorApplyLagged!{T}(arnold::ArnoldiHH{T}, A::StridedMatrix)
  n = size(arnold.G, 1)
  for col = 1 : size(A, 2)
    reflectorApply!(arnold, unsafe_view(A, :, col), col)
  end
  return A
end

# NB copy from julia base, but now for a vector
@inline function reflectorApply!(x::AbstractVector, τ::Number, A::StridedVector)
  m = length(A)
  @inbounds begin
    # dot
    vAj = A[1]
    for i = 2:m
        vAj += x[i]'*A[i]
    end

    vAj = τ'*vAj

    # ger
    A[1] -= vAj
    for i = 2:m
        A[i] -= x[i]*vAj
    end
  end
end

function testArnoldiRelation(n, s)
  A = rand(n, n)
  r0 = A * ones(n)

  arnold = ArnoldiHH(A, r0, reduce = false, s = s + 1)

  for it = 1 : s
    expand!(arnold)

    GL = reflectorApplyLagged!(arnold, eye(n, it))
    GR = reflectorApply!(arnold, eye(n, n), it + 1)

    H = zeros(n, it)
    H[1 : it + 1, 1 : it] = getHessenberg(arnold)

    # Verify Arnoldi-type relation
    res = A * GL - GR' * H
    println("Arnoldi relation residual norm = ", vecnorm(res))

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

    println("Reduced:   ", vecnorm(A * getSolution(arnoldRed) - r0))
    println("Unreduced: ", vecnorm(A * getSolution(arnoldURed) - r0))
  end

end

end
