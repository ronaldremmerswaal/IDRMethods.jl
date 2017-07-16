module ArnoldiBasis

export ArnoldiHH, expand!, getColumn, getLinearComBasis, getBasis, getSolution, getSolution!, getReducedHessenberg

include("harmenView.jl")

# Iteratively builds the Arnoldi basis using Householder reflections; without storing the Hessenberg matrix.
type ArnoldiHH{T}
  A
  G::StridedMatrix{T}
  τ::StridedVector{T}
  latestIdx::Int

  Qe1::StridedVector{T}   # Equals Q' * e1 * ρ0
end

function ArnoldiHH{T}(s::Int, A, r0::StridedVector{T})
  G = Matrix{T}(length(r0), s)
  G[:, 1] = r0

  τ = Vector{T}(s)
  τ[1] = LinAlg.reflector!(unsafe_view(G, :, 1))

  Qe1 = zeros(T, s)
  Qe1[1] = G[1, 1]

  ArnoldiHH{T}(A, G, τ, 1, Qe1)
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

  # New Givens for Hessenberg matrix (is applied in-place)
  givensRot, arnold.G[arnold.latestIdx - 1, arnold.latestIdx] = LinAlg.givens(arnold.G, arnold.latestIdx - 1, arnold.latestIdx, arnold.latestIdx)
  # arnold.G[arnold.latestIdx, arnold.latestIdx] = zero(T)

  # Apply Givens to rhs of small system (Qe1)
  arnold.Qe1 = givensRot * arnold.Qe1
end

@inline getSolution{T}(arnold::ArnoldiHH{T}) = getSolution!(Vector{T}(size(arnold.G, 1)), arnold)
function getSolution!{T}(sol::StridedVector{T}, arnold::ArnoldiHH{T})
  sol[1 : arnold.latestIdx - 1] = getReducedHessenberg(arnold) \ unsafe_view(arnold.Qe1, 1 : arnold.latestIdx - 1)
  sol[arnold.latestIdx : end] = zero(T)

  return reflectorApply!(arnold, sol, arnold.latestIdx - 1)
end

@inline function getColumn!{T}(col::StridedVector{T}, arnold::ArnoldiHH{T}, colIdx::Int)
  col[:] = zero(T)
  col[colIdx] = one(T)
  return reflectorApply!(arnold, col)
end
@inline getColumn{T}(arnold::ArnoldiHH{T}, colIdx::Int) = getColumn!(Vector{T}(size(arnold.G, 1)), arnold, colIdx)

function getLinearComBasis!{T}(α1::StridedVector{T}, arnold::ArnoldiHH{T}, α::StridedVector{T})
  α1[1 : length(α)] = α
  α1[length(α) + 1 : end] = zero(T)

  return reflectorApply!(arnold, α1)
end
@inline getLinearComBasis{T}(arnold::ArnoldiHH{T}, α::StridedVector{T}) = getLinearComBasis!(Vector{T}(size(arnold.G, 1)), arnold, α)

@inline function getBasis{T}(arnold::ArnoldiHH{T})
  basis = eye(T, size(arnold.G, 1), arnold.latestIdx)
  for col = 1 : arnold.latestIdx
    reflectorApply!(arnold, unsafe_view(basis, :, col))
  end
  return basis
end

@inline getReducedHessenberg{T}(arnold::ArnoldiHH{T}) = UpperTriangular(view(arnold.G, 1 : arnold.latestIdx - 1, 2 : arnold.latestIdx))

@inline function reflectorApply!{T}(arnold::ArnoldiHH{T}, A::StridedVector, nrProj::Int = arnold.latestIdx)
  n = size(arnold.G, 1)
  for reflector = 1 : nrProj
    reflectorApply!(unsafe_view(arnold.G, reflector : n, reflector), arnold.τ[reflector], unsafe_view(A, reflector : n))
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

end
