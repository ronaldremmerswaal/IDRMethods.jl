# An updateable banded Hessenberg matrix, for which we only require the Moore-Penrose pseudoinverse to be applicable to the first basis vector e_1.
# Hence we solve
#   ϕ = argmin_α ||H α - e_1 ρ0||,
# we denote the resulting residual by φ
#   φ = ||H ϕ - e_1 ρ0||
# This allows for short recurrences using either Givens rotations or Householder reflections.

abstract type AbstractBandedHessenberg{T} end

export GivensBandedHessenberg, HHBandedHessenberg, addColumn!, apply!

# type FullBandedHessenberg <: AbstractBandedHessenberg end

# Householder
type HHBandedHessenberg{T} <: AbstractBandedHessenberg{T}
  bandWidth::Int
  q::StridedVector{T}
  ϕ::T
  φ::T
  τ::StridedVector{T}

  nrCols

end
HHBandedHessenberg{T}(bandWidth, rho0::T) = HHBandedHessenberg{T}(bandWidth, Vector{T}(bandWidth + 1), zero(T), rho0, Vector{T}(bandWidth + 1), 0)

function addColumn!{T}(H::HHBandedHessenberg{T}, r::StridedVector{T})
  H.q[1 : end - 1] = unsafe_view(H.q, 2 : H.bandWidth + 1)
  H.τ[1 : end - 1] = unsafe_view(H.τ, 2 : H.bandWidth + 1)

  startIdx::Int64 = max(1, H.bandWidth + 1 - H.nrCols)
  applyProjections!(unsafe_view(r, startIdx : H.bandWidth + 1), unsafe_view(H.q, startIdx : H.bandWidth), unsafe_view(H.τ, startIdx : H.bandWidth))

  H.τ[end] = conj(LinAlg.reflector!(unsafe_view(r, H.bandWidth + 1 : H.bandWidth + 2)))
  H.q[end] = r[end]

  H.ϕ = (1. - H.τ[end]) * H.φ     # Solution update
  H.φ = -H.τ[end] * conj(H.q[end]) * H.φ        # Residual norm update

  H.nrCols += 1
end


# Givens
type GivensBandedHessenberg{T} <: AbstractBandedHessenberg{T}
  bandWidth::Int
  givensRotations::Vector{LinAlg.Givens{T}}
  ϕ::T
  φ::T

  nrCols::Int

end
GivensBandedHessenberg{T}(bandWidth, rho0::T) = GivensBandedHessenberg{T}(bandWidth, Vector{LinAlg.Givens{T}}(bandWidth + 1), zero(T), rho0, 0)

function applyProjections!{T}(r::StridedVector{T}, q::StridedVector{T}, τ::StridedVector{T})
  for (l, qVal) = enumerate(q)
    wDotR = τ[l] * (r[l] + conj(qVal) * r[l + 1])
    r[l] -= wDotR
    r[l + 1] -= qVal * wDotR
  end
end

# NB modifies r as well (not clear from fcn name...)
function addColumn!{T}(H::GivensBandedHessenberg{T}, r::StridedVector{T})
  H.givensRotations[1 : end - 1] = unsafe_view(H.givensRotations, 2 : H.bandWidth + 1)

  startIdx = max(1, H.bandWidth + 1 - H.nrCols)
  for l = startIdx : H.bandWidth
    r[l : l + 1] = H.givensRotations[l] * unsafe_view(r, l : l + 1)
  end

  H.givensRotations[end], r[end - 1] = givens(r[end - 1], r[end], 1, 2)

  H.ϕ = H.givensRotations[end].c * H.φ
  H.φ = -conj(H.givensRotations[end].s) * H.φ

  H.nrCols += 1
end
