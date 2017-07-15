# An updateable banded Hessenberg matrix, for which we only require the Moore-Penrose pseudoinverse to be applicable to the first basis vector e_1.
# Hence we solve
#   ϕ = argmin_α ||H α - e_1 ρ0||,
# we denote the resulting residual by φ
#   φ = ||H ϕ - e_1 ρ0||
# This allows for short recurrences using either Givens rotations or Householder reflections.

module BandedHessenberg

include("harmenView.jl")
abstract type AbstractBandedHessenberg{T} end

export GivensBandedHessenberg, addColumn!

# type FullBandedHessenberg <: AbstractBandedHessenberg end

# Givens
type GivensBandedHessenberg{T} <: AbstractBandedHessenberg{T}
  bandWidth::Int
  givensRotations::Vector{LinAlg.Givens{T}}
  ϕ::T
  φ::T  # The residual norm of the appxoximate solution

  nrCols::Int

end

GivensBandedHessenberg{T}(bandWidth, rho0::T) = GivensBandedHessenberg{T}(bandWidth, Vector{LinAlg.Givens{T}}(bandWidth + 1), zero(T), rho0, 0)

# type HHBandedHessenberg <: AbstractBandedHessenberg end

# NB modifies r as well
function addColumn!{T}(H::GivensBandedHessenberg{T}, r::Vector{T})
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

end
