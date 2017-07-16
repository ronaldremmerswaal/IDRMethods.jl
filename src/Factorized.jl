# For a given square matrix M this stores it's LU factorization, and allows for cheap column updates of M without recomputing the LU factorization by making use of the Sherman-Woodburry-Morrison update formula

module Factorized

export LUFactorized, replaceColumn!, A_ldiv_B!

import Base.A_ldiv_B!

using Base.LinAlg

include("harmenView.jl")

type LUFactorized{T}
  M::StridedMatrix{T}
  lu

  U::StridedMatrix{T}

  maxNrUpdates::Int

  nrUpdates::Int
  updateIdxToCol::Vector{Int}

end
LUFactorized{T}(M::StridedMatrix{T}, maxNrUpdates) = LUFactorized{T}(M, lufact(M), Matrix{T}(size(M, 1), maxNrUpdates), maxNrUpdates, 0, Vector{Int64}(maxNrUpdates))

# Replaces the idx-th column of M with m, where u = M \ m is provided
function replaceColumn!{T}(F::LUFactorized{T}, colIdx::Int, m, u)
  F.M[:, colIdx] = m

  if F.nrUpdates == F.maxNrUpdates
    F.lu = lufact(F.M)
    F.nrUpdates = 0
    return
  end

  F.nrUpdates += 1
  F.U[:, F.nrUpdates] = u
  F.U[colIdx, F.nrUpdates] -= 1.
  scale!(unsafe_view(F.U, :, F.nrUpdates), 1. / u[colIdx])
  F.updateIdxToCol[F.nrUpdates] = colIdx

end

function replaceColumn!{T}(F::LUFactorized{T}, colIdx::Int, m)
  u = copy(m)
  A_ldiv_B!(u, F, m)
  replaceColumn!(F, colIdx, m, u)
end

function A_ldiv_B!{T}(x, F::LUFactorized{T}, b)
  A_ldiv_B!(x, F.lu, b)

  for idx = 1 : F.nrUpdates
    axpy!(-x[F.updateIdxToCol[idx]], unsafe_view(F.U, :, idx), x)
  end
end

function test(s, nrReplace)
  M = rand(s, s)

  F = LUFactorized(copy(M), nrReplace)

  m = zeros(s)


  # Add some random columns
  for idx = 1 : nrReplace
    b = rand(s)

    # Recompute lu
    M[:, idx] = b
    lu = lufact(M)

    # Reuse factorization
    replaceColumn!(F, idx, b)

    # Test by solving for other random b
    b = rand(s)

    xlu = copy(b)
    A_ldiv_B!(xlu, lu, b)
    xfac = copy(b)
    A_ldiv_B!(xfac, F, b)

    println("Relative error w.r.t. lu = ", vecnorm(xlu - xfac) / vecnorm(xlu))

  end

end

end
