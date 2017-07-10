# For a given square matrix M this stores it's factorization, and allows for cheap collumn updates where the factorization is updated using the Sherman-Woodburry formula.

import Base.A_ldiv_B!

type Factorized
  M
  lu

  U

  maxNrUpdates

  nrUpdates
  updateIdxToCol

  Factorized(M, maxNrUpdates) = new(M, lufact(M), Matrix{eltype(M)}(size(M, 1), maxNrUpdates), maxNrUpdates, 0, Vector{Int64}(maxNrUpdates))
end

# Replaces the idx-th column of M with m, where u = M \ m is provided
function replaceColumn!(F::Factorized, colIdx::Int, m, u)
  copy!(F.M[:, colIdx], m)

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

function replaceColumn!(F::Factorized, colIdx::Int, m)
  u = copy(m)
  A_ldiv_B!(u, F, m)
  replaceColumn!(F, colIdx, m, u)
end

function A_ldiv_B!(x, F::Factorized, b)
  A_ldiv_B!(x, F.lu, b)

  for idx = 1 : F.nrUpdates
    axpy!(-x[F.updateIdxToCol[idx]], unsafe_view(F.U, :, idx), x)
  end
end
