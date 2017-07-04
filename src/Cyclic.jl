# A column-wise cyclic 2D array (matrix), with variable starting column index
module Cyclic

export CyclicMatrix

import Base

type CyclicMatrix{T} <: AbstractArray{T, 2}
  data::Array{T, 2}

  CyclicMatrix{T}(m, n, start = 1) where T = new(Array{T, 2}(m, n), m, n, start)
  CyclicMatrix{T}(data::Array{T, 2}, start = 1) where T = new(data, size(data, 1), size(data, 2), start)

  m::Int
  n::Int
  start::Int        # First column index
end

mapToDataCol(C::CyclicMatrix, j) = mod.(C.start + j - 2, C.n) + 1
mapToDataLin(C::CyclicMatrix, l) = mod.(l + (C.start - 1) * C.m - 1, C.m * C.n) + 1

# Iteration
Base.start(::CyclicMatrix) = mod.((start - 1), n) * m + 1
Base.next(C::CyclicMatrix, state) = (C.data[mod.(state - 1, m * n) + 1], state + 1)
Base.done(C::CyclicMatrix, state) = state > length(C.data)
Base.eltype(::Type{CyclicMatrix}) = eltype(C.data)
Base.length(C::CyclicMatrix) = length(C.data)

# Indexing
Base.size(C::CyclicMatrix) = size(C.data);
Base.size(C::CyclicMatrix, i) = size(C.data, i);
Base.getindex(C::CyclicMatrix, l) = C.data[mapToDataLin(C, l)];
Base.getindex(C::CyclicMatrix, i, j) = C.data[i, mapToDataCol(C, j)];
Base.setindex!(C::CyclicMatrix, v, l) = Base.setindex!(C.data, v, mapToDataLin(C, l));
Base.setindex!(C::CyclicMatrix, v, i, j) = Base.setindex!(C.data, v, i, mapToDataCol(C, j));
Base.endof(C::CyclicMatrix) = endof(C.data);

Base.:*(C::CyclicMatrix, x) = C.data * x[mod.(1 - C.start : C.n - C.start, C.n) + 1]

# BLAS operations
# TODO implement only gemv!, and use similar etc to initialize zeros array
BLAS.gemv!(tA, alpha, C::CyclicMatrix, x, beta, y) = BLAS.gemv!(tA, alpha, C.data, x[mod.(1 - C.start : C.n - C.start, C.n) + 1], beta, y)

function BLAS.gemv(tA, C::CyclicMatrix, x)
  if tA == 'N'
    return BLAS.gemv(tA, C.data, x[mod.(1 - C.start : C.n - C.start, C.n) + 1]);
  else
    alpha = BLAS.gemv(tA, C.data, x);
    return alpha[mapToDataCol(C, 1 : C.n)];
  end
end
# BLAS.gemv(tA, alpha, C::CyclicMatrix, x) = BLAS.gemv(tA, alpha, C.data, x[mod.(1 - C.start : C.n - C.start, C.n) + 1])

end
