module IDRMethods

export fqmrIDRs, biIDRs

using Base.BLAS
using Base.LinAlg

type Projector
  n
  s
  j
  μ
  M
  m
  α
  R0
  u
  κ
  orthSearch
  skewT

  latestIdx
  oldestIdx   # Index in G corresponding to oldest column in M
  gToMIdx     # Maps from G idx to M idx

  lu

  Projector(n, s, R0, κ, orthSearch, skewT, T) = new(n, s, 0, zero(T), [], zeros(T, s), zeros(T, s), R0, zeros(T, s), κ, orthSearch, skewT, 0, 0, [], [])
end

type Hessenberg
  n
  s
  r
  cosine
  sine
  ϕ
  φ

  Hessenberg(n, s, T, rho0) = new(n, s, zeros(T, s + 3), zeros(T, s + 2), zeros(T, s + 2), zero(T), rho0)
end

abstract type IDRSpace end

type Solution
  x
  ρ
  rho0
  tol

  Solution(x, ρ, tol) = new(x, [ρ], ρ, tol)
end

include("harmenView.jl")
include("common.jl")
include("factorized.jl")
include("fqmrIDRs.jl")
include("biIDRs.jl")

function IDRMethod(solution::Solution, idrSpace::IDRSpace, hessenberg::Hessenberg, projector::Projector, maxIt)

  iter = 0
  s = idrSpace.s
  while true
    for k in 1 : s + 1
      iter += 1

      if iter > s
        # Compute u, v: the skew-projection of g along G orthogonal to R0 (for which v = (I - G * inv(M) * R0) * g)
        update!(projector, idrSpace)
        apply!(projector, idrSpace)
      end

      # Compute g = A * v
      expand!(idrSpace, projector)

      if k == s + 1
        nextIDRSpace!(projector, idrSpace)
      end
      # Compute t = (A - μ * I) * g
      mapToIDRSpace!(idrSpace, projector)

      # Update basis of G
      updateG!(idrSpace, hessenberg, k)
      update!(hessenberg, projector, iter)
      updateW!(idrSpace, hessenberg, k, iter)

      # Update x <- x + Q(1, end) * w
      update!(solution, idrSpace, hessenberg, projector, k)
      if isConverged(solution) || iter == maxIt
        return solution.x, solution.ρ
      end
    end
  end

end



end
