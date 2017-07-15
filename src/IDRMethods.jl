module IDRMethods

export fqmrIDRs, biIDRs

using Base.BLAS
using Base.LinAlg

abstract type Projector end
abstract type IDRSpace end
abstract type Solution end

include("harmenView.jl")
include("common.jl")
include("fqmrIDRs.jl")
include("biIDRs.jl")

function IDRMethod(solution::Solution, idrSpace::IDRSpace, projector::Projector, maxIt)

  iter = 0
  s = idrSpace.s
  while true
    for k in 1 : s + 1
      iter += 1

      # Compute u, v: the skew-projection of g along G orthogonal to R0 (for which v = (I - G * inv(M) * R0) * g)
      update!(projector, idrSpace)
      apply!(projector, idrSpace)

      # Compute g = A * v
      expand!(idrSpace, projector)

      if k == s + 1
        nextIDRSpace!(projector, idrSpace)
      end
      # Compute t = (A - μ * I) * g
      mapToIDRSpace!(idrSpace, projector)

      # Update basis of G
      update!(idrSpace, projector, k, iter)

      update!(solution, idrSpace, projector)
      if isConverged(solution) || iter == maxIt
        return solution.x, solution.ρ
      end
    end
  end
end

end
