module IDRMethods

export fqmrIDRs, biIDRs

using Base.BLAS
using Base.LinAlg

include("harmenView.jl")
include("common.jl")
include("fqmrIDRs.jl")
include("biIDRs.jl")

function IDRMethod{T}(solution::Solution{T}, idrSpace::IDRSpace{T}, projector::Projector{T}, maxIt)

  iter = 0
  while true
    iter += 1

    # Project g along G orthogonal to R0: v = (I - G (R0'G)^-1 R0') g
    update!(projector, idrSpace)
    apply!(projector, idrSpace)

    # Compute g = A * v
    expand!(idrSpace, projector)

    # Compute t = (A - μ I) v
    mapToIDRSpace!(idrSpace, projector)

    # Update basis of G: g = (I - G G') t; g <- g / |g|
    update!(idrSpace, projector)

    update!(solution, idrSpace, projector)
    if isConverged(solution) || iter == maxIt
      return solution.x, solution.ρ
    end
  end
end

end
