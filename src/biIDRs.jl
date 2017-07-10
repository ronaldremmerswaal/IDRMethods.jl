# function fqmrIDRs(A, b; s = 8, tol = sqrt(eps(real(eltype(b)))), maxIt = size(b, 1), x0 = [], P = Identity(), R0 = [], kappa = 0.7)
#
#   if length(R0) > 0 && size(R0) != (length(b), projDim)
#     error("size(R0) != [", length(b), ", $s] (User provided shadow residuals are of incorrect size)")
#   end
#   if projDim > s
#     error("Dimension of projector may not exceed that of the orthogonal basis")
#   end
#
#   if length(x0) == 0
#     x0 = zeros(b)
#     r0 = copy(b)
#   else
#     r0 = b - A * x0
#   end
#
#   rho0 = vecnorm(r0)
#   scale!(r0, 1.0 / rho0)
#
#   hessenberg = Hessenberg(size(b, 1), s, eltype(b), rho0)
#   idrSpace = BiSpace(A, P, r0, size(b, 1), s, eltype(b))
#   idrSpace.W[:, 1] = 0.0
#   idrSpace.G[:, 1] = r0
#   solution = Solution(x0, rho0, tol)
#   projector = Projector(size(b, 1), projDim, R0, kappa, orthSearch, skewT, eltype(b))
#
#   return IDRMethod(solution, idrSpace, hessenberg, projector, maxIt)
#
# end
