# Iteratively construct the generalized Hessenberg decomposition of A:
#   A * G * U = G * H,
# and approximately solve A * x = b by minimizing the upperbound for
#   ||b - A * G * U * ϕ|| = ||G (e1 * r0 - H * ϕ)|| <= √(j + 1) * ||e1 * r0 - H * ϕ||
# as described in
#
#     Gijzen, Martin B., Gerard LG Sleijpen, and Jens‐Peter M. Zemke.
#     "Flexible and multi‐shift induced dimension reduction algorithms for solving large sparse linear systems."
#     Numerical Linear Algebra with Applications 22.1 (2015): 1-25.
#
function fqmrIDRs(A, b; s = 8, tol = sqrt(eps(real(eltype(b)))), maxIt = size(b, 1), x0 = [], P = Identity(), R0 = [], orthTol = eps(real(eltype(b))), orthSearch = false, kappa = 0.7, orth = "MGS", skewRepeat = 1, orthRepeat = 3, projDim = s)

  if length(R0) > 0 && size(R0) != (length(b), projDim)
    error("size(R0) != [", length(b), ", $s] (User provided shadow residuals are of incorrect size)")
  end
  if projDim > s
    error("Dimension of projector may not exceed that of the orthogonal basis")
  end

  if length(x0) == 0
    x0 = zeros(b)
    r0 = copy(b)
  else
    r0 = b - A * x0
  end

  orthOne = one(real(eltype(b))) / √2

  if orth == "RCGS"
    orthT = RepeatedClassicalGS(orthOne, orthTol, orthRepeat)
  elseif orth == "CGS"
    orthT = ClassicalGS()
  elseif orth == "MGS"
    orthT = ModifiedGS()
  end

  if skewRepeat == 1
    skewT = SingleSkew()
  else
    skewT = RepeatSkew(orthOne, orthTol, skewRepeat)
  end
  rho0 = vecnorm(r0)
  scale!(r0, 1.0 / rho0)
  hessenberg = Hessenberg(size(b, 1), s, eltype(b), rho0)
  idrSpace = FQMRSpace(A, P, r0, orthT, size(b, 1), s, eltype(b))
  idrSpace.W[:, 1] = 0.0
  idrSpace.G[:, 1] = r0
  solution = Solution(x0, rho0, tol)
  projector = Projector(size(b, 1), projDim, R0, kappa, orthSearch, skewT, eltype(b))

  return IDRMethod(solution, idrSpace, hessenberg, projector, maxIt)

end
