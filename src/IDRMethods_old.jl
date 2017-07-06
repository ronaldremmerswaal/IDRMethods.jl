module IDRMethods_old

export fqmrIDRs

using SugarBLAS
import Base.*

function fqmrIDRs(A, b; s::Integer = 8, tol = 1E-6, maxIt::Integer = size(b, 1))

  n = size(b, 1)
  s = min(n, s)

  # Allocate memory (size n)
  g = copy(b)
  x = zeros(b)
  w = zeros(b)
  v = zeros(b)

  # (size s)
  cosine = zeros(eltype(b), s + 2)
  sine = zeros(eltype(b), s + 2)
  u = zeros(eltype(b), s + 2)
  h = zeros(eltype(b), s + 2)
  r = zeros(eltype(b), s + 3)
  m = zeros(eltype(b), s)
  M = Array{eltype(b)}(s, s)

  # (size n x s)
  G = Array{eltype(b)}(n, s);
  W = zeros(eltype(b), n, s + 1);
  R0 = Array{eltype(b)}(n, s)

  j = 0
  mu = 0.
  iter = 0
  stopped = false
  rho = norm(g)
  tol *= rho
  phi = 0.
  phihat = rho
  resHist = zeros(maxIt + 1)
  resHist[1] = rho

  @blas! g *= 1 / rho

  # Column permutations
  permG = [1 : s...]

  while !stopped
    @inbounds for k in 1 : s + 1
      # Generate s vectors in G_j
      iter += 1
      u[1 : s + 2] = 0.
      u[s + 1] = 1.

      if iter == s + 1
        # Compute R0 only when needed, hence if we are about to enter j = 1
        R0 = qr(rand(eltype(b), n, s))[1]
        M = BLAS.gemm('C', 'N', 1.0, R0, G);
      end

      v = copy(g)
      if iter > s
        # Project orthogonal to R0
        @blas! m = R0' * g
        u[1 : s] = M \ m
        v -= G * u[1 : s]
        u[1 : s] = -u[permG]
        M[:, permG[1]] = m
      end
      # Permute the columns
      cosine[1 : s + 1] = cosine[2 : s + 2]
      sine[1 : s + 1] = sine[2 : s + 2]

      pGEnd = permG[1]
      permG[1 : s - 1] = permG[2 : s]
      permG[s] = pGEnd

      # Add new vectors
      G[:, permG[s]] = g

      # TODO preconditioner
      vhat = v
      g = A * vhat

      if k == s + 1
        j += 1
        # Compute mu for new space G_j
        mu = computeMu(g, v)
      end
      if j > 0
        @blas! g -= mu * v
      end
      @blas! h = mu * u

      # Orthogonalize g with current vectors in G_j
      orthogonalize!(G, g, h, s, k, permG)

      # Update the QR factorization of the Hessenberg matrix
      r[1] = 0.
      r[2 : s + 3] = h
      applyGivensRot!(r, sine, cosine, iter, s)

      # Update the solution
      phi = cosine[s + 2] * phihat
      phihat = -conj(sine[s + 2]) * phihat
      if iter > s
        vhat -= W * r[[s + 2 - k : s + 1; 1 : s + 1 - k]]
      else
        vhat -= view(W, :, 1 : k) * r[s + 2 - k : s + 1]
      end
      W[:, k > s ? 1 : k + 1] = vhat / r[s + 2]
      x += phi * W[:, k > s ? 1 : k + 1]

      # Compute an upperbound for the residual norm
      rho = abs(phihat) * sqrt(j + 1.)
      resHist[iter + 1] = rho
      if rho < tol || iter > maxIt
        stopped = true
        break
      end
    end
  end

  return x, resHist[1 : iter + 1]
end

function orthogonalize!(G::Matrix, g, h, s, k, permG)
  if k < s + 1
    alpha = Array{eltype(g)}(k)
    @inbounds for l in s - k + 1 : s
      alpha[k - s + l] = vecdot(view(G, :, permG[l]), g)
      @blas! g -= alpha[k - s + l] * view(G, :, permG[l])
    end
    h[s + 1 - k + 1 : s + 1] += alpha
  end
  h[s + 2] = norm(g)
  @blas! g *= 1 / h[s + 2]
end

function applyGivensRot!(r, sine, cosine, iter, s)
  @inbounds for l = max(1, s + 3 - iter) : s + 1
    oldRl = r[l]
    r[l] = cosine[l] * oldRl + sine[l] * r[l + 1]
    r[l + 1] = -conj(sine[l]) * oldRl + cosine[l] * r[l + 1]
  end

  a = r[s + 2]
  b = r[s + 3]
  if abs(a) < eps()
    sine[s + 2] = 1.
    cosine[s + 2] = 0.
    r[s + 2] = b
  else
    t = abs(a) + abs(b)
    rho = t * sqrt(abs(a / t) ^ 2 + abs(b / t) ^ 2)
    alpha = a / abs(a)

    sine[s + 2] = alpha * conj(b) / rho
    cosine[s + 2] = abs(a) / rho
    r[s + 2] = alpha * rho
  end
end

@inline function computeMu(t, v)
  # Compute residual minimizing mu
  kappa = 0.7
  tv = vecdot(t, v)
  tt = vecdot(t, t)

  omega = tv / tt
  rho = tv / (sqrt(tt) * norm(v))
  if abs(rho) < kappa
    omega *= kappa / abs(rho)
  end
  return abs(omega) > eps() ? 1. / omega : 1.
end

end
