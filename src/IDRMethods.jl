module IDRMethods

export fqmrIDRs

using SugarBLAS
import Base.*

function fqmrIDRs{T}(A, b::AbstractArray{T, 1}; s::Integer = 8, tol::AbstractFloat = 1E-6, maxIt::Integer = size(b, 1))

  n = size(b, 1);
  s = min(n, s);

  # Allocate memory (size n)
  g = copy(b);
  x = zeros(b);
  w = zeros(b);
  v = zeros(b);

  # (size s)
  cosine = zeros(T, s + 2);
  sine = zeros(T, s + 2);
  u = zeros(T, s + 2);
  h = zeros(T, s + 2);
  r = zeros(T, s + 3);
  m = zeros(T, s);
  M = Array{T}(s, s);

  # (size n x s)
  G = Vector{Vector{T}}(s);
  W = Vector{Vector{T}}(s + 1);
  R0 = Array{T}(n, s);

  j = 0;
  mu = 0.;
  iter = 0;
  stopped = false;
  rho = norm(g);
  tol *= rho;
  phi = 0.;
  phihat = rho;
  resHist = zeros(T, maxIt + 1);
  resHist[1] = rho;

  @blas! g *= 1 / rho;

  # Column permutations
  permG = [1 : s...];

  while !stopped
    @inbounds for k in 1 : s + 1
      # Generate s vectors in G_j
      iter += 1;
      u[1 : s + 2] = 0.;
      u[s + 1] = 1.;

      if iter == s + 1
        # Compute R0 only when needed, hence if we are about to enter j = 1
        R0 = qr(rand(T, n, s))[1];
        M = innerProducts(R0, G);
      end

      if iter <= s
        # Initialization: construct Arnoldi basis
        v = copy(g);
      else
        # Project orthogonal to R0
        @blas! m = R0' * g;
        gamma = M \ m;
        v = g - G * gamma;
        u[1 : s] = -gamma[permG];
      end
      # Permute the columns
      cosine[1 : s + 1] = cosine[2 : s + 2];
      sine[1 : s + 1] = sine[2 : s + 2];

      pGEnd = permG[1];
      permG[1 : s - 1] = permG[2 : s];
      permG[s] = pGEnd;

      # Add new vectors
      G[permG[end]] = copy(g);
      W[k] = copy(w);
      if iter > s
        M[:, permG[end]] = m;
      end

      # TODO preconditioner
      vhat = v;
      @blas! g = A * vhat;

      if k == s + 1
        j += 1;
        # Compute mu for new space G_j
        mu = computeMu(g, v);
      end
      if j > 0
        @blas! g -= mu * v;
      end
      @blas! h = mu * u;

      # Orthogonalize g with current vectors in G_j
      orthogonalize!(G, g, h, s, k, permG);

      # Update the QR factorization of the Hessenberg matrix
      r[1] = 0.;
      r[2 : s + 3] = h;
      applyGivensRot!(r, sine, cosine, iter, s);

      # Update the solution
      phi = cosine[s + 2] * phihat;
      phihat = -conj(sine[s + 2]) * phihat;
      if iter > s
        w = vhat - W * r[[s + 2 - k : s + 1; 1 : s + 1 - k]];
      else
        w = vhat - W * r[s + 2 - k : s + 1];
      end
      @blas! w *= 1. / r[s + 2];
      @blas! x += phi * w;

      # Compute an upperbound for the residual norm
      rho = abs(phihat) * sqrt(j + 1.);
      resHist[iter + 1] = rho;
      if rho < tol || iter > maxIt
        stopped = true;
        break;
      end
    end
  end

  return x, resHist[1 : iter + 1]
end

function orthogonalize!{T}(G::Vector{Vector{T}}, g::AbstractArray{T, 1}, h::AbstractArray{T, 1}, s, k, permG)
  if k < s + 1
    alpha = Array{T}(k);
    for l in s - k + 1 : s
      alpha[k - s + l] = vecdot(G[permG[l]], g);
      @blas! g -= alpha[k - s + l] * G[permG[l]];
    end
    h[s + 1 - k + 1 : s + 1] += alpha;  # TODO doesn't work with @blas!
  end
  h[s + 2] = norm(g);
  @blas! g *= 1 / h[s + 2];
end

function applyGivensRot!(r, sine, cosine, iter, s)
  @fastmath @inbounds for l = max(1, s + 3 - iter) : s + 1
    oldRl = r[l];
    r[l] = cosine[l] * oldRl + sine[l] * r[l + 1];
    r[l + 1] = -conj(sine[l]) * oldRl + cosine[l] * r[l + 1];
  end

  a = r[s + 2];
  b = r[s + 3];
  if abs(a) < eps()
    sine[s + 2] = 1.;
    cosine[s + 2] = 0.;
    r[s + 2] = b;
  else
    t = abs(a) + abs(b);
    rho = t * sqrt(abs(a / t) ^ 2 + abs(b / t) ^ 2);
    alpha = a / abs(a);

    sine[s + 2] = alpha * conj(b) / rho;
    cosine[s + 2] = abs(a) / rho;
    r[s + 2] = alpha * rho;
  end
end

@inline function computeMu(t, v)
  # Compute residual minimizing mu
  kappa = 0.7;
  tv = vecdot(t, v);
  tt = vecdot(t, t);

  omega = tv / tt;
  rho = tv / (sqrt(tt) * norm(v));
  if abs(rho) < kappa
    omega *= kappa / abs(rho);
  end
  return abs(omega) > eps() ? mu = 1. / omega : 1.;
end

@inline function *{T}(V::Vector{Vector{T}}, gamma::Vector{T})
  v = zeros(V[1]);
  for (idx, gam) in enumerate(gamma)
    @blas! v += gam * V[idx];
  end
  return v;
end

@inline function innerProducts{T}(R::Array{T}, V::Vector{Vector{T}})
  M = zeros(T, size(R, 2), size(V, 1));
  for idx = 1 : size(M, 1)
    for (jdx, v) = enumerate(V)
      M[idx, jdx] = vecdot(R[:, idx], v);
    end
  end
  return M;
end

end
