import numpy as np

# ============================================================
# Basic utilities: target support and base distribution
# ============================================================

def make_target_points(n_modes=8, radius=2.0):
    """
    Arrange N target points on a circle of given radius.
    This defines the discrete target distribution:
        p_data = sum_i ν_i δ_{y_i}.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_modes, endpoint=False)
    ys = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    # And maybe give the center a smaller or equal weight in pdata

    return ys  # shape (N, 2)

def make_paired_targets(radius=2.0, pair_sep=0.08, n_pairs=4):
    # n_pairs pairs -> 2*n_pairs targets
    base_angles = np.linspace(0.0, 2*np.pi, n_pairs, endpoint=False)
    ys_list = []
    for theta in base_angles:
        ys_list.append([
            radius * np.cos(theta - pair_sep/2),
            radius * np.sin(theta - pair_sep/2),
        ])
        ys_list.append([
            radius * np.cos(theta + pair_sep/2),
            radius * np.sin(theta + pair_sep/2),
        ])
    ys = np.array(ys_list)  # shape (2*n_pairs, 2)
    return ys
def make_square_targets(side=3.0, points_per_edge=4):
    half = side / 2.0
    xs = np.linspace(-half, half, points_per_edge)
    ys = np.linspace(-half, half, points_per_edge)

    # four edges, no duplicate corners
    pts = []
    for x in xs:
        pts.append([x, -half])
        pts.append([x,  half])
    for y in ys[1:-1]:
        pts.append([-half, y])
        pts.append([ half, y])
    return np.array(pts)

# ys = make_square_targets()
def make_noisy_ring(n_modes=64, radius=2.0, jitter=0.25, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    angles = np.linspace(0.0, 2*np.pi, n_modes, endpoint=False)
    radii = radius + rng.normal(scale=jitter, size=n_modes)
    ys = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    return ys

# ys = make_noisy_ring()


def sample_base(n_samples, dim=2, rng=None):
    """
    Sample from the base distribution p_base = N(0, I_dim).
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(size=(n_samples, dim))

# ============================================================
# Semi-discrete OT: approximate Kantorovich potential φ
# ============================================================

def compute_kantorovich_potential(
    ys,
    n_iters=2000,
    batch_size=2048,
    lr=0.1,
    seed=0,
    verbose=False,
):
    """
    Approximate the semi-discrete OT Kantorovich potential φ on support {y_i}.

    We maximize the dual:
        F(φ) = sum_i ν_i φ_i - E_{x~p_base}[ max_j (φ_j - 0.5 ||x - y_j||^2) ]
    where p_base = N(0, I) and ν_i = 1/N.

    Gradient:
        dF/dφ_i = ν_i - P(argmax_j(...) = i).

    We approximate the expectation with Monte Carlo and do gradient ascent.
    """
    rng = np.random.default_rng(seed)
    N, dim = ys.shape
    phi = np.zeros(N, dtype=np.float64)
    nu = np.ones(N, dtype=np.float64) / N

    for it in range(n_iters):
        x = sample_base(batch_size, dim, rng)       # (B, dim)
        diff = x[:, None, :] - ys[None, :, :]       # (B, N, dim)
        cost = 0.5 * np.sum(diff ** 2, axis=-1)     # (B, N): 0.5||x - y_i||^2
        scores = phi[None, :] - cost                # (B, N)
        argmax = np.argmax(scores, axis=1)          # (B,)

        counts = np.bincount(argmax, minlength=N)
        freqs = counts.astype(np.float64) / batch_size   # empirical mass at each mode
        grad = nu - freqs                                # ν_i - empirical freq

        phi += lr * grad  # gradient ascent step

        if verbose and (it + 1) % max(1, n_iters // 10) == 0:
            F = np.dot(phi, nu) - np.mean(np.max(scores, axis=1))
            print(f"[OT dual] iter {it+1:5d}/{n_iters}, F ≈ {F:.4f}")

    # φ is only defined up to an additive constant; therefore we will center it, 
    # that is adding any constant to φ does not change the OT objective.
    phi -= np.mean(phi)
    return phi

# ============================================================
# Vector fields: OT vs robustified (smoothed Hopf–Lax)
# ============================================================

def velocity_ot(x, t, phi, ys, eps_time=1e-3):
    """
    OT velocity field v_OT(x,t) = ∇_x Q_t(φ,x) for the Hopf–Lax semigroup:

        Q_t(φ,x) = max_i [ φ_i - ||x - y_i||^2 / (2(1 - t)) ].

    Gradient is piecewise:
        ∇_x Q_t = (y_j - x) / (1 - t)
    where j is the argmax index.
    """
    x = np.asarray(x)
    B, dim = x.shape
    tau = max(eps_time, 1.0 - t)  # this is to avoid division by 0 near t≈1

    diff = x[:, None, :] - ys[None, :, :]          # (B, N, dim)
    cost = np.sum(diff ** 2, axis=-1) / (2.0 * tau)
    scores = phi[None, :] - cost                   # (B, N)
    argmax = np.argmax(scores, axis=1)             # (B,)

    v = (ys[argmax] - x) / tau
    return v


def velocity_robust(x, t, phi, ys, eps_smooth=0.3, eps_time=1e-3):
    """
    Robust / SBP-inspired velocity field v_rob(x,t) = ∇_x Q^{ε}_τ(φ,x),
    where Q^{ε}_τ is the smoothed Hopf–Lax potential:

        Q^{ε}_τ(φ,x) = ε log Σ_i exp( φ_i/ε - ||x - y_i||^2 / (2ε τ) ) + const,

    with τ = 1 - t here (we use the same time-param as OT).

    Gradient:
        ∇_x Q^{ε}_τ = Σ_i w_i (y_i - x) / τ,

    where w_i is a softmax over i.
    """
    x = np.asarray(x)
    B, dim = x.shape
    tau = max(eps_time, 1.0 - t)

    diff = x[:, None, :] - ys[None, :, :]           # (B, N, dim)
    sq_dist = np.sum(diff ** 2, axis=-1)            # (B, N)

    # exponent_i = φ_i/ε - ||x - y_i||^2 / (2 ε τ)
    exp_arg = (phi[None, :] / eps_smooth) - sq_dist / (2.0 * eps_smooth * tau)
    exp_arg = exp_arg - np.max(exp_arg, axis=1, keepdims=True)  # this is for numerical stability
    weights = np.exp(exp_arg)
    weights /= np.sum(weights, axis=1, keepdims=True)           # (B, N)

    y_bar = weights @ ys                                        # (B, dim)
    v = (y_bar - x) / tau
    return v


# ============================================================
# ODE simulation helpers
# ============================================================

def simulate_ode_path(v_fn, x0, n_steps=1000, t0=0.0, t1=1.0):
    """
    Euler integrate dX/dt = v_fn(X,t) and then we return the full path:
        path[k] = X at time t_k = t0 + k * dt.
    """
    x = np.asarray(x0).copy()
    B, dim = x.shape
    dt = (t1 - t0) / n_steps

    path = np.zeros((n_steps + 1, B, dim), dtype=np.float64)
    path[0] = x

    for k in range(n_steps):
        t = t0 + (k + 0.5) * dt
        v = v_fn(x, t)
        x = x + dt * v
        path[k + 1] = x

    return path


def simulate_ode(v_fn, x0, n_steps=1000, t0=0.0, t1=1.0):
    """
    Euler integrate dX/dt = v_fn(X,t), return only the final state at t1.
    """
    x = np.asarray(x0).copy()
    dt = (t1 - t0) / n_steps
    for k in range(n_steps):
        t = t0 + (k + 0.5) * dt
        v = v_fn(x, t)
        x = x + dt * v
    return x

# ============================================================
# Robustness metric: sensitivity of endpoints to φ-noise
# ============================================================

def endpoint_sensitivity(xT_clean, xT_noisy):
    """
    Robustness metric is the mean squared error:

        err = E[ || X_T^noisy - X_T^clean ||^2 ]

    where X_T^clean is the endpoint under the clean potential φ,
    and X_T^noisy is the endpoint under perturbed φ.

    Lower is better (meaning that the method is less sensitive to potential noise).
    """
    xT_clean = np.asarray(xT_clean)
    xT_noisy = np.asarray(xT_noisy)
    diff = xT_noisy - xT_clean
    sq = np.sum(diff ** 2, axis=1)
    return float(np.mean(sq))


