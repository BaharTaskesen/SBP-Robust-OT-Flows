import numpy as np


# ============================================================
# Basic utilities: target support and base distribution
# ============================================================
### Possible ways to generate data:
## # ys = make_noisy_ring()
# ys = make_square_targets()
# ys_ring = make_target_points(n_modes=8, radius=2.0)
# y_center = np.array([[0.0, 0.0]])
# ys = np.vstack([ys_ring, y_center])
# ys = make_paired_targets(radius=2.0, pair_sep=0.08, n_pairs=4)

######################################
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




def make_circle_plus_outlier(n_modes=8, radius=2.0, outlier_dist=6.0):
    """
    Creates:
      - n_modes points evenly spaced on a circle of radius
      - 1 far-out outlier at distance outlier_dist on the x-axis
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_modes, endpoint=False)
    ys_circle = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    # Outlier target far away
    y_outlier = np.array([[outlier_dist, 0.0]])   # shape (1, 2)

    # Combine
    ys = np.vstack([ys_circle, y_outlier])        # shape (n_modes + 1, 2)
    return ys
