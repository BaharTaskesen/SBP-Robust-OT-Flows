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


def make_helix_targets(
    n_points=64,
    d=20,
    radius=2.0,
    height=6.0,
    cycles=3.0,
):
    """
    Build a 1 dimensional helix embedded in R^d.
    Only dims 0,1,2 contain the helix. Remaining dims are zero.

    Param:
        n_points: number of discrete support points on the helix
        d: ambient dimension
        radius: radius in the (x0, x1) plane
        height: total vertical extent along x2
        cycles: number of turns
    """
    t = np.linspace(0.0, cycles * 2.0 * np.pi, n_points)
    Y = np.zeros((n_points, d), dtype=np.float64)

    # 3d helix in coordinates 0,1,2
    Y[:, 0] = radius * np.cos(t)
    Y[:, 1] = radius * np.sin(t)
    Y[:, 2] = (t / (cycles * 2.0 * np.pi)) * height

    return Y



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
def make_curved_line(n_points=30, amp=1.0):
    xs = np.linspace(-3.5, 3.5, n_points)
    ys = xs + amp * np.sin(1.5 * xs)   # gentle sinusoidal curve
    return np.stack([xs, ys], axis=1)

def make_c_shape_targets(
    n_points=40,
    radius=2.5,
    center_x=0.8,
    center_y=0.0,
    theta_min=-3.0 * np.pi / 4.0,
    theta_max= 3.0 * np.pi / 4.0,
    jitter_std=0.0,
    rng=None,
):
    """
    Build a horseshoe or C shaped target.

    Geometry:
      - Start from a circular arc of given radius
      - Centered at (center_x, center_y)
      - Use angles from theta_min to theta_max (open side opposite to the missing arc)

    With default parameters:
      - Center is slightly to the right of origin
      - Arc wraps around origin and is open on the left side
    """
    if rng is None:
        rng = np.random.default_rng(0)

    angles = np.linspace(theta_min, theta_max, n_points)
    xs = center_x + radius * np.cos(angles)
    ys = center_y + radius * np.sin(angles)

    pts = np.stack([xs, ys], axis=1)

    if jitter_std > 0.0:
        pts = pts + rng.normal(scale=jitter_std, size=pts.shape)

    return pts


def make_diagonal_line_targets(n_points=20, x_min=-3.0, x_max=3.0):
    """
    Build a diagonal line from (x_min, x_min) to (x_max, x_max).
    Returns a set of evenly spaced points along the line.
    """
    xs = np.linspace(x_min, x_max, n_points)
    ys = xs   # diagonal: y = x
    pts = np.stack([xs, ys], axis=1)
    return pts

def make_hierarchical_targets(
    n_macro=3,
    n_micro_per_macro=4,
    macro_radius=2.5,
    micro_radius=0.25,
    macro_phase=0.0,
):
    """
    Build a hierarchical target support:
      - n_macro macro clusters, placed on a circle of radius macro_radius
      - each macro cluster has n_micro_per_macro micro modes
        arranged on a small circle of radius micro_radius around the macro center
    Returns:
      ys: array of shape (n_macro * n_micro_per_macro, 2)
    """
    angles_macro = np.linspace(0.0, 2.0 * np.pi, n_macro, endpoint=False) + macro_phase

    ys_list = []
    for theta in angles_macro:
        # macro center position
        cx = macro_radius * np.cos(theta)
        cy = macro_radius * np.sin(theta)

        # micro modes around the macro center
        angles_micro = np.linspace(0.0, 2.0 * np.pi, n_micro_per_macro, endpoint=False)
        for phi in angles_micro:
            mx = cx + micro_radius * np.cos(phi)
            my = cy + micro_radius * np.sin(phi)
            ys_list.append([mx, my])

    ys = np.array(ys_list, dtype=np.float64)
    return ys




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
