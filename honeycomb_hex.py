import tidy3d as td
import numpy as np


def create_hexagonal_honeycomb_slab(
    lattice_const: float = 1.0,      # Lattice constant a (um)
    disk_radius: float = 0.225,       # Disk radius (um)
    slab_thickness: float = 0.22,    # Slab thickness / z span (um)
    hex_radius: int = 5,             # Hexagonal region radius R (in unit cells)
    twist_angle: float = 0.0,        # Rotation angle (degrees)
    material: td.Medium = td.Medium(permittivity=3.47**2),
    centerz: float = 0.0,
) -> list:
    """
    Creates a honeycomb photonic crystal slab using hexagonal region selection
    via axial coordinates (i, j, k = -i-j).  Equivalent to the Lumerical script:

        for i = -R:R, for j = -R:R
            k = -i - j
            if |i|<=R and |j|<=R and |k|<=R:
                Sublattice A:  x = a*(i+j/2),  y = a*(j*sqrt(3)/2 - 1/(2*sqrt(3)))
                Sublattice B:  x = a*(i+j/2),  y = a*(j*sqrt(3)/2 + 1/(2*sqrt(3)))

    The hexagonal mask keeps exactly 3*R*(R+1)+1 unit cells (a regular hexagon
    of radius R in axial coordinates), unlike the rectangular crop used in
    create_centered_honeycomb_slab.

    Parameters
    ----------
    lattice_const  : float  – Lattice constant a (um).
    disk_radius    : float  – Radius of each cylinder (um).
    slab_thickness : float  – Cylinder height / slab z-span (um).
    hex_radius     : int    – Hexagonal region radius R (number of unit-cell
                              repeat units from center).
    twist_angle    : float  – In-plane rotation angle applied to every disk
                              position around (0, 0) (degrees).
    material       : td.Medium – Tidy3D medium for the cylinders.
    centerz        : float  – Z-coordinate of the slab mid-plane (um).

    Returns
    -------
    list of td.Cylinder
        All cylinder geometries that constitute the slab.
    """
    a = lattice_const
    R = hex_radius

    # Sublattice offset: a / (2 * sqrt(3)) = a * sqrt(3) / 6
    dy_offset = a / (2.0 * np.sqrt(3))

    # Rotation matrix (applied around origin)
    theta = np.radians(twist_angle)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s],
                    [s,  c]])

    structures = []

    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            k = -i - j
            # Hexagonal region condition (axial / cube coordinates)
            if abs(i) <= R and abs(j) <= R and abs(k) <= R:
                x_base = a * (i + j / 2.0)
                y_row  = a * (j * np.sqrt(3) / 2.0)

                # Two sublattice positions within the unit cell
                sublattice_positions = [
                    np.array([x_base, y_row - dy_offset]),  # Sublattice A
                    np.array([x_base, y_row + dy_offset]),  # Sublattice B
                ]

                for pos in sublattice_positions:
                    # Apply in-plane rotation
                    pos_rot = rot @ pos

                    cyl = td.Cylinder(
                        radius=disk_radius,
                        length=slab_thickness,
                        axis=2,
                        center=(float(pos_rot[0]), float(pos_rot[1]), centerz),
                    )
                    structures.append(cyl)

    return structures
