import datetime
from barnes_hut import Node
from multiprocessing import Pool
import numpy as np

def calculate_forces(particle, root):
    force = root.get_force_with_particle(particle)
    force += get_central_force(particle)
    return force
N_particles = 5000
N_dimensions = 3

# Constants
solar_mass = 1.989e30  # kg
earth_mass = 6e24  # kg
AU = 1.496e11  # m (Astronomical Unit)
G = 6.67408e-11  # m^3 kg^-1 s^-2

# Scaling factors
mass_scale = solar_mass
distance_scale = AU
time_scale = np.sqrt(distance_scale**3 / (G * mass_scale))


# Scaled parameters
size = 150e9 / AU
G_scaled = 1

# Initialize the particles
particles = np.zeros((N_particles, 8))
disk_thickness = 0
min_radius = 0.5
max_radius = 1
# Generate random radii and angles for a circular disk
radii = np.random.uniform(min_radius, max_radius, N_particles)
angles = np.random.uniform(0, 2 * np.pi, N_particles)

# Convert cylindrical to Cartesian coordinates
particles[:, 0] = radii * np.cos(angles)  # x-coordinate
particles[:, 1] = radii * np.sin(angles)  # y-coordinate
# For a flat disk, set z-coordinate to 0 or a small value
particles[:, 2] = np.random.uniform(-disk_thickness, disk_thickness, N_particles)

particles[:, 6] = earth_mass / mass_scale
particles[:, 7] = 6371e3 / AU

normal_vector = np.array([0, 0, 1])

def get_velocity_circular_orbit(p_list, central_mass=1):
    positions = p_list[:, :N_dimensions]
    norms = np.linalg.norm(positions, axis=1)
    v_magnitudes = np.sqrt(G_scaled * central_mass / norms)
    v_list = np.cross(normal_vector, positions)
    v_list = v_list / np.linalg.norm(v_list, axis=1)[:, np.newaxis] * v_magnitudes[:, np.newaxis]
    return v_list

particles[:, N_dimensions:2 * N_dimensions] = get_velocity_circular_orbit(particles)

def get_force(p1, p2):
    r = p1[:N_dimensions] - p2[:N_dimensions]
    r_norm = np.linalg.norm(r)
    return G_scaled * p2[6] * p1[6] * r / r_norm ** 3 if r_norm > 0 else 0

def get_central_force(p1):
    r = p1[:N_dimensions]
    r_norm = np.linalg.norm(r)
    return -G_scaled * p1[6] * r / r_norm ** 3 if r_norm > 0 else 0

def calculate_potential_energy(particles):
    PE = 0
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            distance = np.linalg.norm(particles[i, :N_dimensions] - particles[j, :N_dimensions])
            PE += -G_scaled * particles[i, 6] * particles[j, 6] / distance
    return PE

dt = 0.1
t = 0
pos_history = []
energy_history = []


def main():
    root = Node.create_tree(particles)
    with Pool() as pool:
        initial_forces = np.array(pool.starmap(calculate_forces, [(particle, root) for particle in particles]))
    half_step_velocities = particles[:, N_dimensions:2 * N_dimensions] + 0.5 * dt * initial_forces / particles[:, 6][:,
                                                                                                     np.newaxis]
    global t
    for _ in range(1000):
        # Update positions using half-step velocities
        particles[:, :N_dimensions] += half_step_velocities * dt

        root = Node.create_tree(particles)

        # Parallel processing
        with Pool() as pool:
            forces = np.array(pool.starmap(calculate_forces, [(particle, root) for particle in particles]))

        # Update velocities from half-step to next half-step
        half_step_velocities += forces * dt / particles[:, 6][:, np.newaxis]

        t += dt

        if _ % 1 == 0:
            time = datetime.timedelta(seconds=t * time_scale)
            #KE = np.sum(0.5 * particles[:, 6] * np.linalg.norm(particles[:, N_dimensions:2 * N_dimensions], axis=1) ** 2)
            #PE = calculate_potential_energy(particles)
            #E = KE + PE
            #energy_history.append(E)
            #print(f"Time: {time}, Position: {particles[0, :N_dimensions]}, E: {E}")
            print(f"N: {_}, Time: {time}, Position: {particles[0, :N_dimensions]}")
            pos_history.append(particles[:, :N_dimensions].copy())

    np.save("pos_history", pos_history)

if __name__ == '__main__':
    main()