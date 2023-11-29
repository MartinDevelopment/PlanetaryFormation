import numpy as np
import datetime
from barnes_hut import Node

N_particles = 1000
N_dimensions = 3

# Constants
solar_mass = 1.989e30  # kg
earth_mass = 6e24  # kg
AU = 1.496e11  # m (Astronomical Unit)
G = 6.67408e-11  # m^3 kg^-1 s^-2

# Scaling factors
mass_scale = solar_mass
distance_scale = AU  # 1 AU
time_scale = np.sqrt(distance_scale**3 / (G * mass_scale))  # seconds

# Scaled parameters
size = 150e9 / AU  # in AU
velocity_mean = 0  # AU/s
velocity_deviation = 1  # AU/s

# Set G to 1 in scaled units
G_scaled = 1

# Initialize the particles
particles = np.zeros((N_particles, 8))
# Give all particles a random position uniformly distributed, in AU
particles[:, :3] = np.random.uniform(low=0, high=0.5, size=(N_particles, 3)) * size
# Give all particles mass equal to the mass of the earth, scaled
particles[:, 6] = earth_mass / mass_scale
particles[:, 7] = 6371e3 / AU

# Define the normal vector of the accretion disc
normal_vector = np.array([0, 0, 1])


# Function to get velocity for circular orbit in scaled units
def get_velocity_circular_orbit(p_list) -> np.ndarray:
    positions = p_list[:, :N_dimensions]
    norms = np.linalg.norm(positions, axis=1)
    v_list = np.sqrt(G_scaled / norms**3)[:, np.newaxis] * np.cross(normal_vector, positions)
    return v_list

# Giving all particles a velocity for orbit
particles[:, N_dimensions:2 * N_dimensions] = get_velocity_circular_orbit(particles)


# Function to get force in scaled units
def get_force(p1, p2) -> np.ndarray:
    r = p1[:N_dimensions] - p2[:N_dimensions]
    return G_scaled * p2[6] * p1[6] * r / np.linalg.norm(r) ** 3


# Function to get central force in scaled units
def get_central_force(p1) -> np.ndarray:
    r = p1[:N_dimensions]
    return -G_scaled * p1[6] * r / np.linalg.norm(r) ** 3

def calculate_potential_energy(particles):
    PE = 0
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            distance = np.linalg.norm(particles[i, :N_dimensions] - particles[j, :N_dimensions])
            PE += -G_scaled * particles[i, 6] * particles[j, 6] / distance
    return PE


# Simulation parameters
dt = 0.01  # in scaled time units
t = 0
pos_history = []

# Simulate the particles using Verlet integration
for _ in range(200):
    # Calculate initial acceleration
    root = Node.create_tree(particles)
    acceleration = np.zeros((N_particles, N_dimensions))
    for idx, particle in enumerate(particles):
        acceleration[idx] += root.get_force_with_particle(particle)
        acceleration[idx] += get_central_force(particle)
    acceleration /= particles[:, 6][:, np.newaxis]

    # Update positions
    particles[:, :N_dimensions] += particles[:, N_dimensions:2 * N_dimensions] * dt + 0.5 * acceleration * dt**2

    # Calculate new acceleration
    root = Node.create_tree(particles)
    new_acceleration = np.zeros((N_particles, N_dimensions))
    for idx, particle in enumerate(particles):
        new_acceleration[idx] += root.get_force_with_particle(particle)
        new_acceleration[idx] += get_central_force(particle)
    new_acceleration /= particles[:, 6][:, np.newaxis]

    # Update velocities
    average_acceleration = 0.5 * (acceleration + new_acceleration)
    particles[:, N_dimensions:2 * N_dimensions] += average_acceleration * dt

    # Update the time
    t += dt

    # Print the time and the position of the first particle every 1000th iteration
    if _ % 1 == 0:
        time = datetime.timedelta(seconds=t * time_scale)  # Convert back to real time for display
        #KE = np.sum(0.5 * particles[:, 6] * np.linalg.norm(particles[:, N_dimensions:2 * N_dimensions], axis=1) ** 2)
        #PE = calculate_potential_energy(particles)
        #E = KE + PE
        #print(f"Time: {time}, Position: {particles[0, :N_dimensions]}, E: {E}")
        print(f"N: {_}, Time: {time}, Position: {particles[0, :N_dimensions]}")

        # Save particle position
        pos_history.append(particles[:, :N_dimensions].copy())

# Save particle position to file
np.save("pos_history", pos_history)