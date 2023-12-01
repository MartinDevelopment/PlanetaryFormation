import numpy as np

root_size = 2

G = 1
theta = 0.5
dt = 0.1

class Node:
    def __init__(self, quadrant: np.ndarray):
        self.quadrant = quadrant
        self.children = []
        self.particles = []
        self.size = 2 ** (1/3) * (quadrant[0][1] - quadrant[0][0])
        self.centre_of_mass = None
        self.mass = None

    def create_child(self, quadrant: np.ndarray):
        child = Node(quadrant)
        self.children.append(child)
        return child

    def get_child(self, quadrant: np.ndarray):
        for child in self.children:
            if np.array_equal(child.quadrant, quadrant):
                return child
        return self.create_child(quadrant)

    def add_particle(self, particle):
        self.particles.append(particle)
        if len(self.particles) >= 2:
            quadrant = self.get_quadrant(particle)
            self.get_child(quadrant).add_particle(particle)

    def get_quadrant(self, particle):
        means = np.mean(self.quadrant, axis=1)
        position = np.array(
            [[means[dim], dim_size[1]] if particle[dim] >= means[dim] else [dim_size[0], means[dim]] for dim, dim_size
             in enumerate(self.quadrant)])
        return position

    def set_particles(self):
        self.particles = np.array(self.particles)

    def set_mass(self):
        self.mass = np.sum(self.particles[:, 6])

    def set_centre_of_mass(self):
        if self.mass > 0:
            masses = self.particles[:, 6]
            positions = self.particles[:, :3]
            self.centre_of_mass = np.sum(masses[:, np.newaxis] * positions, axis=0) / self.mass

    def traverse_tree(self, func):
        func(self)
        for child in self.children:
            child.traverse_tree(func)

    def get_angle_with_particle(self, particle):
        distance = np.linalg.norm(self.centre_of_mass - particle[:3])
        if distance == 0:
            # Handle the zero distance case
            return float('inf')
        return self.size / distance

    def get_force_with_particle(self, particle):
        angle = self.get_angle_with_particle(particle)
        if angle > theta or np.isin(particle, self.particles).all():
            return sum(child.get_force_with_particle(particle) for child in self.children)
        else:
            r = particle[:3] - self.centre_of_mass
            r_norm = np.linalg.norm(r)
            if r_norm > 0:
                for i in self.particles:
                    Node.collision_detection(particle, i)
                return G * particle[6] * self.mass * r / r_norm ** 3
            return 0

    # @classmethod
    # def collision_detection(cls, particle1, particle2):
    #     dx = particle1[:3] - particle2[:3]
    #     dv = particle1[3:6] - particle2[3:6]
    #     dv_norm_squared = np.dot(dv, dv)
    #     dx_norm_squared = np.dot(dx, dx)
    #     radii_sum_squared = (particle1[7] + particle2[7]) ** 2
    #     dxdv = np.dot(dx, dv)
    #     discriminant = dxdv ** 2 - dx_norm_squared * dv_norm_squared + radii_sum_squared * dv_norm_squared
    #     if discriminant >= 0:
    #         sqrt_discriminant = np.sqrt(discriminant)
    #         t12 = (-2 * dxdv - 2 * sqrt_discriminant) / dv_norm_squared
    #         if 0 < t12 < dt:
    #             print(True)
    #             print(t12)
    #             print("COLLISION")
    @classmethod
    def collision_detection(cls, particle1, particle2):
        dx = particle1[:3] - particle2[:3]
        dv = particle1[3:6] - particle2[3:6]

        dx_norm_squared = np.dot(dx, dx)
        dv_norm_squared = np.dot(dv, dv)

        radii_sum_squared = (particle1[7]**2 + particle2[7]**2)
        dxdv = np.dot(dx, dv)
        discriminant = dxdv ** 2 - dx_norm_squared * dv_norm_squared + radii_sum_squared * dv_norm_squared

        if discriminant < 0:
            return  # No collision possible, exit early

        sqrt_discriminant = np.sqrt(discriminant)
        t12 = (-2 * dxdv - 2 * sqrt_discriminant) / dv_norm_squared
        print(t12)
        if t12 <= 0 or t12 >= dt:
            return  # No collision within the time step, exit early

        print(t12)
        print("COLLISION")

    @classmethod
    def create_tree(cls, particles):
        zero_quadrant = np.array([[-root_size, root_size] for _ in range(3)])
        root = Node(zero_quadrant)
        for p in particles:
            root.add_particle(p)
        root.traverse_tree(Node.set_particles)
        root.traverse_tree(Node.set_mass)
        root.traverse_tree(Node.set_centre_of_mass)
        return root



