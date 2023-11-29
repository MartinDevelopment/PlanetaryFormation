import pygame
import numpy as np

# Pygame setup
pygame.init()
width, height = 1400, 750
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Particle Simulation")
clock = pygame.time.Clock()


# Function to convert simulation coordinates to screen coordinates
def to_screen_coords(pos, scale=2 / height):
    coords = int(width / 2 + pos[0] / scale), int(height / 2 - pos[1] / scale)
    return coords

# Simulation loop
particles = np.load("pos_history.npy")
running = True
for i in range(np.size(particles, 0)):

    # Event handler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not running:
        break

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the particles
    for particle in particles[i]:
        particle_pos = particle[:2]
        particle_pos = to_screen_coords(particle_pos)
        pygame.draw.circle(screen, (255, 0, 0), particle_pos, 5)

    # Draw the sun at the center
    pygame.draw.circle(screen, (255, 255, 0), to_screen_coords(np.array([0, 0])), 10)

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(10)
pygame.quit()