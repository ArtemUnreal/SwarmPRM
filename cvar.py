import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# GJK Algorithm for Collision Detection
def support(A, B, direction):
    """Find the support point in Minkowski difference along the given direction."""
    p1 = A[np.argmax(np.dot(A, direction))]  # Farthest point in set A along direction
    p2 = B[np.argmax(np.dot(B, -direction))]  # Farthest point in set B along -direction
    return p1 - p2

def gjk(A, B, max_iterations=30):
    """Gilbert-Johnson-Keerthi (GJK) algorithm for collision detection between sets A and B."""
    direction = np.array([1.0, 0.0])  # Initial search direction
    simplex = [support(A, B, direction)]  # Initial simplex
    direction = -simplex[0]  # Next search direction
    
    for _ in range(max_iterations):
        new_point = support(A, B, direction)
        if np.dot(new_point, direction) <= 0:
            # No collision
            return False
        
        simplex.append(new_point)
        
        if handle_simplex(simplex, direction):
            # Collision detected
            return True
    
    return False  # Если не нашли за max_iterations

def handle_simplex(simplex, direction):
    """Handle the simplex to check for origin and update direction."""
    if len(simplex) == 2:
        # Line case
        A, B = simplex
        AB = B - A
        AO = -A
        if np.dot(AB, AO) > 0:
            direction[:] = np.array([AB[1], -AB[0]])  # Перпендикулярный вектор в 2D
        else:
            simplex.pop(1)
            direction[:] = AO
    elif len(simplex) == 3:
        # Triangle case
        A, B, C = simplex
        AB = B - A
        AC = C - A
        AO = -A
        
        AB_perp = np.array([AB[1], -AB[0]])
        AC_perp = np.array([AC[1], -AC[0]])
        
        if np.dot(AB_perp, AO) > 0:
            simplex.pop(2)
            direction[:] = AB_perp
        elif np.dot(AC_perp, AO) > 0:
            simplex.pop(1)
            direction[:] = AC_perp
        else:
            return True  # Collision detected
    
    return False

# EPA Algorithm for Penetration Depth
def epa(simplex, A, B):
    """Expanding Polytope Algorithm (EPA) to find the penetration depth."""
    max_iterations = 30
    tolerance = 1e-6
    
    polytope = simplex[:]
    edges = [[polytope[i], polytope[j]] for i in range(len(polytope)) for j in range(i + 1, len(polytope))]
    
    for _ in range(max_iterations):
        min_dist = float('inf')
        closest_edge = None
        normal = None
        
        for edge in edges:
            edge_vector = edge[1] - edge[0]
            normal = np.cross(edge_vector, np.array([0, 0, 1]))[:2]  # Perpendicular in 2D
            normal /= np.linalg.norm(normal)
            distance = np.dot(normal, edge[0])
            
            if distance < min_dist:
                min_dist = distance
                closest_edge = edge
        
        if min_dist < tolerance:
            return normal * min_dist
        
        new_point = support(A, B, normal)
        polytope.append(new_point)
        edges.append([closest_edge[0], new_point])
        edges.append([closest_edge[1], new_point])
        edges.remove(closest_edge)
    
    return normal * min_dist

# Calculate Signed Distance Function (SDF)
def calculate_sdf(robot_mean, obstacle_vertices):
    # Convert robot_mean to a list of points with the same format as obstacle_vertices
    robot_points = np.array([robot_mean])
    
    gjk_result = gjk(robot_points, obstacle_vertices)
    if not gjk_result:
        # No collision: return positive distance
        sdf = np.linalg.norm(robot_mean - obstacle_vertices.mean(axis=0))
    else:
        # Collision detected: return negative penetration depth using EPA
        sdf = -epa(robot_points, obstacle_vertices)  # Передаем два множества: точки робота и препятствия
    
    return sdf

# Compute CVaR for collision risk
def compute_cvar(sdf_mean, sdf_var, alpha):
    return sdf_mean + norm.ppf(1 - alpha) * np.sqrt(sdf_var)

# Generate random obstacle within limits
def generate_random_obstacle(xlim, ylim, size=0.2):
    x_center = np.random.uniform(xlim[0] + size, xlim[1] - size)
    y_center = np.random.uniform(ylim[0] + size, ylim[1] - size)
    half_size = size / 2
    return np.array([[x_center - half_size, y_center - half_size],
                     [x_center + half_size, y_center - half_size],
                     [x_center + half_size, y_center + half_size],
                     [x_center - half_size, y_center + half_size]])

# Check collision between two obstacles
def check_collision(obstacle1, obstacle2):
    return gjk(obstacle1, obstacle2)

# Generate obstacles without collision
def generate_obstacles(num_obstacles, xlim, ylim, max_attempts=100):
    obstacles = []
    for i in range(num_obstacles):
        attempts = 0
        while attempts < max_attempts:
            new_obstacle = generate_random_obstacle(xlim, ylim)
            if all(not check_collision(new_obstacle, existing_obstacle) for existing_obstacle in obstacles):
                obstacles.append(new_obstacle)
                print(f"Obstacle {i + 1}/{num_obstacles} added.")
                break
            attempts += 1
        if attempts == max_attempts:
            print(f"Could not generate obstacle {i + 1}/{num_obstacles} within {max_attempts} attempts.")
            break
    return obstacles

# Visualize robot and obstacles
def visualize_robot_obstacles(robot_mean, obstacles, sdf_values):
    plt.figure(figsize=(6, 6))
    plt.scatter(*robot_mean, color='blue', label="Robot Mean")
    
    for i, obstacle in enumerate(obstacles):
        plt.fill(obstacle[:, 0], obstacle[:, 1], color='red', alpha=0.3)
        plt.text(obstacle.mean(axis=0)[0], obstacle.mean(axis=0)[1], f"SDF: {sdf_values[i]:.2f}", fontsize=8, color="black")
    
    plt.title("Randomized Obstacles and Robot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
mean_robot = np.array([0.5, 0.5])
cov_robot = [[0.01, 0], [0, 0.01]]

# Generate obstacles
num_obstacles = 5
xlim = (0, 1)
ylim = (0, 1)
obstacles = generate_obstacles(num_obstacles, xlim, ylim)

# Calculate SDF for each obstacle
sdf_values = [calculate_sdf(mean_robot, obstacle) for obstacle in obstacles]

# Visualize results
visualize_robot_obstacles(mean_robot, obstacles, sdf_values)
