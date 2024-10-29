import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial import distance_matrix
import networkx as nx

# --- Collision Detection (GJK and EPA) --- #
def support(A, B, direction):
    """Find the support point in Minkowski difference along the given direction."""
    p1 = A[np.argmax(np.dot(A, direction))]
    p2 = B[np.argmax(np.dot(B, -direction))]
    return p1 - p2

def gjk(A, B, max_iterations=30):
    """GJK algorithm for collision detection between sets A and B."""
    direction = np.array([1.0, 0.0])
    simplex = [support(A, B, direction)]
    direction = -simplex[0]

    for _ in range(max_iterations):
        new_point = support(A, B, direction)
        if np.dot(new_point, direction) <= 0:
            return False  # No collision
        simplex.append(new_point)
        if handle_simplex(simplex, direction):
            return True  # Collision detected
    return False

def handle_simplex(simplex, direction):
    """Check the simplex for origin and update direction."""
    if len(simplex) == 2:
        A, B = simplex
        AB = B - A
        AO = -A
        if np.dot(AB, AO) > 0:
            direction[:] = np.array([AB[1], -AB[0]])
        else:
            simplex.pop(1)
            direction[:] = AO
    elif len(simplex) == 3:
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
            return True
    return False

# --- CVaR Collision Detection --- #
def calculate_sdf(robot_mean, obstacle_vertices):
    robot_points = np.array([robot_mean])
    gjk_result = gjk(robot_points, obstacle_vertices)
    if not gjk_result:
        sdf = np.linalg.norm(robot_mean - obstacle_vertices.mean(axis=0))
    else:
        sdf = -epa(robot_points, obstacle_vertices)
    return sdf

def compute_cvar(sdf_mean, sdf_var, alpha):
    return sdf_mean + norm.ppf(1 - alpha) * np.sqrt(sdf_var)

# --- PRM Generation and Pathfinding --- #
def generate_random_node(xlim, ylim):
    """Generate a random node within the defined limits."""
    x = np.random.uniform(xlim[0], xlim[1])
    y = np.random.uniform(ylim[0], ylim[1])
    return np.array([x, y])

def check_collision(node1, node2, obstacles, alpha=0.05):
    """Check collision-free path between two nodes using CVaR-based GJK."""
    mean = (node1 + node2) / 2
    for obstacle in obstacles:
        sdf = calculate_sdf(mean, obstacle)
        if compute_cvar(sdf, 0.01, alpha) <= 0:  # Risk threshold check
            return False  # Collision detected
    return True

def build_graph(nodes, obstacles, xlim, ylim):
    """Build PRM graph with collision-checked edges."""
    G = nx.Graph()
    num_nodes = len(nodes)

    # Add all nodes to the graph
    for i in range(num_nodes):
        G.add_node(i, pos=nodes[i])

    # Create distance matrix and add edges if no collision is detected
    dist_matrix = distance_matrix(nodes, nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if dist_matrix[i, j] < 0.3:  # Set a distance threshold for connections
                if check_collision(nodes[i], nodes[j], obstacles):
                    G.add_edge(i, j, weight=dist_matrix[i, j])
    return G

def find_path(G, start, goal):
    """Find shortest path using Dijkstra's algorithm."""
    return nx.shortest_path(G, source=start, target=goal, weight='weight')

# --- Visualization --- #
def visualize_path(nodes, obstacles, path):
    plt.figure(figsize=(6, 6))
    plt.title("Path from Start to Goal Avoiding Obstacles")

    # Obstacles
    for obstacle in obstacles:
        plt.fill(obstacle[:, 0], obstacle[:, 1], color='red', alpha=0.3)

    # Nodes
    plt.scatter(nodes[:, 0], nodes[:, 1], color='gray', alpha=0.6)

    # Path
    path_nodes = nodes[path]
    plt.plot(path_nodes[:, 0], path_nodes[:, 1], color='blue', linewidth=2, label="Optimal Path")

    # Start and goal points
    plt.scatter(*nodes[path[0]], color='green', s=100, label="Start")
    plt.scatter(*nodes[path[-1]], color='purple', s=100, label="Goal")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main Execution --- #
xlim, ylim = (0, 1), (0, 1)
start = np.array([0.1, 0.1])
goal = np.array([0.9, 0.9])
num_nodes = 20

# Generate obstacles
obstacles = [np.array([[0.3, 0.3], [0.35, 0.3], [0.35, 0.35], [0.3, 0.35]]),
             np.array([[0.6, 0.6], [0.65, 0.6], [0.65, 0.65], [0.6, 0.65]]),
             np.array([[0.4, 0.1], [0.45, 0.1], [0.45, 0.15], [0.4, 0.15]])]

# Generate PRM nodes including start and goal
nodes = np.array([start] + [generate_random_node(xlim, ylim) for _ in range(num_nodes)] + [goal])

# Build graph and find path
G = build_graph(nodes, obstacles, xlim, ylim)
start_index, goal_index = 0, len(nodes) - 1

try:
    path = find_path(G, start_index, goal_index)
    visualize_path(nodes, obstacles, path)
except nx.NetworkXNoPath:
    print("No path found between start and goal.")
