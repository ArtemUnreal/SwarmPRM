import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 1. Create Gaussian distributions
def create_gaussian(v):
    """
    Creates a 2D Gaussian distribution with parameters from vector v.
    
    v: parameter vector [x, y, sigma1, sigma2, rho]
    Returns: mean and covariance matrix
    """
    x, y, sigma1, sigma2, rho = v
    mean = np.array([x, y])
    cov = np.array([[sigma1**2, rho * sigma1 * sigma2],
                    [rho * sigma1 * sigma2, sigma2**2]])
    return mean, cov

# Check if matrix is positive definite
def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

# 2. Generate nodes in free space
def sample_free(n, sigma_lb, sigma_ub, rho_lb, rho_ub, x_range, y_range):
    """
    Generate n nodes (Gaussian distributions) in free space.
    
    n: number of nodes
    sigma_lb, sigma_ub: lower and upper bounds for sigma1 and sigma2
    rho_lb, rho_ub: lower and upper bounds for rho
    x_range, y_range: ranges for x and y coordinates
    
    Returns: list of nodes (parameter vectors)
    """
    nodes = []
    for _ in range(n):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        sigma1 = np.random.uniform(sigma_lb, sigma_ub)
        sigma2 = np.random.uniform(sigma_lb, sigma_ub)
        rho = np.random.uniform(rho_lb, rho_ub)
        nodes.append([x, y, sigma1, sigma2, rho])
    return nodes

# 3. Find nearest nodes
def near(V, v, rn):
    """
    Returns all nodes within radius rn from node v.
    
    V: set of nodes
    v: current node
    rn: connection radius
    
    Returns: list of nodes within radius
    """
    def wasserstein_distance(v1, v2):
        """Calculates the Wasserstein distance between two Gaussian distributions."""
        mean1, cov1 = create_gaussian(v1)
        mean2, cov2 = create_gaussian(v2)
        
        if not is_positive_definite(cov1) or not is_positive_definite(cov2):
            return np.inf  
        
        # Calculate Euclidean distance between means
        mean_diff = np.linalg.norm(mean1 - mean2)
        
        try:
            sqrt_term = np.linalg.cholesky(cov1) @ np.linalg.cholesky(cov2)
            cov_diff = np.trace(cov1 + cov2 - 2 * sqrt_term)
        except np.linalg.LinAlgError:
            cov_diff = np.inf
        
        return mean_diff + cov_diff
    
    neighbours = []
    for v_prime in V:
        if wasserstein_distance(v, v_prime) <= rn:
            neighbours.append(v_prime)
    return neighbours

# 4. Check collision-free path
def collision_free(g1, g2, obstacles):
    """
    Checks if the path between two Gaussian distributions is collision-free.
    
    g1, g2: Gaussian distributions
    obstacles: list of obstacles
    
    Returns: True if path is free, False otherwise
    """
    for obstacle in obstacles:
        if not in_free(g1, obstacle) or not in_free(g2, obstacle):
            return False
    return True

def in_free(g, obstacle):
    """
    Checks if the Gaussian distribution g is in free space, without obstacles.
    
    g: Gaussian distribution
    obstacle: obstacle (x, y, radius)
    
    Returns: True if distribution is free, otherwise False
    """
    x, y = g[0]  # Distribution coordinates
    ox, oy, r = obstacle  # Obstacle: (x, y, radius)
    return np.linalg.norm([x - ox, y - oy]) > r

# 5. Main function for roadmap construction
def roadmap_construction(n, rn, obstacles, D):
    """
    Constructs a graph based on nodes and edges connecting Gaussian distributions.
    
    n: number of nodes
    rn: connection radius
    obstacles: list of obstacles
    D: set of initial and target PDFs
    
    Returns: graph (V, E)
    """
    V = list(D) + sample_free(n, 0.1, 2.0, -0.9, 0.9, (0, 10), (0, 10))
    E = []
    
    for v in V:
        gv = create_gaussian(v)
        neighbours = near([node for node in V if node != v], v, rn)
        
        for v_prime in neighbours:
            gv_prime = create_gaussian(v_prime)
            
            if collision_free(gv, gv_prime, obstacles):
                E.append((v, v_prime))
                E.append((v_prime, v))
    
    return V, E

# 6. Graph visualization with obstacles and highlighting start/goal nodes
def visualize_graph(V, E, obstacles, D):
    """
    Visualizes the graph on a plane with obstacles and highlights start and goal nodes.
    
    V: list of nodes
    E: list of edges
    obstacles: list of obstacles (circles)
    D: list of start and goal nodes
    """
    G = nx.Graph()
    
    # Add nodes and edges
    for v in V:
        G.add_node(tuple(v[:2]))  # Use only x, y coordinates
    for e in E:
        G.add_edge(tuple(e[0][:2]), tuple(e[1][:2]))  # Connect by x, y coordinates
    
    # Draw obstacles
    fig, ax = plt.subplots()
    
    for obstacle in obstacles:
        ox, oy, r = obstacle
        circle = plt.Circle((ox, oy), r, color='red', fill=True, alpha=0.3)
        ax.add_artist(circle)
    
    # Highlight start and goal nodes
    start_goal_positions = [tuple(v[:2]) for v in D]
    
    # Draw graph
    pos = {tuple(v[:2]): tuple(v[:2]) for v in V}
    nx.draw(G, pos, with_labels=False, node_size=50, node_color='blue', edge_color='black')
    
    nx.draw_networkx_nodes(G, pos, nodelist=start_goal_positions, node_color='green', node_size=100)
    
    plt.title("Gaussian Roadmap Graph with Obstacles and Start/Goal Nodes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.savefig('/app/image_gmm.png')
    plt.show()

# Usage example
if __name__ == "__main__":
    # Example obstacles: circles with center coordinates and radius
    obstacles = [(4, 4, 1), (6, 6, 1.5), (2, 8, 0.8)]

    # Start and goal nodes
    D = [[1, 1, 0.5, 0.5, 0.1], [9, 9, 0.5, 0.5, -0.1]]
    
    V, E = roadmap_construction(n=100, rn=2.5, obstacles=obstacles, D=D)
    
    visualize_graph(V, E, obstacles, D)
