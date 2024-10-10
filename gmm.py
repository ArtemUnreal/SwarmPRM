import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 1. Создание гауссовых распределений
def create_gaussian(v):
    """
    Создает двумерное гауссово распределение с параметрами из вектора v.
    
    v: вектор параметров [x, y, sigma1, sigma2, rho]
    Возвращает: среднее и ковариационную матрицу
    """
    x, y, sigma1, sigma2, rho = v
    mean = np.array([x, y])
    cov = np.array([[sigma1**2, rho * sigma1 * sigma2],
                    [rho * sigma1 * sigma2, sigma2**2]])
    return mean, cov

# Проверка положительной определённости матрицы
def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

# 2. Генерация узлов в свободном пространстве
def sample_free(n, sigma_lb, sigma_ub, rho_lb, rho_ub, x_range, y_range):
    """
    Генерация n узлов (гауссовых распределений) в свободном пространстве.
    
    n: количество узлов
    sigma_lb, sigma_ub: нижняя и верхняя границы для sigma1 и sigma2
    rho_lb, rho_ub: нижняя и верхняя границы для rho
    x_range, y_range: диапазоны для координат x и y
    
    Возвращает: список узлов (векторов параметров)
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

# 3. Нахождение ближайших узлов
def near(V, v, rn):
    """
    Возвращает все узлы, находящиеся в пределах радиуса rn от узла v.
    
    V: множество узлов
    v: текущий узел
    rn: радиус связи
    
    Возвращает: список узлов в пределах радиуса
    """
    def wasserstein_distance(v1, v2):
        """Вычисление расстояния Васерштейна между двумя гауссовыми распределениями."""
        mean1, cov1 = create_gaussian(v1)
        mean2, cov2 = create_gaussian(v2)
        
        if not is_positive_definite(cov1) or not is_positive_definite(cov2):
            return np.inf  
        
        # Рассчитываем Евклидово расстояние между средними
        mean_diff = np.linalg.norm(mean1 - mean2)
        
        try:
            # Проверяем подкоренные значения перед вычислением квадратного корня
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

# 4. Проверка свободного пути (без столкновений)
def collision_free(g1, g2, obstacles):
    """
    Проверяет, свободен ли путь между двумя гауссовыми распределениями.
    
    g1, g2: гауссовы распределения
    obstacles: список препятствий
    
    Возвращает: True, если путь свободен, False - если нет
    """
    for obstacle in obstacles:
        if not in_free(g1, obstacle) or not in_free(g2, obstacle):
            return False
    return True

def in_free(g, obstacle):
    """
    Проверяет, находится ли гауссово распределение g в свободной области, без препятствий.
    
    g: гауссово распределение
    obstacle: препятствие (x, y, радиус)
    
    Возвращает: True, если распределение свободно, иначе False
    """
    x, y = g[0]  # Координаты распределения
    ox, oy, r = obstacle  # Препятствие: (x, y, радиус)
    return np.linalg.norm([x - ox, y - oy]) > r

# 5. Основная функция построения дорожной карты
def roadmap_construction(n, rn, obstacles, D):
    """
    Строит граф на основе узлов и рёбер, соединяющих гауссовы распределения.
    
    n: количество узлов
    rn: радиус связи
    obstacles: список препятствий
    D: множество начальных и целевых PDF
    
    Возвращает: граф (V, E)
    """
    # Создаем узлы в параметрическом пространстве
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

# 6. Визуализация графа с препятствиями и выделением начальных и целевых узлов
def visualize_graph(V, E, obstacles, D):
    """
    Визуализирует граф на плоскости с учётом препятствий и выделением начальных и целевых узлов.
    
    V: список узлов
    E: список рёбер
    obstacles: список препятствий (окружностей)
    D: список начальных и целевых узлов
    """
    G = nx.Graph()
    
    # Добавляем узлы и рёбра
    for v in V:
        G.add_node(tuple(v[:2]))  # Используем только координаты x, y
    for e in E:
        G.add_edge(tuple(e[0][:2]), tuple(e[1][:2]))  # Соединяем по x, y координатам
    
    # Рисуем препятствия
    fig, ax = plt.subplots()
    
    for obstacle in obstacles:
        ox, oy, r = obstacle
        circle = plt.Circle((ox, oy), r, color='red', fill=True, alpha=0.3)
        ax.add_artist(circle)
    
    # Отображаем начальные и целевые узлы
    start_goal_positions = [tuple(v[:2]) for v in D]
    
    # Рисуем граф
    pos = {tuple(v[:2]): tuple(v[:2]) for v in V}
    nx.draw(G, pos, with_labels=False, node_size=50, node_color='blue', edge_color='black')
    
    nx.draw_networkx_nodes(G, pos, nodelist=start_goal_positions, node_color='green', node_size=100)
    
    plt.title("Gaussian Roadmap Graph with Obstacles and Start/Goal Nodes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()

# Пример использования
if __name__ == "__main__":
    # Пример препятствий: круги с координатами центра и радиусом
    obstacles = [(4, 4, 1), (6, 6, 1.5), (2, 8, 0.8)]

    # Начальные и целевые узлы
    D = [[1, 1, 0.5, 0.5, 0.1], [9, 9, 0.5, 0.5, -0.1]]
    
    V, E = roadmap_construction(n=100, rn=2.5, obstacles=obstacles, D=D)
    
    visualize_graph(V, E, obstacles, D)
