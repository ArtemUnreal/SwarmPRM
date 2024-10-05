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
    """Проверяет, является ли матрица положительно определённой."""
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
        
        # Проверка положительной определённости ковариационных матриц
        if not is_positive_definite(cov1) or not is_positive_definite(cov2):
            return np.inf  # Возвращаем бесконечное расстояние, если матрицы не положительно определённые
        
        # Рассчитываем Евклидово расстояние между средними
        mean_diff = np.linalg.norm(mean1 - mean2)
        
        # Рассчитываем след метрики Васерштейна между ковариационными матрицами
        cov_diff = np.trace(cov1 + cov2 - 2 * np.linalg.cholesky(cov1) @ np.linalg.cholesky(cov2))
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
    # Проверка на пересечение с препятствиями (placeholder)
    for obstacle in obstacles:
        if not in_free(g1, g2, obstacle):
            return False
    return True

def in_free(g, obstacle):
    """
    Проверяет, находится ли гауссово распределение g в свободной области, без препятствий.
    
    g: гауссово распределение
    obstacle: препятствие
    
    Возвращает: True, если распределение свободно, иначе False
    """
    # Здесь должна быть проверка расстояния от г до препятствия
    # Placeholder для примера
    return True

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

# 6. Визуализация графа
def visualize_graph(V, E):
    """
    Визуализирует граф на плоскости.
    
    V: список узлов
    E: список рёбер
    """
    G = nx.Graph()
    
    # Добавляем узлы и рёбра
    for v in V:
        G.add_node(tuple(v[:2]))  # Используем только координаты x, y
    for e in E:
        G.add_edge(tuple(e[0][:2]), tuple(e[1][:2]))  # Соединяем по x, y координатам
    
    # Рисуем граф
    pos = {tuple(v[:2]): tuple(v[:2]) for v in V}
    nx.draw(G, pos, with_labels=False, node_size=50, node_color='blue')
    
    plt.title("Gaussian Roadmap Graph")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# Пример использования
if __name__ == "__main__":
    # Пример препятствий (можно модифицировать)
    obstacles = []

    # Начальные и целевые узлы (например, из начальных и целевых PDF)
    D = [[1, 1, 0.5, 0.5, 0.1], [9, 9, 0.5, 0.5, -0.1]]
    
    # Строим дорожную карту
    V, E = roadmap_construction(n=100, rn=2.5, obstacles=obstacles, D=D)
    
    # Визуализируем результат
    visualize_graph(V, E)
