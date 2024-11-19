import heapq
import math

class Graph:
    def __init__(self):
        self.edges = {}

    def add_edge(self, u, v, weight):
        if u not in self.edges:
            self.edges[u] = []
        self.edges[u].append((v, weight))

    def neighbors(self, node):
        return self.edges.get(node, [])

def euclidean_heuristic(node, goal, positions):
    """
    Calculate Euclidean distance heuristic between a node and the goal.
    :param node: Current node
    :param goal: Goal node
    :param positions: Dictionary of node positions {node: (x, y)}
    """
    x1, y1 = positions[node]
    x2, y2 = positions[goal]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def hybrid_shortest_path(graph, start, goal, heuristic, positions, epsilon=0.1):
    """
    Hybrid algorithm combining Dijkstra's and A* search.

    :param graph: Graph object
    :param start: Starting node
    :param goal: Goal node
    :param heuristic: Heuristic function h(n)
    :param positions: Dictionary of node positions for heuristic calculation
    :param epsilon: Threshold for heuristic effectiveness
    :return: Shortest path and its cost
    """
    open_set = []
    heapq.heappush(open_set, (0, start))  # Priority queue with (f(n), node)

    g_cost = {start: 0}  # Actual cost from start to node
    came_from = {}  # Path reconstruction

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_cost[goal]

        for neighbor, weight in graph.neighbors(current):
            tentative_g = g_cost[current] + weight

            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g

                # Calculate heuristic effectiveness
                h_current = heuristic(current, goal, positions)
                h_neighbor = heuristic(neighbor, goal, positions)
                delta_h = abs(h_neighbor - h_current)

                if delta_h < epsilon:  # If heuristic is unreliable, switch to Dijkstra
                    f = tentative_g
                else:
                    f = tentative_g + h_neighbor

                heapq.heappush(open_set, (f, neighbor))
                came_from[neighbor] = current

    return None, float('inf')  # No path found

# Example Usage
graph = Graph()
positions = {
    'A': (0, 0), 'B': (1, 2), 'C': (2, 4), 'D': (3, 1), 'E': (5, 0)
}
graph.add_edge('A', 'B', 2.5)
graph.add_edge('A', 'C', 4)
graph.add_edge('B', 'C', 1)
graph.add_edge('B', 'D', 2)
graph.add_edge('C', 'E', 3)
graph.add_edge('D', 'E', 2)

path, cost = hybrid_shortest_path(graph, 'A', 'E', euclidean_heuristic, positions)
print("Path:", path)
print("Cost:", cost)
