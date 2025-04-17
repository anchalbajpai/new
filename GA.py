import numpy as np
import random
import matplotlib.pyplot as plt
import io
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

class PDTSPGeneticAlgorithm:
    def __init__(self, num_trucks, num_drones, num_customers, travel_costs, service_times, time_windows,
                 truck_capacity, drone_capacity, customer_locations,
                 population_size=50, generations=100, mutation_rate=0.01, crossover_rate=0.7):
        self.num_trucks = num_trucks
        self.num_drones = num_drones
        self.num_customers = num_customers
        self.travel_costs = travel_costs
        self.service_times = service_times
        self.time_windows = time_windows
        self.truck_capacity = truck_capacity
        self.drone_capacity = drone_capacity
        self.customer_locations = customer_locations
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.customers = set(range(1, self.num_customers + 1))

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = {
                'truck_routes': [[] for _ in range(self.num_trucks)],
                'drone_routes': [[] for _ in range(self.num_drones)]
            }
            customers = list(range(1, self.num_customers + 1))
            random.shuffle(customers)
            for i, customer in enumerate(customers):
                vehicle_type = 'truck' if i % 2 == 0 else 'drone'
                vehicle_index = (i // 2) % (self.num_trucks if vehicle_type == 'truck' else self.num_drones)
                if vehicle_type == 'truck':
                    solution['truck_routes'][vehicle_index].append(customer)
                else:
                    solution['drone_routes'][vehicle_index].append(customer)
            population.append(solution)
        return population

    def calculate_route_fitness(self, route, is_drone=False):
        if not route:
            return 0, 0, 0

        total_distance = 0
        total_time_window_violations = 0
        speed_factor = 2 if is_drone else 1

        if is_drone:
            for customer in route:
                travel_to = self.travel_costs.get((0, customer), float('inf')) / speed_factor
                travel_back = self.travel_costs.get((customer, 0), float('inf')) / speed_factor
                route_time = travel_to
                earliest, latest = self.time_windows[customer]
                if route_time < earliest:
                    route_time = earliest
                if route_time > latest:
                    total_time_window_violations += (route_time - latest)
                route_time += self.service_times.get(customer, 0)
                route_time += travel_back
                total_distance += (travel_to + travel_back)
            return total_distance, total_time_window_violations, 0
        else:
            route_time = 0
            current_location = 0
            for customer in route:
                travel_cost = self.travel_costs.get((current_location, customer), float('inf'))
                route_time += travel_cost / speed_factor
                total_distance += travel_cost
                earliest, latest = self.time_windows[customer]
                if route_time < earliest:
                    route_time = earliest
                if route_time > latest:
                    total_time_window_violations += (route_time - latest)
                route_time += self.service_times.get(customer, 0)
                current_location = customer
            return_cost = self.travel_costs.get((current_location, 0), float('inf'))
            route_time += return_cost / speed_factor
            total_distance += return_cost
            return total_distance, total_time_window_violations, 0

    def calculate_fitness(self, solution):
        total_distance = 0
        total_time_window_violations = 0
        visited_customers = set()

        for truck_route in solution['truck_routes']:
            dist, time_viol, _ = self.calculate_route_fitness(truck_route, is_drone=False)
            total_distance += dist
            total_time_window_violations += time_viol
            visited_customers.update(truck_route)

        for drone_route in solution['drone_routes']:
            dist, time_viol, _ = self.calculate_route_fitness(drone_route, is_drone=True)
            total_distance += dist
            total_time_window_violations += time_viol
            visited_customers.update(drone_route)

        missing_customers = self.customers - visited_customers
        unassigned_penalty = len(missing_customers) * 10000
        fitness = total_distance + (1000 * total_time_window_violations) + unassigned_penalty
        return fitness

    def crossover(self, parent1, parent2):
        child = {
            'truck_routes': [[] for _ in range(self.num_trucks)],
            'drone_routes': [[] for _ in range(self.num_drones)]
        }
        split = random.randint(1, self.num_customers - 1)
        customers_part1 = set()

        for t in range(self.num_trucks):
            for c in parent1['truck_routes'][t][:split]:
                child['truck_routes'][t].append(c)
                customers_part1.add(c)

        for d in range(self.num_drones):
            for c in parent1['drone_routes'][d][:split]:
                child['drone_routes'][d].append(c)
                customers_part1.add(c)

        for customer in range(1, self.num_customers + 1):
            if customer not in customers_part1:
                vehicle_type = 'truck' if random.random() > 0.5 else 'drone'
                vehicle_index = random.randint(0, self.num_trucks - 1) if vehicle_type == 'truck' else random.randint(0, self.num_drones - 1)
                if vehicle_type == 'truck':
                    child['truck_routes'][vehicle_index].append(customer)
                else:
                    child['drone_routes'][vehicle_index].append(customer)
        return child

    def mutate(self, solution):
        if random.random() < self.mutation_rate:
            all_routes = solution['truck_routes'] + solution['drone_routes']
            flat_customers = [(i, j) for i, route in enumerate(all_routes) for j in range(len(route))]
            if len(flat_customers) < 2:
                return solution
            (i1, j1), (i2, j2) = random.sample(flat_customers, 2)
            all_routes[i1][j1], all_routes[i2][j2] = all_routes[i2][j2], all_routes[i1][j1]
        return solution

    def solve(self):
        population = self.initialize_population()
        best_fitness = float('inf')
        best_solution = None
        fitness_history = []

        for generation in range(self.generations):
            population_fitness = [self.calculate_fitness(sol) for sol in population]
            current_best_fitness = min(population_fitness)
            current_best_solution = population[population_fitness.index(current_best_fitness)]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = current_best_solution

            if generation % 10 == 0:
                print(f"Generation {generation} - Best Fitness: {best_fitness}")

            fitness_history.append(best_fitness)
            new_population = []
            for _ in range(self.population_size // 2):
                candidates = random.sample(population, 3)
                parent1 = min(candidates, key=self.calculate_fitness)
                candidates.remove(parent1)
                parent2 = min(candidates, key=self.calculate_fitness)
                if random.random() < self.crossover_rate:
                    child1 = self.crossover(parent1, parent2)
                    child2 = self.crossover(parent2, parent1)
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            population = new_population

        # Optional: Plot fitness history (for debugging)
        plt.figure(figsize=(10, 5))
        plt.plot(fitness_history)
        plt.title('Best Fitness over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        # Do not call plt.show() here; we simply want to debug.
        fitness_fig = plt.gcf()
        plt.close(fitness_fig)
        return best_solution, best_fitness, fitness_fig


    # def plot_routes(self, solution, title='Solution Routes'):
    #     plt.figure(figsize=(10, 8))

    #     # Plot truck routes
    #     for i, truck_route in enumerate(solution['truck_routes']):
    #         if not truck_route:
    #             continue
    #         route_coords = [self.customer_locations[0]] + [self.customer_locations[cust] for cust in truck_route] + [self.customer_locations[0]]
    #         x, y = zip(*route_coords)
    #         plt.plot(x, y, color='blue', marker='o', linewidth=2,
    #                 label='Truck Route' if i == 0 else None)
    #         # Add T# label near the middle
    #         mid_idx = len(x) // 2
    #         plt.text(x[mid_idx], y[mid_idx], f"T{i+1}", fontsize=9, color='blue', weight='bold')

    #     # Plot drone routes (one legend entry only)
    #     drone_colors = ['green', 'purple', 'red', 'brown', 'cyan', 'pink', 'gray']
    #     drone_legend_added = False

    #     for i, drone_route in enumerate(solution['drone_routes']):
    #         if not drone_route:
    #             continue
    #         drone_color = drone_colors[i % len(drone_colors)]
    #         for customer in drone_route:
    #             trip = [self.customer_locations[0], self.customer_locations[customer], self.customer_locations[0]]
    #             x, y = zip(*trip)
    #             line, = plt.plot(
    #                 x, y,
    #                 color=drone_color,
    #                 marker='x',
    #                 linewidth=2,
    #                 label='Drone Route' if not drone_legend_added else None
    #             )
    #             line.set_dashes([5, 5])  # 5pt line, 5pt gap
    #             drone_legend_added = True

    #             # Label D# on outbound leg
    #             mid_x = (x[0] + x[1]) / 2
    #             mid_y = (y[0] + y[1]) / 2
    #             plt.text(mid_x, mid_y, f"D{i+1}", fontsize=9, color=drone_color, weight='bold')


    #     # Mark depot and customers
    #     for cust, coord in self.customer_locations.items():
    #         if cust == 0:
    #             plt.plot(*coord, 'ks', markersize=10, label='Depot')
    #         else:
    #             plt.text(coord[0], coord[1], str(cust), fontsize=9, ha='right', va='bottom')

    #     plt.title(title)
    #     plt.xlabel('X Coordinate')
    #     plt.ylabel('Y Coordinate')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()

   

    def plot_routes(self, solution, title='GA Optimized Routes'):
    # Create a NetworkX graph
        G = nx.Graph()

        # Use self.customer_locations to get original positions.
        # Force the depot (node 0) at the center by placing all other nodes in a circle around it.
        pos = {}
        n_cust = len(self.customer_locations) - 1  # Number of customers excluding depot
        depot_coord = (0, 0)  # Depot is at the center
        
        # Place depot at the center
        pos[0] = depot_coord
        
        # Place customers in a circular pattern around the depot
        for idx, j in enumerate(self.customer_locations.keys()):
            if j == 0:
                continue  # Skip depot, it is already at (0, 0)
            angle = 2 * math.pi * idx / n_cust
            pos[j] = (4 * math.cos(angle), 4 * math.sin(angle))  # Scale radius as needed

        # Build a label dictionary for nodes; label depot as '0'
        labeldict = {0: "Depot"}
        # Optionally, keep any customer labels you want (here we use just the number)
        for node in self.customer_locations:
            if node != 0:
                labeldict[node] = str(node)

        # Plot Truck Routes:
        # For each truck, force the route to start and end at depot (0).
        for i, truck_route in enumerate(solution['truck_routes']):
            if not truck_route:
                continue
            route = [0] + truck_route + [0]
            for j in range(len(route) - 1):
                u, v = route[j], route[j + 1]
                G.add_edge(u, v, vehicle='truck', route_number=i+1)

        # Plot Drone Routes:
        # For each drone, add an edge from depot to each assigned customer
        # (We display only a single directed trip: depot → customer)
        for i, drone_route in enumerate(solution['drone_routes']):
            if not drone_route:
                continue
            for customer in drone_route:
                G.add_edge(0, customer, vehicle='drone', route_number=i+1, color='green')

        # Begin plotting.
        plt.figure(figsize=(10, 8))

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
        nx.draw_networkx_labels(G, pos, labels=labeldict, font_size=10, font_weight='bold')

        # Draw edges: truck routes in solid blue, drone routes in dashed green.
        truck_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('vehicle') == 'truck']
        nx.draw_networkx_edges(G, pos, edgelist=truck_edges, edge_color='blue', width=2)

        drone_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('vehicle') == 'drone']
        nx.draw_networkx_edges(G, pos, edgelist=drone_edges, edge_color='green', width=2, style='dashed')

        # Add route labels on the midpoints:
        # Truck route labels:
        for i, truck_route in enumerate(solution['truck_routes']):
            if truck_route:
                route = [0] + truck_route + [0]
                xs = [pos[n][0] for n in route]
                ys = [pos[n][1] for n in route]
                mid_x = sum(xs) / len(xs)
                mid_y = sum(ys) / len(ys)
                plt.text(mid_x, mid_y, f"T{i+1}", fontsize=12, color='blue',
                        weight='bold', ha='center', va='center')

        # Drone route labels: label each drone edge with D#
        for i, drone_route in enumerate(solution['drone_routes']):
            if drone_route:
                for customer in drone_route:
                    # For a direct trip, compute the midpoint between depot (0) and customer.
                    mid_x = (pos[0][0] + pos[customer][0]) / 2
                    mid_y = (pos[0][1] + pos[customer][1]) / 2
                    plt.text(mid_x, mid_y, f"D{i+1}", fontsize=12, color='green',
                            weight='bold', ha='center', va='center')

        # Custom legend
        truck_line = Line2D([0], [0], color='blue', lw=2, linestyle='-')
        drone_line = Line2D([0], [0], color='green', lw=2, linestyle='--')
        plt.legend([truck_line, drone_line], ['Truck Route', 'Drone Route'], loc='best', fontsize=12)

        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.axis('off')
        plt.tight_layout()
        plt.show()



    def get_solution_text(self, solution):
        """
        Build and return a readable text summary of the solution.
        """
        result_lines = []
        result_lines.append("Optimized Delivery Schedule:\n")

        result_lines.append("Truck Routes:")
        for i, truck_route in enumerate(solution['truck_routes']):
            if truck_route:
                route_str = "0"
                for cust in truck_route:
                    route_str += f" → {cust}"
                route_str += " → 0"
                result_lines.append(f"Truck {i+1}: {route_str}")
            else:
                result_lines.append(f"Truck {i+1}: No assignment")

        result_lines.append("\nDrone Routes:")
        for i, drone_route in enumerate(solution['drone_routes']):
            if drone_route:
                for cust in drone_route:
                    result_lines.append(f"Drone {i+1}: 0 → {cust} → 0")
            else:
                result_lines.append(f"Drone {i+1}: No assignment")

        return "\n".join(result_lines)


# --- Helper Function for Streamlit Integration ---
def run_ga(num_customers, num_trucks, num_drones, travel_costs, service_times, time_windows, customer_locations,
           population_size=30, generations=100, mutation_rate=0.1, crossover_rate=0.7):
    """
    Runs the GA solver with externally supplied data.
    Returns: (best_solution, best_fitness, result_text, image_buffer, solver_instance)
    """
    solver = PDTSPGeneticAlgorithm(
        num_trucks=num_trucks,
        num_drones=num_drones,
        num_customers=num_customers,
        travel_costs=travel_costs,
        service_times=service_times,
        time_windows=time_windows,
        truck_capacity=None,
        drone_capacity=None,
        customer_locations=customer_locations,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate
    )
    best_solution, best_fitness, fitness_fig = solver.solve()

    result_text = solver.get_solution_text(best_solution)
    
    # Let solver.plot_routes create the figure.
    solver.plot_routes(best_solution, title="GA Optimized Routes")
    # Capture the current figure (created by solver.plot_routes) to a bytes buffer.
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    fitness_buf = io.BytesIO()
    fitness_fig.savefig(fitness_buf, format="png")
    fitness_buf.seek(0)
    plt.close(fitness_fig)

    
    return best_solution, best_fitness, result_text, buf, fitness_buf, solver

