import numpy as np
import random
import matplotlib.pyplot as plt
import io

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
        plt.close()  # Close the figure to avoid interference.

        return best_solution, best_fitness

    def plot_routes(self, solution, title='Solution Routes'):
        # Create a new figure for plotting
        plt.figure(figsize=(10, 8))
        # Plot truck routes
        for i, truck_route in enumerate(solution['truck_routes']):
            if not truck_route:
                continue
            route_coords = [self.customer_locations[0]] + [self.customer_locations[cust] for cust in truck_route] + [self.customer_locations[0]]
            x, y = zip(*route_coords)
            plt.plot(x, y, marker='o', label=f'Truck {i+1}', linewidth=2)
        # Plot drone routes
        for i, drone_route in enumerate(solution['drone_routes']):
            if not drone_route:
                continue
            drone_label_added = False
            for customer in drone_route:
                trip = [self.customer_locations[0], self.customer_locations[customer], self.customer_locations[0]]
                x, y = zip(*trip)
                label = f'Drone {i+1}' if not drone_label_added else None
                plt.plot(x, y, linestyle='--', marker='x', linewidth=1.5, label=label)
                drone_label_added = True
        # Mark depot and customer locations
        for cust, coord in self.customer_locations.items():
            if cust == 0:
                plt.plot(*coord, 'ks', markersize=10, label='Depot')
            else:
                plt.text(coord[0], coord[1], str(cust), fontsize=10, ha='right')
        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # Do not call plt.show() so that the active figure can be captured.

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
                route_str = "0"
                for cust in drone_route:
                    route_str += f" → {cust}"
                route_str += " → 0"
                result_lines.append(f"Drone {i+1}: {route_str}")
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
    best_solution, best_fitness = solver.solve()
    result_text = solver.get_solution_text(best_solution)
    
    # Let solver.plot_routes create the figure.
    solver.plot_routes(best_solution, title="GA Optimized Routes")
    # Capture the current figure (created by solver.plot_routes) to a bytes buffer.
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    
    return best_solution, best_fitness, result_text, buf, solver
