from docplex.cp.model import *
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import io

def pdstsp_dp_drop_pickup(c=8, kappa_t=2, kappa_d=3, beta=55, seed=42, solve_time=25, verbose=True):
    random.seed(seed)
    depot = 0
    J = list(range(1, c + 1))  # Customer IDs: 1..c

    # ---------------------------
    # CUSTOMER ROLE ASSIGNMENT
    # ---------------------------
    # Randomly designate some customers as truck-only (e.g., 20-40% of customers).
    num_customers = len(J)
    num_truck_only = random.randint(max(1, int(0.2 * num_customers)), max(1, int(0.4 * num_customers)))
    truck_only_customers = set(random.sample(J, num_truck_only))
    
    # The remaining customers are drone-eligible.
    drone_eligible_customers = list(set(J) - truck_only_customers)
    
    # Randomly assign drone roles for the eligible customers:
    # For these customers, assign separate sets for drone "drop" and "pickup" roles.
    if drone_eligible_customers:
        drop_customers = set(random.sample(drone_eligible_customers, random.randint(1, len(drone_eligible_customers))))
        pickup_customers = set(random.sample(drone_eligible_customers, random.randint(1, len(drone_eligible_customers))))
    else:
        drop_customers = set()
        pickup_customers = set()
    
    # Dictionary for later tracking (if needed). It maps a customer j to the drone missions that serve j.
    customer_drone_missions = {j: [] for j in J}

    # ---------------------------
    # DATA: service times, time windows, travel times.
    # ---------------------------
    service_times = {j: random.randint(5, 15) for j in J}
    time_windows = {j: (random.randint(0, 30), random.randint(60, 130)) for j in J}
    full_nodes = [depot] + J
    # Truck travel times with small integer noise.
    tau_t = {(i, j): random.randint(8, 15) + random.randint(0, 1)
             for i in full_nodes for j in full_nodes if i != j}
    # Drone travel times.
    tau_d = {(i, j): random.randint(5, 10)
             for i in full_nodes for j in full_nodes if i != j}

    # ---------------------------
    # BUILD POSSIBLE DRONE MISSIONS
    # ---------------------------
    # We generate two types of missions:
    #    (i) depot -> drop -> depot (serving one customer)
    #    (ii) depot -> drop -> pickup -> depot (serving two customers)
    drone_missions = []
    # Generate drop missions only for customers eligible for a drop.
    for j in drop_customers:
        length = tau_d[depot, j] + service_times[j] + tau_d[j, depot]
        if length <= beta:
            # Represent the mission as a tuple (mission_type, (drop,))
            drone_missions.append(('drop', (j,)))
            customer_drone_missions[j].append(('drop', (j,)))
    # Generate drop-pickup missions: a drop from one customer and a pickup from another.
    for drop in drop_customers:
        for pickup in pickup_customers:
            if drop == pickup:
                continue
            length = (tau_d[depot, drop] + service_times[drop] +
                      tau_d[drop, pickup] + service_times[pickup] +
                      tau_d[pickup, depot])
            if length <= beta:
                # Represent the mission as a tuple (mission_type, (drop, pickup))
                drone_missions.append(('drop-pickup', (drop, pickup)))
                customer_drone_missions[drop].append(('drop-pickup', (drop, pickup)))
                customer_drone_missions[pickup].append(('drop-pickup', (drop, pickup)))
    
    # ---------------------------
    # CP MODEL SETUP
    # ---------------------------
    mdl = CpoModel()

    # For each customer j, create a mandatory service interval.
    Itv = {j: interval_var(size=service_times[j], name=f"Itv_{j}") for j in J}

    # ---- TRUCK ALTERNATIVES (MULTI-VEHICLE)
    # For each truck t and customer j, create an optional alternative interval.
    # This is now named ItvAlt to follow the paper's notation.
    ItvAlt = {t: {j: interval_var(optional=True, size=service_times[j], name=f"ItvAlt_{t}_{j}")
                  for j in J} for t in range(kappa_t)}
    
    # ---- DRONE ALTERNATIVES
    # For each possible drone mission, create one optional interval.
    DroneMsnVars = [interval_var(optional=True, name=f"DroneMsn_{i}") for i in range(len(drone_missions))]
    
    # Alternative assignment for each customer.
    # Every customer always has truck alternatives available.
    # For drone-eligible customers, we add drone mission intervals as additional options.
    for j in J:
        eligible = []
        # Truck alternatives are always present.
        for t in range(kappa_t):
            eligible.append(ItvAlt[t][j])
        # Add drone alternatives only if the customer is drone-eligible.
        if j not in truck_only_customers:
            for i, (tp, nodes) in enumerate(drone_missions):
                if j in nodes:
                    eligible.append(DroneMsnVars[i])
        mdl.add(alternative(Itv[j], eligible))
    
    # ---------------------------
    # TRUCK ROUTING (MULTI-VEHICLE TSP)
    # ---------------------------
    truck_seq = {}
    for t in range(kappa_t):
        truck_seq[t] = sequence_var([ItvAlt[t][j] for j in J], name=f"Truck_{t}_Seq")
    for t in range(kappa_t):
        tau_matrix = []
        for i in J:
            row = []
            for j2 in J:
                row.append(99999 if i == j2 else tau_t[i, j2])
            tau_matrix.append(row)
        mdl.add(no_overlap(truck_seq[t], transition_matrix(tau_matrix)))
    
    # ---------------------------
    # DRONE SEQUENCING (ALLOW MULTIPLE MISSIONS PER DRONE)
    # ---------------------------
    # Each drone mission interval represents a full round-trip.
    # Partition DroneMsnVars among drones using round-robin.
    drone_msn_groups = [[] for _ in range(kappa_d)]
    for i, v in enumerate(DroneMsnVars):
        drone_msn_groups[i % kappa_d].append(v)
    
    drone_seq = {}
    for d in range(kappa_d):
        group = drone_msn_groups[d][:]
        random.shuffle(group)  # Optional: randomize ordering
        drone_seq[d] = sequence_var(group, name=f"Drone_{d}_Seq")
        n = len(group)
        # Zero transition matrix indicates instantaneous turnaround.
        zero_matrix = [[0 for _ in range(n)] for _ in range(n)]
        mdl.add(no_overlap(drone_seq[d], transition_matrix(zero_matrix)))
    
    # ---------------------------
    # OBJECTIVE: MINIMIZE MAKESPAN
    # ---------------------------
    makespan_expr = mdl.max([end_of(Itv[j]) for j in J])
    mdl.add(minimize(makespan_expr))
    
    # ---------------------------
    # SOLVE THE MODEL
    # ---------------------------
    sol = mdl.solve(TimeLimit=solve_time, LogVerbosity='Quiet')
    if not sol:
        return "No solution found.", None
    
    # ---------------------------
    # OUTPUT AND VISUALIZATION
    # ---------------------------
    output_text = ""
    
    # --- Truck Tasks ---
    output_text += "Truck tasks:\n\n"
    for t in range(kappa_t):
        route = []
        t_jobs = []
        intervals = sol[truck_seq[t]]
        if intervals is not None:
            for interval in intervals:
                if interval.is_present():
                    name = interval.get_name()  # format: "ItvAlt_t_j"
                    st = interval.get_start()
                    try:
                        j = int(name.split('_')[-1])
                        route.append(j)
                        t_jobs.append((st, name))
                    except:
                        pass
            output_text += f"Truck {t} tasks: 0"
            for j in route:
                output_text += f" → {j}"
                
            output_text += f" → 0"
            output_text += "\n"
            for st, name in sorted(t_jobs):
                j = int(name.split('_')[-1])
                ed = st + service_times[j]
                output_text += f"  Truck {t} task for Customer {j} in [{st},{ed}]\n"
            output_text += "\n"
    
    # --- Drone Missions ---
    output_text += "Drone tasks:\n\n"
    for d in range(kappa_d):
        intervals = sol[drone_seq[d]]
        if intervals is not None:
            intervals_sorted = [iv for iv in intervals if iv.is_present()]
            intervals_sorted.sort(key=lambda iv: iv.get_start())
            for interval in intervals_sorted:
                mission_name = interval.get_name()  # e.g. "DroneMsn_3"
                i = int(mission_name.split('_')[-1])
                tp, nodes = drone_missions[i]
                st, ed = interval.get_start(), interval.get_end()
                if tp == "drop":
                    output_text += f"  Drone {d}: 0 → {nodes[0]}(Drop) → 0, time [{st},{ed}]"
                elif tp == "drop-pickup":
                    output_text += f"  Drone {d}: 0 → {nodes[0]}(Drop) → {nodes[1]}(Pickup) → 0, time [{st},{ed}]"
        else:
            output_text += f"  Drone {d}: No missions assigned."
        output_text += "\n"
    
    output_text += f"\nFinal Objective Value: {sol.get_objective_values()[0]}\n"
    
    # ---------------------------
    # VISUALIZATION (using networkx)
    # ---------------------------
    G = nx.DiGraph()
    pos = {0: (0, 0)}
    labeldict = {0: "Depot"}
    n_cust = len(J)
    for idx, j in enumerate(J):
        angle = 2 * math.pi * idx / n_cust
        pos[j] = (4 * math.cos(angle), 4 * math.sin(angle))
        lab = f"{j}{'T' if j in truck_only_customers else ''}"
        G.add_node(j)
        labeldict[j] = lab
    
    # Draw truck routes: add edge labels as "T{t}".
    for t in range(kappa_t):
        route = []
        intervals = sol[truck_seq[t]]
        if intervals is not None:
            for interval in intervals:
                if interval.is_present():
                    try:
                        j = int(interval.get_name().split('_')[-1])
                        route.append(j)
                    except:
                        pass
            if route:
                full_path = [0] + route
                for u, v in zip(full_path[:-1], full_path[1:]):
                    G.add_edge(u, v, color='tab:blue', width=2.2, style='solid', label=f"T{t}")
    
    # Draw drone missions: add edge labels as "D{d}".
    colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for d in range(kappa_d):
        col = colors[d % len(colors)]
        intervals = sol[drone_seq[d]]
        if intervals is not None:
            intervals_sorted = [iv for iv in intervals if iv.is_present()]
            intervals_sorted.sort(key=lambda iv: iv.get_start())
            for interval in intervals_sorted:
                mission_name = interval.get_name()
                i = int(mission_name.split('_')[-1])
                tp, nodes = drone_missions[i]
                # Edge from depot to first node in the mission.
                G.add_edge(0, nodes[0], color=col, style='solid', label=f"D{d}", width=2.5)
                # If drop-pickup, add edge with the same drone label.
                if tp == "drop-pickup":
                    G.add_edge(nodes[0], nodes[1], color=col, style='dashdot', label=f"D{d}", width=2.5)
                # Edge from last node of mission back to depot.
                G.add_edge(nodes[-1], 0, color=col, style='dotted', label=f"D{d}", width=2)
    
    # Draw the nodes and labels.
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=950, node_color='#f9f9c5')
    nx.draw_networkx_labels(G, pos, labels=labeldict, font_size=14, font_weight='bold')
    
    # Draw edges.
    for (u, v, data) in G.edges(data=True):
        style = data.get('style', 'solid')
        color = data.get('color', 'tab:blue')
        width = data.get('width', 2.5)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, width=width, style=style, arrows=True)
    
    # ---------------------------
    # Add edge labels on the arrows
    # ---------------------------
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=10)
    from matplotlib.lines import Line2D

# Draw graph
    edges = G.edges(data=True)
    edge_colors = [e[2]['color'] for e in edges]
    edge_styles = [e[2]['style'] for e in edges]
    edge_widths = [e[2].get('width', 1.0) for e in edges]

    # Prepare edge styles for drawing
    style_map = {'solid': '-', 'dotted': ':', 'dashdot': '-.'}
    edge_lines = []
    for (u, v, attr) in G.edges(data=True):
        style = style_map.get(attr.get('style', 'solid'), '-')
        edge_lines.append(((pos[u], pos[v]), attr.get('color', 'black'), style, attr.get('width', 1.0)))

    # Draw edges manually with correct style
    for line, color, style, width in edge_lines:
        (x1, y1), (x2, y2) = line
        plt.plot([x1, x2], [y1, y2], color=color, linestyle=style, linewidth=width)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)
    nx.draw_networkx_labels(G, pos, labels=labeldict, font_size=10)

    # Construct legend
    legend_elements = [
        Line2D([0], [0], color='tab:blue', lw=2, label='Truck Route'),
        Line2D([0], [0], color='tab:orange', lw=2, linestyle='-', label='Drone Drop-only'),
        Line2D([0], [0], color='tab:orange', lw=2, linestyle='-.', label='Drone Drop→Pickup'),
        Line2D([0], [0], color='tab:orange', lw=2, linestyle=':', label='Drone Return')
    ]

    plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
    plt.title("Vehicle and Drone Missions")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.title("Truck (T#) & Drone (D#) Routes", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    
    if verbose:
        print(output_text)
    

    return output_text, buf

# Example usage:
if __name__ == "__main__":
    text, buf = pdstsp_dp_drop_pickup()
    print(text)
    # For Jupyter Notebook: from PIL import Image; display(Image.open(buf))
