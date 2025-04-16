import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# --- Import your solvers ---
from solver import pdstsp_dp_drop_pickup  # CP solver (assumed implemented already)
from solver2 import run_ga             # GA helper function with updated returns

# --- Helper function to generate random problem data for GA ---
def generate_random_data(num_customers):
    depot_id = 0
    total_nodes = num_customers + 1  # Include depot
    # Generate random coordinates for depot and customers.
    coordinates = {i: (random.uniform(0, 100), random.uniform(0, 100)) for i in range(total_nodes)}
    # Compute travel cost using Euclidean distance plus an offset.
    travel_costs = {
        (i, j): int(np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))) + 10
        for i in range(total_nodes) for j in range(total_nodes) if i != j
    }
    # Generate random service times (depot has 0).
    service_times = {i: random.randint(4, 8) for i in range(total_nodes)}
    service_times[depot_id] = 0
    # Define time windows: wider window for depot and custom windows for customers.
    time_windows = {
        i: (5 * (i - 1), 5 * (i - 1) + 20) if i != 0 else (0, 100)
        for i in range(total_nodes)
    }
    return travel_costs, service_times, time_windows, coordinates

# --- GA Solver Runner ---
def run_ga_solver(num_customers, num_trucks, num_drones):
    travel_costs, service_times, time_windows, coordinates = generate_random_data(num_customers)
    best_solution, best_fitness, result_text, result_img, solver = run_ga(
        num_customers, num_trucks, num_drones,
        travel_costs, service_times, time_windows, coordinates,
        population_size=30, generations=100, mutation_rate=0.1, crossover_rate=0.7
    )
    st.subheader("Genetic Algorithm (GA) Results")
    
    # Display the GA solution using tabs.
    tab1, tab2 = st.tabs(["Schedule", "Route Map"])
    with tab1:
        st.markdown("### Optimized Delivery Schedule")
        st.code(result_text, language='text')
    with tab2:
        st.markdown("### Optimized Route Visualization")
        image = Image.open(result_img)
        st.image(image, caption="Optimized Vehicle Routes", use_column_width=True)
    
    st.markdown(f"**Best Fitness:** {best_fitness}")

# --- CP Solver Runner ---
def run_cp_solver(num_customers, kappa_t, kappa_d, beta):
    # Call the CP solver function (assumed to return text and an image buffer).
    result_text, result_img = pdstsp_dp_drop_pickup(c=num_customers, kappa_t=kappa_t, kappa_d=kappa_d, beta=beta, verbose=False)
    st.subheader("Constraint Programming (CP) Results")
    if result_img is None:
        st.error("No feasible solution found with the current parameters. Try adjusting the configuration.")
    else:
        tab1, tab2 = st.tabs(["Schedule", "Route Map"])
        with tab1:
            st.markdown("### Optimized Delivery Schedule")
            st.code(result_text, language='text')
        with tab2:
            st.markdown("### Optimized Route Visualization")
            image = Image.open(result_img)
            st.image(image, caption="Optimized Vehicle Routes", use_column_width=True)

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide", page_title="Truck & Drone Route Optimizer", page_icon="🚚")
    st.title("🚚 Truck & Drone Route Optimizer")
    st.markdown("""
        This application addresses a simplified **Parcel Delivery with Trucks and Drones** problem.
        Given a set of customers and a fleet of trucks and drones, the objective is to generate an optimized
        delivery schedule and route plan. Users can select between a **Genetic Algorithm (GA)** approach and a 
        **Constraint Programming (CP)** approach.
    """)
    
    # --- Sidebar: Method Selection and Parameter Inputs ---
    with st.sidebar:
        st.header("Configuration")
        method = st.radio("Select Solution Method:", options=["Genetic Algorithm", "Constraint Programming"])
        st.subheader("Customer Settings")
        num_customers = st.slider("Number of Customers", min_value=5, max_value=30, value=10)
        
        if method == "Genetic Algorithm":
            st.subheader("Vehicle Fleet (GA)")
            num_trucks = st.slider("Number of Trucks 🚛", min_value=1, max_value=5, value=2)
            num_drones = st.slider("Number of Drones 🛸", min_value=1, max_value=5, value=2)
        else:
            st.subheader("Vehicle Fleet (CP)")
            kappa_t = st.slider("Number of Trucks 🚛", min_value=1, max_value=5, value=2)
            kappa_d = st.slider("Number of Drones 🛸", min_value=1, max_value=5, value=2)
            st.subheader("Drone Parameters (CP)")
            beta = st.number_input("Drone Endurance (Beta)", min_value=50, max_value=500, value=100, step=10)
        
        st.markdown("Adjust the parameters and click 'Solve' to generate an optimized delivery schedule.")
    
    # --- Main Content Area: Display current configuration ---
    st.header("Current Configuration")
    if method == "Genetic Algorithm":
        cols = st.columns(3)
        cols[0].metric("Customers", num_customers)
        cols[1].metric("Trucks", num_trucks)
        cols[2].metric("Drones", num_drones)
    else:
        cols = st.columns(4)
        cols[0].metric("Customers", num_customers)
        cols[1].metric("Trucks", kappa_t)
        cols[2].metric("Drones", kappa_d)
        cols[3].metric("Endurance", beta)
    
    # --- Solve Button ---
    if st.button("Solve Optimization Problem", use_container_width=True):
        with st.spinner("Optimizing routes and schedules..."):
            if method == "Genetic Algorithm":
                run_ga_solver(num_customers, num_trucks, num_drones)
            else:
                run_cp_solver(num_customers, kappa_t, kappa_d, beta)
    
    # --- Additional Information ---
    st.markdown("---")
    st.header("Additional Information")
    with st.expander("How The Optimizer Works"):
        st.markdown("""
        This application implements advanced solution methodologies for the **Pickup and Delivery Problem with Trucks and Drones (PDSTSP)**.
        
        **Genetic Algorithm (GA):**
        - Uses bio-inspired methods to evolve a population of delivery routes.
        - Balances exploration (mutation, crossover) and exploitation (selection of the best solutions).
        
        **Constraint Programming (CP):**
        - Formulates the delivery problem with decision variables, constraints, and an objective function.
        - Utilizes constraint propagation and branch-and-bound search to find an optimal solution.
        
        The overall objective is to minimize total travel cost and ensure on-time deliveries for all customers.
        """)
    with st.expander("Performance Considerations"):
        st.markdown("""
        **Performance Considerations:**
        - **Problem Size:** As the number of customers increases, the solution space grows exponentially.
        - **Vehicle Fleet Size:** More trucks and drones increase the complexity of route assignments.
        - **Parameter Settings:** Drone endurance (beta) and other vehicle constraints directly impact computational time.
        
        The solvers are optimized for efficiency through intelligent heuristics and constraint propagation techniques.
        """)
    st.markdown("---")
    st.markdown("© 2025 Truck-Drone Route Optimizer • Created with Streamlit")

if __name__ == "__main__":
    main()
