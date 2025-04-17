import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# --- Import your solvers ---
from CP import pdstsp_dp_drop_pickup  # CP solver (assumed implemented already)
from GA import run_ga  # GA helper function with updated returns


# --- Helper function to generate random problem data for GA ---
def generate_random_data(num_customers):
    depot_id = 0
    total_nodes = num_customers + 1  # Include depot
    coordinates = {
        i: (random.uniform(0, 100), random.uniform(0, 100)) for i in range(total_nodes)
    }
    travel_costs = {
        (i, j): int(np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))) + 10
        for i in range(total_nodes)
        for j in range(total_nodes)
        if i != j
    }
    service_times = {i: random.randint(4, 8) for i in range(total_nodes)}
    service_times[depot_id] = 0
    time_windows = {
        i: (5 * (i - 1), 5 * (i - 1) + 20) if i != 0 else (0, 100)
        for i in range(total_nodes)
    }
    return travel_costs, service_times, time_windows, coordinates


# --- GA Solver Runner ---
def run_ga_solver(num_customers, num_trucks, num_drones):
    travel_costs, service_times, time_windows, coordinates = generate_random_data(num_customers)
    solution, fitness, result_text, route_buf, fitness_buf, solver = run_ga(
        num_customers,
        num_trucks,
        num_drones,
        travel_costs,
        service_times,
        time_windows,
        coordinates,
        population_size=30,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.7,
    )

    st.subheader("\U0001F9EA Genetic Algorithm (GA) Results")
    tab1, tab2, tab3 = st.tabs(["Schedule", "Route Map", "Fitness Plot"])

    with tab1:
        st.markdown("### Optimized Delivery Schedule")
        st.code(result_text, language="text")
        st.code(f"Fitness Score: {fitness}", language="text")

    with tab2:
        st.markdown("### Optimized Route Visualization")
        image = Image.open(route_buf)
        st.image(image, caption="Optimized Vehicle Routes", use_column_width=True)

    with tab3:
        st.markdown("### Fitness Value Over Generations")
        image = Image.open(fitness_buf)
        st.image(image, caption="Fitness Value vs. Generation", use_column_width=True)


# --- CP Solver Runner ---
def run_cp_solver(num_customers, kappa_t, kappa_d, beta):
    result_text, result_img = pdstsp_dp_drop_pickup(
        c=num_customers, kappa_t=kappa_t, kappa_d=kappa_d, beta=beta, verbose=False
    )

    st.subheader("\U0001F9EE Constraint Programming (CP) Results")
    if result_img is None:
        st.error("No feasible solution found with the current parameters. Try adjusting the configuration.")
    else:
        tab1, tab2 = st.tabs(["Schedule", "Route Map"])
        with tab1:
            st.markdown("### Optimized Delivery Schedule")
            st.code(result_text, language="text")
        with tab2:
            st.markdown("### Optimized Route Visualization")
            image = Image.open(result_img)
            st.image(image, caption="Optimized Vehicle Routes", use_column_width=True)


# --- Streamlit App ---
def main():
    st.set_page_config(
        layout="wide", page_title="Truck & Drone Route Optimizer", page_icon="🚚"
    )

    st.title("Truck & Drone Route Optimizer")
    st.markdown("""
        In this project, we address the **Multiple Drone Delivery Problem**, a complex NP-Hard optimization challenge, using two distinct solution strategies:

        **Approaches:**
        - **Mixed Integer Linear Programming (MILP)** solved via a Genetic Algorithm  
        - **Constraint Programming (CP)** model
    """)

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("📁 Configuration")
        method = st.radio("Select Solver:", ["Genetic Algorithm", "Constraint Programming"])

        st.subheader(":busts_in_silhouette: Customer Settings")
        num_customers = st.slider("Number of Customers", 5, 30, 10)

        if method == "Genetic Algorithm":
            st.subheader(":truck: Fleet Settings (GA)")
            num_trucks = st.slider("Number of Trucks", 1, 5, 2)
            num_drones = st.slider("Number of Drones", 1, 5, 2)
        else:
            st.subheader(":truck: Fleet Settings (CP)")
            kappa_t = st.slider("Number of Trucks", 1, 5, 2)
            kappa_d = st.slider("Number of Drones", 1, 5, 2)

            st.subheader(":battery: Drone Endurance (CP)")
            beta = st.number_input("Beta (Max Flight Time)", 50, 500, 100, 10)

        st.markdown("---")
        st.markdown("Click **Solve** to compute optimal delivery routes and schedules.")

    # --- Current Configuration Display ---
    st.header(":gear: Current Configuration")
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
    if st.button("🔄 Solve Optimization Problem", use_container_width=True):
        with st.spinner("Running solver and optimizing routes..."):
            if method == "Genetic Algorithm":
                run_ga_solver(num_customers, num_trucks, num_drones)
            else:
                run_cp_solver(num_customers, kappa_t, kappa_d, beta)

    # --- Info Sections ---
    st.markdown("---")
    st.header(":information_source: Additional Information")

    with st.expander("How the Optimizer Works"):
        st.markdown("""
        This application solves the **Pickup and Delivery Problem with Trucks and Drones (PDSTSP)** using:

        - **Genetic Algorithm (GA):** Bio-inspired population-based search method that evolves delivery routes using crossover and mutation.
        - **Constraint Programming (CP):** Declarative approach using constraints and logical inference to prune the search space.

        The goal is to minimize total travel cost and delivery time while respecting service and time constraints.
        """)

    with st.expander("Performance Notes"):
        st.markdown("""
        - **Larger problem sizes** result in exponentially larger search spaces.
        - **Drone endurance** and fleet size heavily affect feasibility.
        - **Heuristics and optimization techniques** are used for faster convergence.
        """)

    st.markdown("---")
    st.caption("\u00a9 2025 Truck-Drone Route Optimizer | Built with Streamlit")


if __name__ == "__main__":
    main()
