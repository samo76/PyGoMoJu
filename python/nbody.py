#!/usr/bin/env python3
"""
N-Body Simulation in Python with NumPy
Simulates gravitational interactions between celestial bodies using Newtonian physics.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys

# Try to import numba, but make it optional
try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    print("Numba not available. Using NumPy-only implementation.")


class NBodySimulation:
    def __init__(self, num_bodies=1000, num_iterations=1000, dt=0.01, seed=42):
        """
        Initialize the N-body simulation.

        Args:
            num_bodies: Number of celestial bodies to simulate
            num_iterations: Number of simulation steps
            dt: Time step (delta t)
            seed: Random seed for reproducibility
        """
        self.num_bodies = num_bodies
        self.num_iterations = num_iterations
        self.dt = dt
        self.G = 6.67430e-11  # Gravitational constant

        # Set the random seed for reproducibility
        np.random.seed(seed)

        # Initialize positions, velocities, and masses
        # Using universe size of 1e10 for positions
        self.positions = np.random.uniform(-1e10, 1e10, (num_bodies, 3))
        # Initial velocities near zero
        self.velocities = np.random.uniform(-1e3, 1e3, (num_bodies, 3))
        # Masses between 1e20 and 1e25 kg
        self.masses = np.random.uniform(1e20, 1e25, num_bodies)

        # Initialize forces array
        self.forces = np.zeros((num_bodies, 3))

        # Store position history for visualization
        self.position_history = []

        # Flag to choose computation method
        self.use_numba = HAVE_NUMBA

    def compute_forces(self):
        """
        Compute the gravitational forces between all pairs of bodies.
        Uses either vectorized NumPy operations or Numba JIT compilation if available.

        Returns:
            forces: Array of forces for each body
        """
        if self.use_numba and HAVE_NUMBA:
            # Use Numba JIT-compiled function for parallel execution
            self.forces = compute_forces_numba(self.positions, self.masses, self.G)
        else:
            # Use vectorized NumPy operations
            self.compute_forces_numpy()

        return self.forces

    def compute_forces_numpy(self):
        """
        Compute forces using vectorized NumPy operations.
        This avoids the O(N^2) nested loop problem.
        """
        # Reset forces
        self.forces.fill(0.0)

        # Compute pairwise displacement vectors (NxNx3)
        delta_r = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]

        # Compute squared distances (NxN matrix)
        r_squared = (
            np.sum(delta_r**2, axis=2) + 1e-20
        )  # Add small value to avoid division by zero

        # Create mask for i != j to avoid self-interactions
        mask = ~np.eye(self.num_bodies, dtype=bool)

        # Compute inverse distances (NxN matrix)
        inv_r = np.zeros_like(r_squared)
        inv_r[mask] = 1.0 / np.sqrt(r_squared[mask])

        # Compute force magnitudes (NxN matrix)
        force_mag = self.G * np.outer(self.masses, self.masses) * inv_r**3

        # Compute force vectors (NxNx3)
        force_vec = force_mag[:, :, np.newaxis] * delta_r

        # Sum forces along axis=1 to get total force per body
        self.forces = np.sum(force_vec, axis=1)

        return self.forces

    def update_positions_and_velocities(self):
        """
        Update positions and velocities using the computed forces.
        """
        # Calculate accelerations (F = ma, so a = F/m)
        accelerations = self.forces / self.masses[:, np.newaxis]

        # Update velocities (v += a * dt)
        self.velocities += accelerations * self.dt

        # Update positions (p += v * dt)
        self.positions += self.velocities * self.dt

    def run_simulation(self):
        """
        Run the N-body simulation for the specified number of iterations.

        Returns:
            execution_time: Time taken to run the simulation
        """
        # Record the starting time
        start_time = time.time()

        # Store initial positions
        self.position_history.append(self.positions.copy())

        # Run simulation for specified number of iterations
        for _ in range(self.num_iterations):
            self.compute_forces()
            self.update_positions_and_velocities()
            self.position_history.append(self.positions.copy())

        # Calculate execution time
        execution_time = time.time() - start_time
        return execution_time

    def calculate_energy(self):
        """
        Calculate the total energy of the system (kinetic + potential).

        Returns:
            energy: Total energy of the system
        """
        # Initialize energy
        energy = 0.0

        # Calculate kinetic and potential energy
        for i in range(self.num_bodies):
            # Kinetic energy: 0.5 * m * v^2
            velocity_squared = np.sum(self.velocities[i] * self.velocities[i])
            energy += 0.5 * self.masses[i] * velocity_squared

            # Potential energy: -G * m1 * m2 / r
            for j in range(i + 1, self.num_bodies):
                r_vec = self.positions[j] - self.positions[i]
                distance = np.linalg.norm(r_vec)
                if distance > 1e-10:  # Avoid division by zero
                    energy -= self.G * self.masses[i] * self.masses[j] / distance

        return energy

    def visualize(self, num_bodies_to_plot=10, output_file=None):
        """
        Visualize the trajectories of a subset of bodies.

        Args:
            num_bodies_to_plot: Number of bodies to include in the visualization
            output_file: Path to save the visualization. If None, will use a default name.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Select a subset of bodies to plot
        selected_bodies = np.random.choice(
            self.num_bodies, num_bodies_to_plot, replace=False
        )

        for body_idx in selected_bodies:
            # Extract trajectory for this body
            trajectory = np.array(
                [positions[body_idx] for positions in self.position_history]
            )

            # Plot the trajectory
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                label=f"Body {body_idx}",
            )

            # Mark the final position
            ax.scatter(
                trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], marker="o"
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("N-Body Simulation Trajectories")

        # Save the visualization
        if output_file is None:
            output_file = "nbody_python_visualization.png"

        plt.savefig(output_file)
        plt.close()


# Only define the Numba function if Numba is available
if HAVE_NUMBA:

    @numba.njit(parallel=True)
    def compute_forces_numba(positions, masses, G):
        """
        Compute forces using Numba's parallel JIT compilation.

        Args:
            positions: Array of body positions
            masses: Array of body masses
            G: Gravitational constant

        Returns:
            forces: Array of forces for each body
        """
        num_bodies = positions.shape[0]
        forces = np.zeros_like(positions)

        # Compute forces in parallel
        for i in numba.prange(num_bodies):
            for j in range(num_bodies):
                if i != j:
                    # Vector from body i to body j
                    r_vec = positions[j] - positions[i]

                    # Distance squared
                    r_squared = np.sum(r_vec * r_vec)

                    # Avoid division by zero
                    if r_squared < 1e-20:
                        continue

                    # Distance
                    r = np.sqrt(r_squared)

                    # Force magnitude
                    force_mag = G * masses[i] * masses[j] / r_squared

                    # Force vector
                    force_vec = force_mag * r_vec / r

                    # Add force to body i
                    forces[i] += force_vec

        return forces


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="N-Body Simulation in Python with NumPy"
    )

    parser.add_argument(
        "--bodies",
        type=int,
        default=2000,
        help="Number of bodies to simulate (default: 1000)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of simulation iterations (default: 1000)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Time step (default: 0.01)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of trajectories",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output filename for visualization"
    )
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable Numba JIT compilation (use NumPy only)",
    )

    return parser.parse_args()


def main():
    """Main function to run the simulation."""
    # Parse command-line arguments
    args = parse_args()

    # Create and run the simulation
    sim = NBodySimulation(
        num_bodies=args.bodies,
        num_iterations=args.iterations,
        dt=args.dt,
        seed=args.seed,
    )

    # Set computation method
    if args.no_numba:
        sim.use_numba = False

    # Calculate initial energy
    initial_energy = sim.calculate_energy()
    print(f"Initial system energy: {initial_energy:.6e}")

    # Run the simulation
    execution_time = sim.run_simulation()

    # Calculate final energy
    final_energy = sim.calculate_energy()
    print(f"Final system energy: {final_energy:.6e}")
    print(f"Energy difference: {final_energy - initial_energy:.6e}")

    # Print execution information
    print("Python NumPy Implementation")
    if sim.use_numba:
        print("Acceleration: Numba JIT with parallel execution")
    else:
        print("Acceleration: NumPy vectorization")
    print(f"Number of bodies: {args.bodies}")
    print(f"Number of iterations: {args.iterations}")
    print(f"Time step: {args.dt}")
    print(f"Random seed: {args.seed}")
    print(f"Execution time: {execution_time:.4f} seconds")

    # Visualize if requested
    if args.visualize:
        print("Generating visualization...")
        sim.visualize(num_bodies_to_plot=5, output_file=args.output)
        if args.output:
            print(f"Visualization saved to {args.output}")
        else:
            print("Visualization saved to nbody_python_visualization.png")


if __name__ == "__main__":
    main()
