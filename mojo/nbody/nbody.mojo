from math import sqrt
from random import random_float64, seed as random_seed
from time import monotonic as time_now
from collections import List
from sys import argv

# Constants
alias G = 6.67430e-11  # Gravitational constant

# ------------------------------------------------------
# Body
# ------------------------------------------------------
@fieldwise_init
struct Body(ImplicitlyCopyable, Movable):
    """A celestial body with position, velocity, mass and force."""

    var position: SIMD[DType.float64, 4]  # Using SIMD for position (x,y,z,0)
    var velocity: SIMD[DType.float64, 4]  # Using SIMD for velocity (vx,vy,vz,0)
    var mass: Float64
    var force: SIMD[DType.float64, 4]  # Using SIMD for force (fx,fy,fz,0)

    fn __init__(
        out self,
        position: SIMD[DType.float64, 4],
        velocity: SIMD[DType.float64, 4],
        mass: Float64,
    ):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.force = SIMD[DType.float64, 4](0, 0, 0, 0)


# ------------------------------------------------------
# NBodySimulation
# ------------------------------------------------------
@fieldwise_init
struct NBodySimulation:
    """N-body simulation using Newtonian gravitational physics with SIMD optimizations.
    """

    var bodies: List[Body]
    var num_bodies: Int
    var dt: Float64

    fn __init__(out self, num_bodies: Int, dt: Float64):
        """Initialize the N-body simulation with random positions, velocities, and masses.
        """
        self.num_bodies = num_bodies
        self.dt = dt
        self.bodies = List[Body]()

        # Initialize bodies randomly
        random_seed(42)
        var i = 0
        while i < num_bodies:
            # Generate random position
            var position = SIMD[DType.float64, 4](
                random_float64() * 2e10 - 1e10,  # x
                random_float64() * 2e10 - 1e10,  # y
                random_float64() * 2e10 - 1e10,  # z
                0,  # padding
            )

            # Generate random velocity
            var velocity = SIMD[DType.float64, 4](
                random_float64() * 2e3 - 1e3,  # vx
                random_float64() * 2e3 - 1e3,  # vy
                random_float64() * 2e3 - 1e3,  # vz
                0,  # padding
            )

            # Generate random mass
            var mass = random_float64() * 1e25 + 1e20

            # Create and add the body
            self.bodies.append(Body(position, velocity, mass))
            i += 1

    fn compute_forces(mut self):
        """Compute gravitational forces between all pairs of bodies."""
        # Reset forces
        for i in range(self.num_bodies):
            var body = self.bodies[i]
            body.force = SIMD[DType.float64, 4](0, 0, 0, 0)
            self.bodies[i] = body

        # Compute pairwise forces (N² algorithm)
        for i in range(self.num_bodies):
            for j in range(
                i + 1, self.num_bodies
            ):  # Optimize: only compute each pair once
                var body_i = self.bodies[i]
                var body_j = self.bodies[j]

                # Vector from body i to body j
                var r_vec = body_j.position - body_i.position

                # Distance squared (using SIMD dot product)
                var r_squared = (r_vec * r_vec).reduce_add()

                # Avoid division by zero or extremely small values
                if r_squared < 1e-20:
                    continue

                # Distance
                var r = sqrt(r_squared)

                # Force magnitude
                var force_mag = G * body_i.mass * body_j.mass / r_squared

                # Force vector (direction from i to j, normalized by distance)
                var force_vec = r_vec * (force_mag / r)

                # Apply Newton's third law: equal and opposite forces
                body_i.force += force_vec
                body_j.force -= force_vec  # Opposite direction

                # Update bodies
                self.bodies[i] = body_i
                self.bodies[j] = body_j

    fn update_positions_and_velocities(mut self):
        """Update positions and velocities for all bodies."""
        for i in range(self.num_bodies):
            var body = self.bodies[i]

            # Calculate acceleration (F = ma, so a = F/m)
            var acceleration = body.force / body.mass

            # Update velocity (v += a * dt)
            body.velocity += acceleration * self.dt

            # Update position (p += v * dt)
            body.position += body.velocity * self.dt

            # Update body
            self.bodies[i] = body

    fn run_simulation(mut self, iterations: Int) -> Float64:
        """Run the N-body simulation for the specified number of iterations."""
        # Record starting time
        var start_time = time_now()

        # Run simulation for specified number of iterations
        for _ in range(iterations):
            self.compute_forces()
            self.update_positions_and_velocities()

        # Calculate execution time in seconds
        var end_time = time_now()
        return (end_time - start_time) * 1e-9  # Time in seconds

    fn calculate_energy(self) -> Float64:
        """Calculate the total energy of the system (kinetic + potential)."""
        var energy: Float64 = 0.0

        # Calculate kinetic energy and potential energy
        for i in range(self.num_bodies):
            var body_i = self.bodies[i]

            # Kinetic energy: 0.5 * m * v^2
            energy += (
                0.5
                * body_i.mass
                * (body_i.velocity * body_i.velocity).reduce_add()
            )

            # Potential energy: -G * m1 * m2 / r
            for j in range(i + 1, self.num_bodies):
                var body_j = self.bodies[j]
                var diff = body_i.position - body_j.position
                var distance = sqrt((diff * diff).reduce_add())
                energy -= G * body_i.mass * body_j.mass / distance

        return energy


# ------------------------------------------------------
# main
# ------------------------------------------------------
fn main():
    print("Starting N-body simulation...")

    args = argv()
    # Configuration parameters
    var num_bodies = 2000
    if len(args) > 1:
        try:
            num_bodies = atol(args[1])
        except _:
            print("Cannot parse num_bodies value:", args[1])
            return
    var iterations = 1000
    if len(args) > 2:
        try:
            iterations = atol(args[2])
        except _:
            print("Cannot parse iterations value:", args[2])
            return
    var dt = 0.01
    if len(args) > 3:
        try:
            dt = atof(args[3])
        except _:
            print("Cannot parse iterations dt:", args[3])
            return
    var benchmark_csv = ""
    if len(args) > 4:
        benchmark_csv = args[4]

    # Create and run the simulation
    var simulation = NBodySimulation(num_bodies, dt)

    # Calculate initial energy
    var initial_energy = simulation.calculate_energy()
    print("Initial system energy:", initial_energy)

    # Run simulation and measure time
    var execution_time = simulation.run_simulation(iterations)

    # Calculate final energy
    var final_energy = simulation.calculate_energy()
    print("Final system energy:", final_energy)
    print("Energy difference:", final_energy - initial_energy)

    # Print performance results
    print("Mojo Implementation (SIMD-optimized)")
    print("Number of bodies:", num_bodies)
    print("Number of iterations:", iterations)
    print("Time step:", dt)
    print("Execution time:", execution_time, "seconds")
    print("Done")

    # Write execution time to CSV file
    if len(benchmark_csv) > 0:
        try:
            with open(benchmark_csv, "a") as f:
                f.write("mojo,{}\n".format(execution_time))
        except _:
            print("Cannot open CSV file")
            return
