using StaticArrays
using Random, LinearAlgebra, Printf

# Constants
const G = 6.67430e-11  # Gravitational constant

const SV = SVector{3,Float64}

# ------------------------------------------------------
# Body
# ------------------------------------------------------
"""A celestial body with position, velocity, mass and force."""
mutable struct Body
    position::SV # Using SIMD for position (x,y,z)
    velocity::SV # Using SIMD for velocity (vx,vy,vz)
    mass::Float64
    force::SV # Using SIMD for force (fx,fy,fz)
end

Body(position::SV, velocity::SV, mass::Float64) =
    Body(position, velocity, mass, zero(SV))


# ------------------------------------------------------
# NBodySimulation
# ------------------------------------------------------
"""N-body simulation using Newtonian gravitational physics with SIMD optimizations."""
struct NBodySimulation
    bodies::Vector{Body}
    dt::Float64
end

"""Initialize the N-body simulation with random positions, velocities, and masses."""
function NBodySimulation(num_bodies::Int, dt::Float64)
    # Initialize bodies randomly
    Random.seed!(42)

    bodies = Vector{Body}(undef, num_bodies)

    for i in eachindex(bodies)
        # Generate random position
        position = SV(
            randn() * 2e10 - 1e10,  # x
            randn() * 2e10 - 1e10,  # y
            randn() * 2e10 - 1e10,  # z
        )

        # Generate random velocity
        velocity = SV(
            randn() * 2e3 - 1e3,  # vx
            randn() * 2e3 - 1e3,  # vy
            randn() * 2e3 - 1e3,  # vz
        )

        # Generate random mass
        mass = randn() * 1e25 + 1e20

        # Create and add the body
        bodies[i] = Body(position, velocity, mass)
    end

    return NBodySimulation(bodies, dt)
end

"""Compute gravitational forces between all pairs of bodies."""
function compute_forces(nbodysimulation::NBodySimulation)
    bodies = nbodysimulation.bodies
    num_bodies = length(bodies)

    # Reset forces
    for body in bodies
        body.force = zero(SV)
    end

    # Compute pairwise forces (N² algorithm)
    for i in eachindex(bodies)
        body_i = bodies[i]
        for j in (i+1):num_bodies # Optimize: only compute each pair once
            body_j = bodies[j]

            # Vector from body i to body j
            r_vec = body_j.position - body_i.position

            # Distance squared (dot product)
            r_squared = dot(r_vec, r_vec)

            # Avoid division by zero or extremely small values
            r_squared < 1e-20 && continue

            # Distance
            r = sqrt(r_squared)

            # Force magnitude
            force_mag = G * body_i.mass * body_j.mass / r_squared

            # Force vector (direction from i to j, normalized by distance)
            force_vec = r_vec * (force_mag / r)

            # Apply Newton's third law: equal and opposite forces
            body_i.force += force_vec
            body_j.force -= force_vec  # Opposite direction
        end
    end
    return nothing
end

"""Update positions and velocities for all bodies."""
function update_positions_and_velocities(nbodysimulation::NBodySimulation)
    for body in nbodysimulation.bodies
        # Calculate acceleration (F = ma, so a = F/m)
        acceleration = body.force / body.mass

        # Update velocity (v += a * dt)
        body.velocity += acceleration * nbodysimulation.dt

        # Update position (p += v * dt)
        body.position += body.velocity * nbodysimulation.dt
    end
    return nothing
end

"""Run the N-body simulation for the specified number of iterations."""
function run_simulation(nbodysimulation::NBodySimulation, iterations::Int)
    # Record starting time
    start_time = time_ns()

    # Run simulation for specified number of iterations
    for _ in 1:iterations
        compute_forces(nbodysimulation)
        update_positions_and_velocities(nbodysimulation)
    end

    # Calculate execution time in seconds
    end_time = time_ns()

    return (end_time - start_time) * 1e-9  # Time in seconds
end

"""Calculate the total energy of the system (kinetic + potential)."""
function calculate_energy(nbodysimulation::NBodySimulation)
    bodies = nbodysimulation.bodies
    num_bodies = length(bodies)
    energy = 0.0

    # Calculate kinetic energy and potential energy
    for i in eachindex(bodies)
        body_i = bodies[i]

        # Kinetic energy: 0.5 * m * v²
        energy += 0.5 * body_i.mass * dot(body_i.velocity, body_i.velocity)

        # Potential energy: -G * m1 * m2 / r
        for j in (i+1):num_bodies
            body_j = bodies[j]
            diff = body_i.position - body_j.position
            distance = norm(diff)
            energy -= G * body_i.mass * body_j.mass / distance
        end
    end

    return energy
end

# ------------------------------------------------------
# main
# ------------------------------------------------------
function (@main)(args)
    
    println("Starting N-body simulation...")

    # Configuration parameters
    num_bodies = length(args) > 0 ? parse(Int, args[1]) : 2000
    iterations = length(args) > 1 ? parse(Int, args[2]) : 1000
    dt = length(args) > 2 ? parse(Float64, args[3]) : 0.01

    # Create and run the simulation
    simulation = NBodySimulation(num_bodies, dt)

    # Calculate initial energy
    initial_energy = calculate_energy(simulation)
    @printf "Initial system energy: %.4e\n" initial_energy

    # Run simulation and measure time
    execution_time = run_simulation(simulation, iterations)

    # Calculate final energy
    final_energy = calculate_energy(simulation)
    @printf "Final system energy: %.4e\n" final_energy
    @printf "Energy difference: %.4e\n" final_energy - initial_energy

    # Print performance results
    println("Julia Implementation")
    println("Number of bodies: ", num_bodies)
    println("Number of iterations: ", iterations)
    println("Time step: ", dt)
    @printf "Execution time: %.4f seconds\n" execution_time
    println("Done")
end
