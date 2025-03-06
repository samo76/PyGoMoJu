# PyGoMo: N-Body Simulation Benchmark

A comparative benchmark of N-body gravitational simulation implemented in Python, Go, and Mojo/Max to evaluate their performance characteristics for computationally intensive floating-point operations.

## Overview

This project implements the same N-body simulation algorithm in three different languages and provides a benchmarking framework to compare their performance. The simulation models gravitational interactions between celestial bodies using Newtonian physics.

### Algorithm

The simulation uses a direct N² pairwise force calculation approach based on Newton's law of universal gravitation. For each pair of bodies, the gravitational force is calculated and applied according to:

F = G *(m₁* m₂) / r²

Where:

- G is the gravitational constant (6.67430 × 10⁻¹¹)
- m₁ and m₂ are the masses of the two bodies
- r is the distance between the bodies

### Physics Implementation Details

The simulation follows these key physics principles:

#### 1. Distance Vector Calculation

For each pair of bodies (i and j), we calculate the vector from body i to body j:

```
r_vec = position_j - position_i
```

This vector points from body i toward body j and is essential for determining the direction of the gravitational force.

#### 2. Force Magnitude

The magnitude of the gravitational force is calculated using Newton's law of universal gravitation:

```
force_magnitude = G * mass_i * mass_j / (distance * distance)
```

#### 3. Force Direction

The gravitational force acts along the line connecting the two bodies. The force vector is calculated as:

```
force_vector = force_magnitude * (r_vec / |r_vec|)
```

Where `r_vec / |r_vec|` is the unit vector pointing from body i to body j.

#### 4. Position and Velocity Updates

For each time step (dt):

1. Calculate the acceleration: a = F / m
2. Update velocity: v = v + a * dt
3. Update position: p = p + v * dt

This implements a simple Euler integration method for the equations of motion.

#### 5. Optimizations

All implementations include these key optimizations:

- **Newton's Third Law**: Each pair of bodies is computed only once, with equal and opposite forces applied to both bodies
- **Energy Conservation**: The total energy (kinetic + potential) is calculated to verify simulation accuracy
- **Efficient Distance Calculations**: Using squared distances where possible to avoid unnecessary square root operations

### Implementations

1. **Python**: Uses NumPy for efficient array operations
2. **Go**: Leverages goroutines for parallel execution
3. **Mojo/Max**: Utilizes SIMD vectorization and parallelization features

## Project Structure

```
PyGoMo/
├── python/           # Python implementation
│   └── nbody.py      # N-body simulation in Python with NumPy
├── go/               # Go implementation
│   └── nbody.go      # N-body simulation in Go with goroutines
├── mojo/             # Mojo implementation
│   └── nbody.mojo    # N-body simulation in Mojo with SIMD
└── benchmark/        # Benchmarking tools
    └── benchmark.py  # Script to run and compare implementations
```

## Requirements

### Python

- Python 3.7+
- NumPy
- Matplotlib
- Pandas

### Go

- Go 1.16+

### Mojo

- Mojo/Max SDK

## Running the Benchmark

To run the benchmark comparison across all implementations:

```bash
cd PyGoMo
python benchmark/benchmark.py
```

The benchmark will:

1. Run each implementation with varying numbers of bodies (100, 500, 1000, 2000)
2. Measure and compare execution times
3. Generate visualization of performance results
4. Output a summary table of relative performance

## Implementation Details

### Python Implementation

The Python implementation uses NumPy for efficient array operations and employs optimized loops for the force calculations. While NumPy provides significant advantages over pure Python, it still faces limitations in parallelism for this specific algorithm.

Key optimizations:

- Store forces as a class member to avoid reallocating arrays
- Use `np.sum()` for vector dot products instead of `np.linalg.norm()` where possible
- Optimize the force calculation loop to only compute each pair once

Key code snippet for force calculation:

```python
# Vector from body i to body j
r_vec = self.positions[j] - self.positions[i]
# Distance squared
r_squared = np.sum(r_vec * r_vec)
# Calculate gravitational force
force_mag = self.G * self.masses[i] * self.masses[j] / r_squared
# Force vector (direction from i to j)
force_vec = force_mag * r_vec / np.sqrt(r_squared)
# Apply Newton's third law: equal and opposite forces
self.forces[i] += force_vec
self.forces[j] -= force_vec  # Opposite direction
```

### Go Implementation

The Go implementation uses goroutines to parallelize the force calculations across available CPU cores. The code splits the bodies into batches and processes each batch in a separate goroutine, leveraging Go's lightweight concurrency model.

Key optimizations:

- Added `MagnitudeSquared()` method to avoid unnecessary square root calculations
- Improved parallel implementation to divide work more efficiently among goroutines
- Use mutex for thread-safe force updates

Key code snippet for force calculation:

```go
// Vector from body i to body j
rVec := s.Bodies[j].Position.Sub(s.Bodies[i].Position)
// Distance squared (more efficient than computing magnitude directly)
rSquared := rVec.MagnitudeSquared()
// Calculate gravitational force magnitude
forceMag := s.G * s.Bodies[i].Mass * s.Bodies[j].Mass / rSquared
// Force vector (direction from i to j)
forceVec := rVec.Normalize().MulScalar(forceMag)
// Apply Newton's third law: equal and opposite forces
s.Bodies[i].Force = s.Bodies[i].Force.Add(forceVec)
s.Bodies[j].Force = s.Bodies[j].Force.Sub(forceVec) // Opposite direction
```

### Mojo/Max Implementation

The Mojo implementation is designed to showcase the language's SIMD and parallelism capabilities. It uses:

1. SIMD-friendly data layout with separate arrays for x, y, z components
2. Explicit parallelization through the `parallelize` function
3. Vectorized operations where possible

Key optimizations:

- Use SIMD operations for vector calculations
- Leverage Mojo's built-in parallelization capabilities
- Use `@value` attribute for efficient memory layout

Key code snippet for force calculation:

```mojo
# Vector from body i to body j
r_vec = body_j.position - body_i.position
# Distance squared (using SIMD dot product)
r_squared = (r_vec * r_vec).reduce_add()
# Force magnitude
force_mag = G * body_i.mass * body_j.mass / r_squared
# Force vector (direction from i to j, normalized by distance)
force_vec = r_vec * (force_mag / sqrt(r_squared))
# Apply Newton's third law: equal and opposite forces
body_i.force += force_vec
body_j.force -= force_vec  # Opposite direction
```

## Expected Results

Typically, the performance ranking from fastest to slowest is:

1. Mojo/Max (fastest)
2. Go
3. Python (slowest)

The Mojo implementation generally demonstrates significant speedup (often 10-100x) over Python due to its compiled nature and SIMD optimizations. Go usually sits between Mojo and Python in performance.

## Energy Conservation

All implementations track energy conservation as a way to verify simulation accuracy. The total energy of the system (kinetic + potential) should remain relatively constant throughout the simulation. Each implementation calculates:

- Kinetic energy: 0.5 *m* v²
- Potential energy: -G *m₁* m₂ / r

The difference between initial and final energy is reported to verify simulation correctness.

## Customizing the Simulation

Each implementation accepts command-line parameters to adjust:

- Number of bodies
- Number of iterations
- Time step (dt)
- Random seed (for reproducibility)

Example with Python:

```bash
cd PyGoMo
python python/nbody.py --bodies 2000 --iterations 500
```

## Visualization

The Python implementation includes optional visualization of particle trajectories using Matplotlib. The benchmark script also generates comparative charts of performance metrics.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
