package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"
)

// Vector3 represents a 3D vector.  Use plain floats, not structs.
// This is the most crucial optimization for memory layout and cache efficiency.
type Vector3 [3]float64 // Directly embed the floats

// Add two vectors.  Inline to avoid function call overhead.
//
//go:inline
func add(v1, v2 Vector3) Vector3 {
	return Vector3{v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]}
}

// Sub subtracts two vectors.
//
//go:inline
func sub(v1, v2 Vector3) Vector3 {
	return Vector3{v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]}
}

// MulScalar multiplies a vector by a scalar.
//
//go:inline
func mulScalar(v Vector3, s float64) Vector3 {
	return Vector3{v[0] * s, v[1] * s, v[2] * s}
}

// MagnitudeSquared returns the squared magnitude of a vector.
//
//go:inline
func magnitudeSquared(v Vector3) float64 {
	return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
}

// Normalize a vector.  Avoid a separate magnitude calculation if possible.
//
//go:noinline
func normalize(v Vector3) Vector3 {
	magSq := magnitudeSquared(v)
	if magSq < 1e-20 { // Increased tolerance for stability
		return Vector3{0, 0, 0}
	}
	mag := math.Sqrt(magSq) // Calculate magnitude only once.
	return Vector3{v[0] / mag, v[1] / mag, v[2] / mag}
}

// Body represents a celestial body.
type Body struct {
	Position Vector3
	Velocity Vector3
	Mass     float64
	Force    Vector3
}

// Simulation holds the N-body simulation state.
type Simulation struct {
	Bodies          []Body
	NumBodies       int
	NumIterations   int
	DT              float64
	G               float64
	PositionHistory [][]Vector3 // Use pre-allocated slice for snapshots
}

// NewSimulation creates a new simulation.
func NewSimulation(numBodies, numIterations int, dt float64, seed int64) *Simulation {
	rand.Seed(seed)

	bodies := make([]Body, numBodies)
	for i := 0; i < numBodies; i++ {
		bodies[i] = Body{
			Position: Vector3{
				rand.Float64()*2e10 - 1e10,
				rand.Float64()*2e10 - 1e10,
				rand.Float64()*2e10 - 1e10,
			},
			Velocity: Vector3{
				rand.Float64()*2e3 - 1e3,
				rand.Float64()*2e3 - 1e3,
				rand.Float64()*2e3 - 1e3,
			},
			Mass:  rand.Float64()*1e25 + 1e20,
			Force: Vector3{0, 0, 0}, // Initialize Force here
		}
	}

	// Pre-allocate PositionHistory for all iterations + initial state.
	positionHistory := make([][]Vector3, numIterations+1)
	for i := range positionHistory {
		positionHistory[i] = make([]Vector3, numBodies)
	}

	return &Simulation{
		Bodies:          bodies,
		NumBodies:       numBodies,
		NumIterations:   numIterations,
		DT:              dt,
		G:               6.67430e-11,
		PositionHistory: positionHistory,
	}
}

// ComputeForcesSequential computes forces sequentially.
func (s *Simulation) ComputeForcesSequential() {
	for i := 0; i < s.NumBodies; i++ {
		s.Bodies[i].Force = Vector3{0, 0, 0} // Reset forces at the beginning.
	}

	for i := 0; i < s.NumBodies; i++ {
		for j := i + 1; j < s.NumBodies; j++ {
			rVec := sub(s.Bodies[j].Position, s.Bodies[i].Position)
			rSquared := magnitudeSquared(rVec)
			if rSquared < 1e-20 {
				continue
			}

			// Optimization: Calculate inverse distance once
			invDist := 1.0 / math.Sqrt(rSquared)
			forceMag := s.G * s.Bodies[i].Mass * s.Bodies[j].Mass * invDist * invDist

			// Optimization: Avoid normalize by using invDist directly
			forceVec := mulScalar(rVec, forceMag*invDist)

			s.Bodies[i].Force = add(s.Bodies[i].Force, forceVec)
			s.Bodies[j].Force = sub(s.Bodies[j].Force, forceVec)
		}
	}
}

// ComputeForcesParallel computes forces in parallel using goroutines with optimized work distribution.
func (s *Simulation) ComputeForcesParallel() {
	numCPU := runtime.NumCPU()
	var wg sync.WaitGroup
	wg.Add(numCPU)

	// Reset forces
	for i := 0; i < s.NumBodies; i++ {
		s.Bodies[i].Force = Vector3{0, 0, 0}
	}

	// Optimization: Use static block partitioning for better work distribution
	blockSize := s.NumBodies / numCPU
	if blockSize == 0 {
		blockSize = 1
	}

	for w := 0; w < numCPU; w++ {
		go func(workerID int) {
			defer wg.Done()

			// Determine this worker's block of bodies
			start := workerID * blockSize
			end := start + blockSize
			if workerID == numCPU-1 {
				end = s.NumBodies // Last worker takes any remainder
			}

			// Local force accumulators to avoid mutex contention
			localForces := make([]Vector3, s.NumBodies)

			// Process all pairs where i is in this worker's block
			for i := start; i < end; i++ {
				for j := i + 1; j < s.NumBodies; j++ {
					rVec := sub(s.Bodies[j].Position, s.Bodies[i].Position)
					rSquared := magnitudeSquared(rVec)
					if rSquared < 1e-20 {
						continue
					}

					// Optimization: Calculate inverse distance once
					invDist := 1.0 / math.Sqrt(rSquared)
					forceMag := s.G * s.Bodies[i].Mass * s.Bodies[j].Mass * invDist * invDist

					// Optimization: Avoid normalize by using invDist directly
					forceVec := mulScalar(rVec, forceMag*invDist)

					// Accumulate forces locally
					localForces[i] = add(localForces[i], forceVec)
					localForces[j] = sub(localForces[j], forceVec)
				}
			}

			// Apply accumulated forces at the end with a single lock
			var mutex sync.Mutex
			mutex.Lock()
			for i := 0; i < s.NumBodies; i++ {
				s.Bodies[i].Force = add(s.Bodies[i].Force, localForces[i])
			}
			mutex.Unlock()
		}(w)
	}

	// Wait for all workers to finish
	wg.Wait()
}

// UpdatePositionsAndVelocities updates the positions and velocities.
//
//go:inline
func (s *Simulation) UpdatePositionsAndVelocities() {
	for i := range s.Bodies {
		acc := mulScalar(s.Bodies[i].Force, 1.0/s.Bodies[i].Mass)
		s.Bodies[i].Velocity = add(s.Bodies[i].Velocity, mulScalar(acc, s.DT))
		s.Bodies[i].Position = add(s.Bodies[i].Position, mulScalar(s.Bodies[i].Velocity, s.DT))
	}
}

// SavePositionSnapshot stores the current positions.
//
//go:inline
func (s *Simulation) SavePositionSnapshot(step int) {
	for i := 0; i < s.NumBodies; i++ {
		s.PositionHistory[step][i] = s.Bodies[i].Position
	}
}

// CalculateEnergy calculates the total energy of the system.
func (s *Simulation) CalculateEnergy() float64 {
	energy := 0.0

	for i := 0; i < s.NumBodies; i++ {
		velocitySquared := magnitudeSquared(s.Bodies[i].Velocity)
		energy += 0.5 * s.Bodies[i].Mass * velocitySquared

		for j := i + 1; j < s.NumBodies; j++ {
			rVec := sub(s.Bodies[j].Position, s.Bodies[i].Position)
			rSquared := magnitudeSquared(rVec)
			if rSquared < 1e-20 {
				continue
			}
			// Optimization: Calculate inverse distance once
			invDist := 1.0 / math.Sqrt(rSquared)
			energy -= s.G * s.Bodies[i].Mass * s.Bodies[j].Mass * invDist
		}
	}
	return energy
}

// RunSimulation runs the simulation.
func (s *Simulation) RunSimulation(useParallel bool) time.Duration {
	startTime := time.Now()
	s.SavePositionSnapshot(0) // Save initial state at index 0.

	for i := 0; i < s.NumIterations; i++ {
		if useParallel {
			s.ComputeForcesParallel()
		} else {
			s.ComputeForcesSequential()
		}
		s.UpdatePositionsAndVelocities()
		s.SavePositionSnapshot(i + 1) // Save positions at index i+1
	}

	return time.Since(startTime)
}

func main() {
	numBodiesPtr := flag.Int("bodies", 2000, "Number of bodies")
	numIterationsPtr := flag.Int("iterations", 1000, "Number of iterations")
	dtPtr := flag.Float64("dt", 0.01, "Time step")
	seedPtr := flag.Int64("seed", 42, "Random seed")
	parallelPtr := flag.Bool("parallel", true, "Use parallel execution")
	benchmark_csv := flag.String("benchmark_csv", "", "Print execution time in CSV file")
	flag.Parse()

	sim := NewSimulation(*numBodiesPtr, *numIterationsPtr, *dtPtr, *seedPtr)
	initialEnergy := sim.CalculateEnergy()
	fmt.Printf("Initial system energy: %.6e\n", initialEnergy)

	executionTime := sim.RunSimulation(*parallelPtr)

	finalEnergy := sim.CalculateEnergy()
	fmt.Printf("Final system energy: %.6e\n", finalEnergy)
	fmt.Printf("Energy difference: %.6e\n", finalEnergy-initialEnergy)

	fmt.Println("Go Implementation")
	if *parallelPtr {
		fmt.Println("Mode: Parallel")
	} else {
		fmt.Println("Mode: Sequential")
	}
	fmt.Printf("Number of bodies: %d\n", *numBodiesPtr)
	fmt.Printf("Number of iterations: %d\n", *numIterationsPtr)
	fmt.Printf("Number of CPUs: %d\n", runtime.NumCPU())
	fmt.Printf("Execution time: %.4f seconds\n", executionTime.Seconds())

	if len(*benchmark_csv) > 0 {
		// Open the file for appending or create it if it doesn't exist
		file, err := os.OpenFile(*benchmark_csv, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			fmt.Println("Error opening file:", err)
			return
		}
		defer file.Close()

		// Convert the float to a string
		floatString := strconv.FormatFloat(executionTime.Seconds(), 'f', 6, 64)

		// Write the float to the file followed by a newline
		_, err = fmt.Fprintf(file, "go,%s\n", floatString)
		if err != nil {
			fmt.Println("Error writing to file:", err)
			return
		}
	}
}
