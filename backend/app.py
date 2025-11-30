"""
Quantum Computing Optimizer SaaS - Backend Service
A simple API for quantum-inspired optimization algorithms.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.optimize import minimize

app = Flask(__name__)
CORS(app)


class QuantumInspiredOptimizer:
    """
    Quantum-inspired optimizer using simulated quantum annealing principles.
    This is a classical simulation that mimics quantum behavior for optimization.
    """

    def __init__(self, num_iterations=1000, temperature=1.0, cooling_rate=0.995):
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.cooling_rate = cooling_rate

    def optimize(self, objective_function, initial_state, bounds=None):
        """
        Perform quantum-inspired optimization.
        
        Args:
            objective_function: Function to minimize
            initial_state: Starting point for optimization
            bounds: Optional bounds for the variables
            
        Returns:
            dict: Optimization result with optimal state and value
        """
        current_state = np.array(initial_state, dtype=float)
        current_energy = objective_function(current_state)
        best_state = current_state.copy()
        best_energy = current_energy
        temp = self.temperature

        history = [{"iteration": 0, "energy": float(current_energy)}]

        for i in range(self.num_iterations):
            # Generate a neighbor state using quantum-inspired superposition
            perturbation = np.random.normal(0, temp, size=current_state.shape)
            new_state = current_state + perturbation

            # Apply bounds if provided
            if bounds:
                for j, (low, high) in enumerate(bounds):
                    new_state[j] = np.clip(new_state[j], low, high)

            new_energy = objective_function(new_state)
            delta_energy = new_energy - current_energy

            # Accept or reject based on quantum tunneling probability
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temp):
                current_state = new_state
                current_energy = new_energy

                if current_energy < best_energy:
                    best_state = current_state.copy()
                    best_energy = current_energy

            # Cool down (simulating decoherence)
            temp *= self.cooling_rate

            if (i + 1) % 100 == 0:
                history.append({"iteration": i + 1, "energy": float(best_energy)})

        return {
            "optimal_state": best_state.tolist(),
            "optimal_value": float(best_energy),
            "history": history,
            "iterations": self.num_iterations
        }


# Predefined objective functions for demo purposes
OBJECTIVE_FUNCTIONS = {
    "quadratic": lambda x: np.sum(x ** 2),
    "rosenbrock": lambda x: sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
                                for i in range(len(x) - 1)) if len(x) > 1 else x[0]**2,
    "rastrigin": lambda x: 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x),
    "ackley": lambda x: -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) 
                        - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e
}


@app.route("/")
def index():
    """Health check endpoint."""
    return jsonify({
        "service": "Quantum Computing Optimizer SaaS",
        "version": "1.0.0",
        "status": "running"
    })


@app.route("/api/optimize", methods=["POST"])
def optimize():
    """
    Main optimization endpoint.
    
    Request body:
        - function: Name of the objective function or 'custom'
        - initial_state: Initial state vector
        - bounds: Optional bounds for each variable [[low, high], ...]
        - iterations: Number of iterations (default: 1000)
        - temperature: Initial temperature (default: 1.0)
        - cooling_rate: Cooling rate (default: 0.995)
    
    Returns:
        Optimization results including optimal state and value
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Get parameters
    func_name = data.get("function", "quadratic")
    initial_state = data.get("initial_state", [1.0, 1.0])
    bounds = data.get("bounds")
    iterations = data.get("iterations", 1000)
    temperature = data.get("temperature", 1.0)
    cooling_rate = data.get("cooling_rate", 0.995)
    
    # Validate iterations
    if not isinstance(iterations, int) or iterations < 1 or iterations > 100000:
        return jsonify({"error": "iterations must be an integer between 1 and 100000"}), 400
    
    # Get objective function
    if func_name not in OBJECTIVE_FUNCTIONS:
        return jsonify({
            "error": f"Unknown function: {func_name}",
            "available_functions": list(OBJECTIVE_FUNCTIONS.keys())
        }), 400
    
    objective_function = OBJECTIVE_FUNCTIONS[func_name]
    
    # Create optimizer and run
    optimizer = QuantumInspiredOptimizer(
        num_iterations=iterations,
        temperature=temperature,
        cooling_rate=cooling_rate
    )
    
    result = optimizer.optimize(
        objective_function=objective_function,
        initial_state=initial_state,
        bounds=bounds
    )
    
    return jsonify({
        "success": True,
        "function": func_name,
        "result": result
    })


@app.route("/api/functions", methods=["GET"])
def list_functions():
    """List available objective functions."""
    functions = {
        "quadratic": {
            "name": "Quadratic",
            "description": "Simple sum of squares: f(x) = Σx²",
            "global_minimum": "0 at x = [0, 0, ...]"
        },
        "rosenbrock": {
            "name": "Rosenbrock",
            "description": "Classic optimization benchmark function",
            "global_minimum": "0 at x = [1, 1, ...]"
        },
        "rastrigin": {
            "name": "Rastrigin",
            "description": "Highly multimodal function with many local minima",
            "global_minimum": "0 at x = [0, 0, ...]"
        },
        "ackley": {
            "name": "Ackley",
            "description": "Complex function with exponential and cosine terms",
            "global_minimum": "0 at x = [0, 0, ...]"
        }
    }
    return jsonify({"functions": functions})


@app.route("/api/compare", methods=["POST"])
def compare_methods():
    """
    Compare quantum-inspired optimization with classical methods.
    
    Request body:
        - function: Name of the objective function
        - initial_state: Initial state vector
        - bounds: Optional bounds
    
    Returns:
        Comparison of results from different methods
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    func_name = data.get("function", "quadratic")
    initial_state = data.get("initial_state", [1.0, 1.0])
    bounds = data.get("bounds")
    
    if func_name not in OBJECTIVE_FUNCTIONS:
        return jsonify({"error": f"Unknown function: {func_name}"}), 400
    
    objective_function = OBJECTIVE_FUNCTIONS[func_name]
    
    # Quantum-inspired optimization
    optimizer = QuantumInspiredOptimizer()
    quantum_result = optimizer.optimize(objective_function, initial_state, bounds)
    
    # Classical optimization using scipy
    scipy_bounds = [(b[0], b[1]) for b in bounds] if bounds else None
    classical_result = minimize(
        objective_function,
        initial_state,
        method='L-BFGS-B',
        bounds=scipy_bounds
    )
    
    return jsonify({
        "success": True,
        "function": func_name,
        "quantum_inspired": {
            "optimal_state": quantum_result["optimal_state"],
            "optimal_value": quantum_result["optimal_value"]
        },
        "classical_lbfgsb": {
            "optimal_state": classical_result.x.tolist(),
            "optimal_value": float(classical_result.fun)
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
