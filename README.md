# Quantum Computing Optimizer SaaS

A quantum-inspired optimization service that leverages quantum annealing principles to solve complex optimization problems.

## Features

- **Quantum-Inspired Optimization**: Uses simulated quantum annealing for efficient exploration of solution spaces
- **Multiple Objective Functions**: Supports Quadratic, Rosenbrock, Rastrigin, and Ackley benchmark functions
- **Comparison Mode**: Compare quantum-inspired results with classical L-BFGS-B optimization
- **RESTful API**: Easy-to-use JSON API for integration
- **Web Interface**: Interactive web-based UI for running optimizations

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)

### Running Locally

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Start the backend server:
```bash
python app.py
```

For development with debug mode enabled:
```bash
FLASK_DEBUG=true python app.py
```

3. Open `frontend/index.html` in your browser or serve it with any static file server.

### Running with Docker

```bash
docker-compose up --build
```

The API will be available at `http://localhost:5000` and the frontend at `http://localhost`.

## API Endpoints

### GET /
Health check endpoint.

### GET /api/functions
List all available objective functions.

### POST /api/optimize
Run quantum-inspired optimization.

**Request Body:**
```json
{
  "function": "quadratic",
  "initial_state": [1.0, 1.0],
  "bounds": [[-5, 5], [-5, 5]],
  "iterations": 1000,
  "temperature": 1.0,
  "cooling_rate": 0.995
}
```

**Response:**
```json
{
  "success": true,
  "function": "quadratic",
  "result": {
    "optimal_state": [0.001, -0.002],
    "optimal_value": 0.000005,
    "iterations": 1000,
    "history": [...]
  }
}
```

### POST /api/compare
Compare quantum-inspired optimization with classical methods.

**Request Body:**
```json
{
  "function": "rosenbrock",
  "initial_state": [2.0, 2.0],
  "bounds": [[-5, 5], [-5, 5]]
}
```

## Supported Objective Functions

| Function | Description | Global Minimum |
|----------|-------------|----------------|
| quadratic | Sum of squares: f(x) = Σx² | 0 at x = [0, ...] |
| rosenbrock | Classic optimization benchmark | 0 at x = [1, 1, ...] |
| rastrigin | Highly multimodal function | 0 at x = [0, ...] |
| ackley | Complex exponential/cosine function | 0 at x = [0, ...] |

## Algorithm

The quantum-inspired optimizer uses simulated quantum annealing:

1. **Initialization**: Start with an initial state and high temperature
2. **Perturbation**: Generate neighbor states using Gaussian noise scaled by temperature
3. **Acceptance**: Accept new states based on energy improvement or quantum tunneling probability
4. **Cooling**: Gradually reduce temperature to simulate decoherence
5. **Convergence**: Return the best state found

## Testing

Run tests with pytest:
```bash
pip install pytest
pytest tests/ -v
```

## License

MIT License
