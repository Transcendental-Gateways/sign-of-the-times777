"""
Unit tests for the Quantum Computing Optimizer SaaS backend.
"""

import pytest
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import app, QuantumInspiredOptimizer, OBJECTIVE_FUNCTIONS


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthCheck:
    """Tests for the health check endpoint."""
    
    def test_index_returns_status(self, client):
        """Test that the index endpoint returns service status."""
        response = client.get('/')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['service'] == 'Quantum Computing Optimizer SaaS'
        assert data['status'] == 'running'
        assert 'version' in data


class TestOptimizeEndpoint:
    """Tests for the /api/optimize endpoint."""
    
    def test_optimize_quadratic(self, client):
        """Test optimization with quadratic function."""
        response = client.post('/api/optimize', 
            data=json.dumps({
                'function': 'quadratic',
                'initial_state': [5.0, 5.0],
                'iterations': 500
            }),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'result' in data
        # The optimal value should be close to 0 (global minimum)
        assert data['result']['optimal_value'] < 1.0
    
    def test_optimize_with_bounds(self, client):
        """Test optimization with bounds."""
        response = client.post('/api/optimize',
            data=json.dumps({
                'function': 'quadratic',
                'initial_state': [3.0, 3.0],
                'bounds': [[-5, 5], [-5, 5]],
                'iterations': 500
            }),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        # Check bounds are respected
        for val in data['result']['optimal_state']:
            assert -5 <= val <= 5
    
    def test_optimize_invalid_function(self, client):
        """Test optimization with invalid function name."""
        response = client.post('/api/optimize',
            data=json.dumps({
                'function': 'invalid_function',
                'initial_state': [1.0, 1.0]
            }),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_optimize_no_data(self, client):
        """Test optimization with no data provided."""
        response = client.post('/api/optimize', content_type='application/json')
        assert response.status_code == 400
    
    def test_optimize_invalid_iterations(self, client):
        """Test optimization with invalid iterations."""
        response = client.post('/api/optimize',
            data=json.dumps({
                'function': 'quadratic',
                'initial_state': [1.0, 1.0],
                'iterations': 200000
            }),
            content_type='application/json'
        )
        assert response.status_code == 400


class TestFunctionsEndpoint:
    """Tests for the /api/functions endpoint."""
    
    def test_list_functions(self, client):
        """Test listing available functions."""
        response = client.get('/api/functions')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'functions' in data
        assert 'quadratic' in data['functions']
        assert 'rosenbrock' in data['functions']
        assert 'rastrigin' in data['functions']
        assert 'ackley' in data['functions']


class TestCompareEndpoint:
    """Tests for the /api/compare endpoint."""
    
    def test_compare_methods(self, client):
        """Test comparison between quantum-inspired and classical methods."""
        response = client.post('/api/compare',
            data=json.dumps({
                'function': 'quadratic',
                'initial_state': [2.0, 2.0]
            }),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'quantum_inspired' in data
        assert 'classical_lbfgsb' in data
    
    def test_compare_no_data(self, client):
        """Test comparison with no data provided."""
        response = client.post('/api/compare', content_type='application/json')
        assert response.status_code == 400


class TestQuantumInspiredOptimizer:
    """Tests for the QuantumInspiredOptimizer class."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization with custom parameters."""
        optimizer = QuantumInspiredOptimizer(
            num_iterations=500,
            temperature=2.0,
            cooling_rate=0.99
        )
        assert optimizer.num_iterations == 500
        assert optimizer.temperature == 2.0
        assert optimizer.cooling_rate == 0.99
    
    def test_optimizer_finds_minimum(self):
        """Test that optimizer finds approximate minimum for quadratic function."""
        optimizer = QuantumInspiredOptimizer(num_iterations=1000)
        result = optimizer.optimize(
            objective_function=OBJECTIVE_FUNCTIONS['quadratic'],
            initial_state=[5.0, 5.0],
            bounds=[[-10, 10], [-10, 10]]
        )
        # Should find approximate minimum near 0
        assert result['optimal_value'] < 1.0
        assert len(result['optimal_state']) == 2
        assert 'history' in result
    
    def test_optimizer_respects_bounds(self):
        """Test that optimizer respects variable bounds."""
        optimizer = QuantumInspiredOptimizer(num_iterations=500)
        bounds = [[0, 1], [0, 1]]
        result = optimizer.optimize(
            objective_function=OBJECTIVE_FUNCTIONS['quadratic'],
            initial_state=[0.5, 0.5],
            bounds=bounds
        )
        for i, val in enumerate(result['optimal_state']):
            assert bounds[i][0] <= val <= bounds[i][1]


class TestObjectiveFunctions:
    """Tests for the predefined objective functions."""
    
    def test_quadratic_at_origin(self):
        """Test quadratic function is zero at origin."""
        result = OBJECTIVE_FUNCTIONS['quadratic'](np.array([0.0, 0.0]))
        assert result == 0.0
    
    def test_rosenbrock_at_minimum(self):
        """Test Rosenbrock function at known minimum."""
        result = OBJECTIVE_FUNCTIONS['rosenbrock'](np.array([1.0, 1.0]))
        assert abs(result) < 1e-10
    
    def test_rastrigin_at_origin(self):
        """Test Rastrigin function at origin."""
        result = OBJECTIVE_FUNCTIONS['rastrigin'](np.array([0.0, 0.0]))
        assert abs(result) < 1e-10
    
    def test_ackley_at_origin(self):
        """Test Ackley function at origin."""
        result = OBJECTIVE_FUNCTIONS['ackley'](np.array([0.0, 0.0]))
        assert abs(result) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
