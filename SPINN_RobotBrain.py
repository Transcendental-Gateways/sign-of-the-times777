"""
SPINN Robot Brain - Syntropic Perennial Intelligence Neural Network
Reconfigured for autonomous robot control and decision making
"""

import numpy as np
from scipy.integrate import odeint
from sklearn.cluster import KMeans
import random
from math import factorial, log
import json
from datetime import datetime
from collections import deque
import threading
import time

# ============================================================================
# CORE SNN MODULES
# ============================================================================

def lorenz_system(state, t, sigma=10, beta=8/3, rho=28):
    """Chaotic attractor for adaptive behavior generation"""
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

class PIDController:
    """Motor control and servo stabilization"""
    def __init__(self, Kp=1.0, Ki=0.1, Kd=0.05, set_point=0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.set_point = set_point
        self.integral = 0
        self.last_error = None

    def control(self, measurement, dt=0.1):
        error = self.set_point - measurement
        self.integral += error * dt
        derivative = 0 if self.last_error is None else (error - self.last_error) / dt
        self.last_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return np.clip(output, -100, 100)  # Limit motor commands

    def reset(self):
        self.integral = 0
        self.last_error = None

class SynapticTrace:
    """Convert discrete SNN spikes to continuous analog signal"""
    def __init__(self, alpha=0.85):
        self.alpha = alpha  # Leak rate: 0.95=long memory, 0.50=short memory
        self.trace_value = 0.0
    
    def update(self, spike):
        """
        Decode spike to continuous trace
        Args:
            spike: 1 if firing, 0 if silent
        Returns:
            Continuous analog value for Kalman filter
        """
        # Exponential decay + new spike integration
        self.trace_value = (self.trace_value * self.alpha) + spike
        return self.trace_value
    
    def reset(self):
        self.trace_value = 0.0


class SensorFusion:
    """Multi-sensor integration and pattern recognition"""
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.sensor_history = deque(maxlen=100)
        self.fitted = False
        
        # Synaptic trace decoder for each sensor channel
        self.synaptic_traces = [SynapticTrace(alpha=0.85) for _ in range(3)]

    def process_sensors(self, sensor_data):
        """Process raw sensor readings into actionable patterns with synaptic trace decoding"""
        # Convert discrete sensor readings to continuous traces
        # Treat sensor readings as spike trains (normalized to 0-1)
        spikes = sensor_data / (np.max(sensor_data) + 1e-6)
        
        # Decode each sensor channel through synaptic trace
        continuous_signals = np.array([
            self.synaptic_traces[i].update(spikes[i]) 
            for i in range(min(len(spikes), len(self.synaptic_traces)))
        ])
        
        # Use continuous analog signal for clustering (smoother, more stable)
        sensor_mean = np.mean(continuous_signals)
        self.sensor_history.append(sensor_mean)
        
        if len(self.sensor_history) >= 10:
            data_array = np.array(list(self.sensor_history)).reshape(-1, 1)
            
            if not self.fitted:
                self.kmeans.fit(data_array)
                self.fitted = True
            
            cluster = self.kmeans.predict([[sensor_mean]])
            distances = np.linalg.norm(self.kmeans.cluster_centers_ - sensor_mean, axis=1)
            confidence = float(np.min(distances))
            
            return {
                'cluster': int(cluster[0]),
                'confidence': confidence,
                'pattern': 'obstacle' if cluster[0] < 2 else 'clear',
                'trace_values': continuous_signals.tolist()  # Continuous analog signals
            }
        
        return {
            'cluster': -1, 
            'confidence': 0.0, 
            'pattern': 'calibrating',
            'trace_values': continuous_signals.tolist()
        }

# ============================================================================
# MORPHOGENIC BEHAVIOR GENERATION
# ============================================================================

BEHAVIOR_PRIMITIVES = ['forward', 'backward', 'left', 'right', 'stop', 'scan']
MODIFIERS = ['fast', 'slow', 'careful', 'aggressive']

def generate_behavior_sequences(iterations=5, n_sequences=8):
    """Evolve complex behavior patterns from primitives"""
    sequences = [[random.choice(BEHAVIOR_PRIMITIVES)] for _ in range(n_sequences)]
    
    for i in range(iterations):
        new_sequences = []
        for seq in sequences:
            if len(seq) < 6:  # Max sequence length
                action = random.choice(BEHAVIOR_PRIMITIVES)
                modifier = random.choice(MODIFIERS) if random.random() > 0.5 else None
                new_sequences.append(seq + [(action, modifier) if modifier else action])
            else:
                new_sequences.append(seq)
        sequences = new_sequences
    
    return sequences

def syntropy_field(phi_0, t, lambda_damp=0.2):
    """Temporal coherence for smooth motor transitions"""
    def dphi_dt(phi, t):
        return -lambda_damp * phi + np.random.normal(0, 0.1, len(phi))
    return odeint(dphi_dt, phi_0, t)

def perennial_morphogenic(t, omega=2, n_terms=5):
    """Cyclical behavior patterns (patrol, search, etc.)"""
    fib = [0, 1]
    for _ in range(2, n_terms + 1):
        fib.append(fib[-1] + fib[-2])
    return sum((fib[k] / factorial(k)) * np.exp(-1j * omega * t) for k in range(n_terms)).real

# ============================================================================
# ROBOT BRAIN CORE
# ============================================================================

class RobotBrain:
    """SPINN-based autonomous robot control system"""
    
    def __init__(self):
        print("🤖 Initializing Robot Brain...")
        
        # Core neural substrate
        self.syntropic_field = np.zeros(100)
        self.morphogenic_memory = np.zeros(50)
        
        # Control systems
        self.motor_left = PIDController(Kp=2.0, Ki=0.3, Kd=0.1)
        self.motor_right = PIDController(Kp=2.0, Ki=0.3, Kd=0.1)
        self.sensor_fusion = SensorFusion(n_clusters=5)
        
        # Behavior system
        self.behavior_library = generate_behavior_sequences(5, 8)
        self.current_behavior = None
        self.behavior_state = 'idle'
        
        # Environmental model
        self.world_model = {
            'obstacles': [],
            'position': np.array([0.0, 0.0]),
            'orientation': 0.0,
            'velocity': np.array([0.0, 0.0])
        }
        
        # Decision making
        self.decision_buffer = deque(maxlen=10)
        self.action_history = deque(maxlen=50)
        
        # Real-time loop
        self.running = False
        self.brain_thread = None
        
        print("✓ Robot Brain online")
    
    def perceive(self, sensor_data):
        """Process sensory input through SNN with synaptic trace decoding"""
        # Sensor fusion with continuous trace signals
        perception = self.sensor_fusion.process_sensors(sensor_data)
        
        # Use continuous trace values for smoother field updates
        trace_signals = np.array(perception.get('trace_values', sensor_data[:3]))
        
        # Update syntropic field with sensor harmonics (using decoded traces)
        vibes = np.array([np.sin(omega * np.arange(len(trace_signals)) + phi) 
                         for omega, phi in zip([1, 2, 3], [0, np.pi/2, np.pi])])
        self.syntropic_field[:len(trace_signals)] = trace_signals * np.mean(vibes, axis=0)[:len(trace_signals)]
        
        # Lorenz dynamics for adaptive response (using continuous signals)
        t = np.linspace(0, 1, 10)
        initial = [perception['confidence'], perception['cluster'], np.mean(trace_signals)]
        chaos = odeint(lorenz_system, initial, t)
        
        return {
            'perception': perception,
            'field_energy': float(np.mean(self.syntropic_field**2)),
            'chaos_trajectory': chaos[-1].tolist(),
            'threat_level': float(np.clip(perception['cluster'] / 5.0, 0, 1)),
            'trace_signals': trace_signals.tolist()  # Smooth continuous signals
        }
    
    def think(self, perception):
        """High-level decision making through morphogenic evolution"""
        self.decision_buffer.append(perception)
        
        # Evolve behavior based on recent perceptions
        threat_avg = np.mean([p.get('threat_level', 0) for p in self.decision_buffer])
        
        # Select behavior from evolved library
        if threat_avg > 0.6:
            decision = 'avoid_obstacle'
            behavior_idx = 0  # Defensive behaviors
        elif threat_avg < 0.3:
            decision = 'explore'
            behavior_idx = min(len(self.behavior_library) - 1, 5)
        else:
            decision = 'navigate_careful'
            behavior_idx = 2
        
        self.current_behavior = self.behavior_library[behavior_idx]
        
        # Apply perennial morphogenic field for temporal coherence
        t = np.linspace(0, 1, 20)
        morph_signal = perennial_morphogenic(t, omega=2)
        self.morphogenic_memory = morph_signal[:50]
        
        return {
            'decision': decision,
            'behavior_sequence': self.current_behavior,
            'confidence': float(1.0 - threat_avg),
            'morphogenic_phase': float(np.mean(self.morphogenic_memory))
        }
    
    def act(self, decision, motor_mode='differential'):
        """Execute motor commands through PID control"""
        actions = {
            'avoid_obstacle': {'left': -50, 'right': 50},  # Turn right
            'explore': {'left': 70, 'right': 70},          # Move forward
            'navigate_careful': {'left': 40, 'right': 40}, # Slow forward
            'stop': {'left': 0, 'right': 0}
        }
        
        target = actions.get(decision['decision'], {'left': 0, 'right': 0})
        
        # PID control for smooth motor actuation
        left_cmd = self.motor_left.control(target['left'])
        right_cmd = self.motor_right.control(target['right'])
        
        # Apply syntropy field damping for smooth transitions
        t = np.linspace(0, 0.5, 10)
        damp = syntropy_field(np.array([left_cmd, right_cmd]), t)
        
        motor_output = {
            'left_motor': float(damp[-1, 0]),
            'right_motor': float(damp[-1, 1]),
            'mode': motor_mode,
            'timestamp': time.time()
        }
        
        self.action_history.append(motor_output)
        
        return motor_output
    
    def brain_loop(self, sensor_callback, motor_callback, hz=10):
        """Real-time brain processing loop"""
        dt = 1.0 / hz
        
        print(f"🧠 Brain loop started at {hz}Hz")
        
        while self.running:
            loop_start = time.time()
            
            # Perception
            sensor_data = sensor_callback()
            perception = self.perceive(sensor_data)
            
            # Cognition
            decision = self.think(perception)
            
            # Action
            motor_cmd = self.act(decision)
            motor_callback(motor_cmd)
            
            # Maintain loop timing
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
        
        print("🛑 Brain loop stopped")
    
    def start(self, sensor_callback, motor_callback, hz=10):
        """Start autonomous operation"""
        if not self.running:
            self.running = True
            self.brain_thread = threading.Thread(
                target=self.brain_loop, 
                args=(sensor_callback, motor_callback, hz)
            )
            self.brain_thread.start()
    
    def stop(self):
        """Stop autonomous operation"""
        self.running = False
        if self.brain_thread:
            self.brain_thread.join()
    
    def get_status(self):
        """Get current brain state"""
        return {
            'running': self.running,
            'behavior': self.behavior_state,
            'field_coherence': float(np.var(self.syntropic_field)),
            'morphogenic_phase': float(np.mean(self.morphogenic_memory)),
            'decisions_queued': len(self.decision_buffer),
            'actions_taken': len(self.action_history)
        }

# ============================================================================
# SIMULATION INTERFACE
# ============================================================================

class RobotSimulator:
    """Simulated robot for testing the brain"""
    
    def __init__(self):
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.obstacles = [
            np.array([5.0, 5.0]),
            np.array([-3.0, 8.0]),
            np.array([10.0, -2.0])
        ]
        
    def get_sensors(self):
        """Simulate distance sensors (front, left, right)"""
        sensors = []
        for angle in [-45, 0, 45]:
            direction = self.orientation + np.radians(angle)
            ray = np.array([np.cos(direction), np.sin(direction)])
            
            # Find nearest obstacle
            min_dist = 100.0
            for obs in self.obstacles:
                dist = np.linalg.norm(obs - self.position)
                if dist < min_dist:
                    min_dist = dist
            
            sensors.append(min_dist + np.random.normal(0, 0.1))
        
        return np.array(sensors)
    
    def set_motors(self, motor_cmd):
        """Update robot state from motor commands"""
        left = motor_cmd['left_motor'] / 100.0
        right = motor_cmd['right_motor'] / 100.0
        
        # Differential drive kinematics
        v = (left + right) / 2.0
        omega = (right - left) / 2.0
        
        dt = 0.1
        self.position += dt * v * np.array([np.cos(self.orientation), np.sin(self.orientation)])
        self.orientation += dt * omega
        
        return {'x': float(self.position[0]), 'y': float(self.position[1]), 'theta': float(self.orientation)}

# ============================================================================
# TESTING
# ============================================================================

def test_robot_brain():
    """Test robot brain in simulation"""
    print("=" * 60)
    print("SPINN ROBOT BRAIN TEST")
    print("=" * 60)
    
    brain = RobotBrain()
    sim = RobotSimulator()
    
    print("\n🎮 Running simulation for 5 seconds...")
    
    brain.start(
        sensor_callback=lambda: sim.get_sensors(),
        motor_callback=lambda cmd: sim.set_motors(cmd),
        hz=10
    )
    
    time.sleep(5)
    
    brain.stop()
    
    print("\n📊 Final Status:")
    status = brain.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print(f"\n📍 Final Position: ({sim.position[0]:.2f}, {sim.position[1]:.2f})")
    print(f"📐 Final Orientation: {np.degrees(sim.orientation):.1f}°")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_robot_brain()
