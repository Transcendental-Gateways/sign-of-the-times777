"""
© 2026 Transcendental Gateways. All rights reserved.
Use of this software is governed by the LICENSE-RESEARCH.md file.

SPINN Robot Brain Web API
Real-time control interface for robot brain operations
"""

from flask import Flask, jsonify, request, render_template
from SPINN_RobotBrain import RobotBrain, RobotSimulator
from safety_layer import SafetyMonitor, SafetyConstraints
from lorenz_kalman import LorenzEnhancedKalmanFilter
import numpy as np
import threading
import time

app = Flask(__name__)

# Global robot brain and simulator
brain = None
simulator = None
safety_monitor = None
kalman_filter = None
brain_active = False
last_perception = None
last_action = None

@app.route('/')
def index():
    return render_template('robot.html')

@app.route('/api/robot/init', methods=['POST'])
def init_robot():
    """Initialize robot brain"""
    global brain, simulator, safety_monitor, kalman_filter
    
    brain = RobotBrain()
    simulator = RobotSimulator()
    safety_monitor = SafetyMonitor(SafetyConstraints())
    kalman_filter = LorenzEnhancedKalmanFilter(state_dim=4, measurement_dim=2)
    
    return jsonify({
        'status': 'initialized',
        'message': 'Robot brain with Lorenz-UKF ready'
    })

@app.route('/api/robot/start', methods=['POST'])
def start_robot():
    """Start autonomous operation with safety monitoring"""
    global brain, simulator, brain_active, safety_monitor, last_perception, last_action, kalman_filter
    
    if brain is None:
        return jsonify({'error': 'Brain not initialized'}), 400
    
    if not brain_active:
        def safe_sensor_callback():
            return simulator.get_sensors()
        
        def safe_motor_callback(cmd):
            global last_perception, last_action, kalman_filter
            # SNN-optimized Kalman filter update
            if kalman_filter and last_perception:
                # Extract spike activity from synaptic traces
                trace_signals = last_perception.get('trace_signals', [0, 0, 0])
                spike_rate = np.mean(trace_signals)  # Average spike activity
                
                # Predict with spike-driven Lorenz stabilization
                kalman_filter.predict(dt=0.1, spike_input=spike_rate)
                
                # Update with position measurement and spike rate
                measurement = simulator.position[:2]
                kalman_filter.update(measurement, synaptic_trace=True, spike_rate=spike_rate)
            
            # Apply safety layer
            if safety_monitor:
                safety_monitor.pet_watchdog()
                sensors = simulator.get_sensors()
                safe_left, safe_right = safety_monitor.enforce_safe_command(
                    cmd['left_motor'],
                    cmd['right_motor'],
                    sensors
                )
                cmd['left_motor'] = safe_left
                cmd['right_motor'] = safe_right
            
            last_action = cmd
            return simulator.set_motors(cmd)
        
        # Override brain perceive to capture metrics
        original_perceive = brain.perceive
        def tracked_perceive(sensors):
            global last_perception
            result = original_perceive(sensors)
            last_perception = result
            return result
        brain.perceive = tracked_perceive
        
        brain.start(
            sensor_callback=safe_sensor_callback,
            motor_callback=safe_motor_callback,
            hz=10
        )
        brain_active = True
        
        return jsonify({
            'status': 'running',
            'message': 'Robot brain started with Lorenz-UKF and safety monitoring'
        })
    
    return jsonify({'status': 'already_running'})

@app.route('/api/robot/stop', methods=['POST'])
def stop_robot():
    """Stop autonomous operation"""
    global brain, brain_active
    
    if brain and brain_active:
        brain.stop()
        brain_active = False
        
        return jsonify({
            'status': 'stopped',
            'message': 'Robot brain stopped'
        })
    
    return jsonify({'status': 'not_running'})

@app.route('/api/robot/status', methods=['GET'])
def get_status():
    """Get current robot status with all metrics"""
    global brain, simulator, brain_active, safety_monitor, last_perception, last_action
    
    if brain is None:
        return jsonify({'error': 'Brain not initialized'}), 400
    
    brain_status = brain.get_status()
    
    position = {
        'x': float(simulator.position[0]),
        'y': float(simulator.position[1]),
        'orientation': float(np.degrees(simulator.orientation))
    }
    
    sensors = simulator.get_sensors()
    
    # Get latest perception if available
    if brain_active and last_perception:
        snn_metrics = {
            'synaptic_traces': last_perception.get('trace_signals', [0, 0, 0]),
            'lorenz_state': last_perception.get('chaos_trajectory', [0, 0, 0]),
            'field_energy': last_perception.get('field_energy', 0.0),
            'threat_level': last_perception.get('threat_level', 0.0)
        }
    else:
        snn_metrics = {
            'synaptic_traces': [0, 0, 0],
            'lorenz_state': [0, 0, 0],
            'field_energy': 0.0,
            'threat_level': 0.0
        }
    
    # Get safety metrics
    if safety_monitor:
        safety_status = safety_monitor.get_status()
        safety_metrics = {
            'safety_level': safety_status['safety_level'],
            'motor_left': last_action.get('left_motor', 0.0) if last_action else 0.0,
            'motor_right': last_action.get('right_motor', 0.0) if last_action else 0.0,
            'velocity': float(np.linalg.norm(simulator.position) / 10.0) if simulator else 0.0,
            'min_obstacle': float(np.min(sensors)),
            'watchdog_ok': safety_status['watchdog_ok'],
            'emergency_stop': safety_status['emergency_stop'],
            'failure_count': len(safety_status['active_failures'])
        }
    else:
        safety_metrics = {
            'safety_level': 'UNKNOWN',
            'motor_left': 0.0,
            'motor_right': 0.0,
            'velocity': 0.0,
            'min_obstacle': 0.0,
            'watchdog_ok': True,
            'emergency_stop': False,
            'failure_count': 0
        }
    
    # Get Kalman filter metrics with SNN optimization
    if kalman_filter:
        kalman_state = kalman_filter.get_state()
        snn_opt = kalman_filter.get_snn_optimization_metrics()
        kalman_metrics = {
            'estimated_pos_x': float(kalman_state[0]),
            'estimated_pos_y': float(kalman_state[1]),
            'estimated_vel_x': float(kalman_state[2]),
            'estimated_vel_y': float(kalman_state[3]),
            'position_uncertainty': float(kalman_filter.get_position_uncertainty()),
            'velocity_uncertainty': float(kalman_filter.get_velocity_uncertainty()),
            'lorenz_state': [float(x) for x in kalman_filter.lorenz_state],
            'lorenz_magnitude': float(np.linalg.norm(kalman_filter.lorenz_state)),
            'snn_spike_trace': snn_opt['spike_trace'],
            'snn_current_rho': snn_opt['current_rho'],
            'snn_stabilization_active': snn_opt['stabilization_active'],
            'chaos_to_order_ratio': snn_opt['chaos_to_order_ratio']
        }
    else:
        kalman_metrics = {
            'estimated_pos_x': 0.0,
            'estimated_pos_y': 0.0,
            'estimated_vel_x': 0.0,
            'estimated_vel_y': 0.0,
            'position_uncertainty': 0.0,
            'velocity_uncertainty': 0.0,
            'lorenz_state': [0.0, 0.0, 0.0],
            'lorenz_magnitude': 0.0,
            'snn_spike_trace': 0.0,
            'snn_current_rho': 28.0,
            'snn_stabilization_active': False,
            'chaos_to_order_ratio': 1.0
        }
    
    return jsonify({
        'brain': brain_status,
        'position': position,
        'sensors': sensors.tolist(),
        'active': brain_active,
        'snn_metrics': snn_metrics,
        'safety_metrics': safety_metrics,
        'kalman_metrics': kalman_metrics
    })

@app.route('/api/robot/manual', methods=['POST'])
def manual_control():
    """Manual motor control override"""
    global simulator
    
    data = request.json
    left = data.get('left', 0)
    right = data.get('right', 0)
    
    if simulator:
        motor_cmd = {'left_motor': left, 'right_motor': right}
        pos = simulator.set_motors(motor_cmd)
        
        return jsonify({
            'status': 'ok',
            'position': pos
        })
    
    return jsonify({'error': 'Simulator not initialized'}), 400

@app.route('/api/robot/sensors', methods=['GET'])
def get_sensors():
    """Get raw sensor data"""
    global simulator
    
    if simulator:
        sensors = simulator.get_sensors()
        return jsonify({
            'sensors': sensors.tolist(),
            'timestamp': time.time()
        })
    
    return jsonify({'error': 'Simulator not initialized'}), 400

@app.route('/api/robot/behavior', methods=['GET'])
def get_behavior():
    """Get current behavior library"""
    global brain
    
    if brain:
        return jsonify({
            'current': str(brain.current_behavior) if brain.current_behavior else None,
            'state': brain.behavior_state,
            'library_size': len(brain.behavior_library)
        })
    
    return jsonify({'error': 'Brain not initialized'}), 400

@app.route('/api/robot/reset', methods=['POST'])
def reset_robot():
    """Reset robot to origin"""
    global simulator, brain
    
    if simulator:
        simulator.position = np.array([0.0, 0.0])
        simulator.orientation = 0.0
    
    if brain:
        brain.motor_left.reset()
        brain.motor_right.reset()
        brain.syntropic_field = np.zeros(100)
    
    return jsonify({
        'status': 'reset',
        'message': 'Robot reset to origin'
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 SPINN ROBOT BRAIN WEB INTERFACE")
    print("=" * 60)
    print("\n📊 API Endpoints:")
    print("  POST /api/robot/init      - Initialize robot brain")
    print("  POST /api/robot/start     - Start autonomous mode")
    print("  POST /api/robot/stop      - Stop autonomous mode")
    print("  GET  /api/robot/status    - Get robot status")
    print("  POST /api/robot/manual    - Manual control")
    print("  GET  /api/robot/sensors   - Get sensor readings")
    print("  GET  /api/robot/behavior  - Get behavior info")
    print("  POST /api/robot/reset     - Reset to origin")
    print("\n🚀 Starting server on port 8000...")
    
    # Production mode - debug disabled for security
    app.run(host='0.0.0.0', port=8000, debug=False)
