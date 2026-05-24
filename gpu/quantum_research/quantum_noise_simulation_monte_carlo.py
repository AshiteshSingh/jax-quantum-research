import os
import sys
import json
from datetime import datetime

# Set up the python path to import local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jax_qsim import zero_state, apply_gate, ops
from jax_qsim.noise import apply_channel, amplitude_damping_channel, phase_damping_channel, depolarizing_channel
from jax_qsim.observables import PauliString, expectation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

def simulate_trajectory(key, noise_param, channel_type, init_state, obs):
    """Simulates a single noisy quantum trajectory and computes the expectation of an observable.
    
    Args:
        key: JAX PRNGKey.
        noise_param: Damping / depolarizing rate (float).
        channel_type: 'amplitude', 'phase', or 'depolarizing'.
        init_state: State vector of shape (2,).
        obs: PauliString observable to measure.
    """
    if channel_type == 'amplitude':
        kraus = amplitude_damping_channel(noise_param)
    elif channel_type == 'phase':
        kraus = phase_damping_channel(noise_param)
    elif channel_type == 'depolarizing':
        kraus = depolarizing_channel(noise_param)
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")
        
    noisy_state, _ = apply_channel(init_state, kraus, [0], key)
    return expectation(noisy_state, obs)

# Vectorize over multiple trajectory random keys (Monte Carlo batch)
vectorized_trajectories = jax.vmap(simulate_trajectory, in_axes=(0, None, None, None, None))

# Vectorize over noise parameters to generate the full curve
vectorized_curve = jax.vmap(vectorized_trajectories, in_axes=(None, 0, None, None, None))

def run_noise_simulation():
    print("=" * 70)
    print("      Quantum Noise & Open Systems: Monte Carlo Quantum Trajectories")
    print("=" * 70)
    print("Simulating open quantum systems with JAX-accelerated quantum trajectories...\n")
    
    # 1. Define simulation configurations
    noise_vals = jnp.linspace(0.0, 1.0, 30)
    trajectory_counts = [10, 100, 1000]
    
    init_state_1 = apply_gate(zero_state(1), ops.X, [0])     # |1> state
    init_state_plus = apply_gate(zero_state(1), ops.H, [0])  # |+> state
    
    obs_z = PauliString({0: 'Z'})
    obs_x = PauliString({0: 'X'})
    
    # Base key for random number generation
    base_key = jax.random.PRNGKey(101)
    
    # Create the beautiful Catppuccin Macchiato styled plotting canvas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#1e1e2e')
    plt.subplots_adjust(wspace=0.3)
    
    # --- SIMULATION 1: Amplitude Damping ---
    print("1. Running Amplitude Damping Simulation (decay from |1> to |0>)...")
    exact_amp = 1.0 - noise_vals # Population of |1> is (1 - <Z>)/2, which is 1 - gamma
    
    ax1 = axes[0]
    ax1.set_facecolor('#24273a')
    ax1.plot(noise_vals, exact_amp, label='Exact Analytical', color='#f9e2af', linewidth=3, zorder=5)
    
    # Colors for different trajectory numbers
    traj_colors = {10: '#f38ba8', 100: '#89b4fa', 1000: '#a6e3a1'}
    
    for num_trajs in trajectory_counts:
        # Generate random keys for each trajectory
        subkeys = jax.random.split(base_key, num_trajs)
        # Compute expectations of Z: shape (num_noise_vals, num_trajs)
        z_expectations = vectorized_curve(subkeys, noise_vals, 'amplitude', init_state_1, obs_z)
        # Convert to population of |1>: (1 - Z_expect) / 2
        populations_1 = (1.0 - z_expectations) / 2.0
        # Average over all trajectories: shape (num_noise_vals,)
        avg_pop = jnp.mean(populations_1, axis=1)
        
        ax1.scatter(noise_vals, avg_pop, label=f'{num_trajs} Trajectories', color=traj_colors[num_trajs], alpha=0.8, s=40)
        
    ax1.set_title('Amplitude Damping (|1> Relaxation)', fontsize=13, fontweight='bold', color='#cdd6f4', pad=12)
    ax1.set_xlabel('Damping Rate ($\\gamma$)', fontsize=11, color='#cdd6f4')
    ax1.set_ylabel('Population of State $|1\\rangle$', fontsize=11, color='#cdd6f4')
    ax1.grid(True, linestyle='--', color='#585b70', alpha=0.4)
    ax1.tick_params(colors='#cdd6f4', labelsize=10)
    ax1.legend(facecolor='#1e1e2e', edgecolor='#cba6f7', labelcolor='#cdd6f4', loc='upper right')
    
    # --- SIMULATION 2: Phase Damping (Pure Dephasing) ---
    print("2. Running Phase Damping Simulation (dephasing of |+> state)...")
    exact_phase = jnp.sqrt(1.0 - noise_vals) # <X> exact is sqrt(1 - gamma)
    
    ax2 = axes[1]
    ax2.set_facecolor('#24273a')
    ax2.plot(noise_vals, exact_phase, label='Exact Analytical', color='#f9e2af', linewidth=3, zorder=5)
    
    for num_trajs in trajectory_counts:
        subkeys = jax.random.split(base_key, num_trajs)
        # Compute expectations of X: shape (num_noise_vals, num_trajs)
        x_expectations = vectorized_curve(subkeys, noise_vals, 'phase', init_state_plus, obs_x)
        # Average over all trajectories: shape (num_noise_vals,)
        avg_x = jnp.mean(x_expectations, axis=1)
        
        ax2.scatter(noise_vals, avg_x, label=f'{num_trajs} Trajectories', color=traj_colors[num_trajs], alpha=0.8, s=40)
        
    ax2.set_title('Phase Damping (Dephasing of $|+\\rangle$)', fontsize=13, fontweight='bold', color='#cdd6f4', pad=12)
    ax2.set_xlabel('Dephasing Rate ($\\gamma$)', fontsize=11, color='#cdd6f4')
    ax2.set_ylabel('Expectation Value $\\langle X \\rangle$', fontsize=11, color='#cdd6f4')
    ax2.grid(True, linestyle='--', color='#585b70', alpha=0.4)
    ax2.tick_params(colors='#cdd6f4', labelsize=10)
    ax2.legend(facecolor='#1e1e2e', edgecolor='#cba6f7', labelcolor='#cdd6f4')
    
    # --- SIMULATION 3: Depolarizing Channel ---
    print("3. Running Depolarizing Noise Simulation (decay of |+> state)...")
    exact_depol = 1.0 - (4.0 / 3.0) * noise_vals # <X> exact is 1 - 4/3 * p
    
    ax3 = axes[2]
    ax3.set_facecolor('#24273a')
    ax3.plot(noise_vals, exact_depol, label='Exact Analytical', color='#f9e2af', linewidth=3, zorder=5)
    
    for num_trajs in trajectory_counts:
        subkeys = jax.random.split(base_key, num_trajs)
        # Compute expectations of X: shape (num_noise_vals, num_trajs)
        x_expectations = vectorized_curve(subkeys, noise_vals, 'depolarizing', init_state_plus, obs_x)
        # Average over all trajectories: shape (num_noise_vals,)
        avg_x = jnp.mean(x_expectations, axis=1)
        
        ax3.scatter(noise_vals, avg_x, label=f'{num_trajs} Trajectories', color=traj_colors[num_trajs], alpha=0.8, s=40)
        
    ax3.set_title('Depolarizing Noise on $|+\\rangle$', fontsize=13, fontweight='bold', color='#cdd6f4', pad=12)
    ax3.set_xlabel('Depolarization Probability ($p$)', fontsize=11, color='#cdd6f4')
    ax3.set_ylabel('Expectation Value $\\langle X \\rangle$', fontsize=11, color='#cdd6f4')
    ax3.grid(True, linestyle='--', color='#585b70', alpha=0.4)
    ax3.tick_params(colors='#cdd6f4', labelsize=10)
    ax3.legend(facecolor='#1e1e2e', edgecolor='#cba6f7', labelcolor='#cdd6f4')
    
    # Main Figure Title
    fig.suptitle('Monte Carlo Quantum Trajectories vs. Exact Analytical Solutions', 
                 fontsize=16, fontweight='bold', color='#cdd6f4', y=0.98)
    
    # Save the output plot
    plot_dir = os.path.join(ROOT, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'noise_simulation_monte_carlo_{TS}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save JSON results
    results_dir = os.path.join(ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f'noise_simulation_monte_carlo_{TS}.json')
    json.dump({"experiment": "noise_simulation", "noise_vals": noise_vals.tolist(), "trajectory_counts": trajectory_counts},
              open(json_path, "w"), indent=2)
    
    print("\n" + "=" * 70)
    print(f"Success! Noise simulation complete.")
    print(f"Stunning visualization saved to '{plot_path}'")
    print(f"Results JSON saved to '{json_path}'")
    print("=" * 70)

if __name__ == '__main__':
    run_noise_simulation()
