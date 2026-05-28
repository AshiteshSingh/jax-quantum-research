"""
Experiment 2: Variational Quantum Classifier (VQC)
Trains a 2-qubit VQC to solve the non-linear XOR classification problem.
Uses jax.vmap to run vectorized predictions over data batches, and JIT-compiled Adam.
"""

import os
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys

# Ensure jax_qsim is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_qsim.circuit import Circuit
import jax_qsim.statevector as sv

# ==============================================================================
# Setup Experiment Parameters
# ==============================================================================
NUM_QUBITS = 2
VAR_PARAMS = 6 # 6 variational parameters
TOTAL_PARAMS = NUM_QUBITS + VAR_PARAMS # 2 inputs (angle encoded) + 6 variational

# Generate XOR Dataset
key = jax.random.PRNGKey(24)
key, k1, k2 = jax.random.split(key, 3)
X_data = jax.random.uniform(k1, (200, 2), minval=-1.5, maxval=1.5)
Y_data = jnp.where(X_data[:, 0] * X_data[:, 1] < 0, 1.0, 0.0)

# Setup paths
os.makedirs("results", exist_ok=True)

# ==============================================================================
# Differentiable Variational Classifier
# ==============================================================================
def circuit_single(full_params):
    """Executes a single circuit and measures <Z_1>."""
    c = Circuit(NUM_QUBITS)
    # 1. Angle encoding of 2D input coordinates
    c.rx(0, 0)
    c.rx(1, 1)
    
    # 2. Variational Ansatz
    c.ry(0, 2)
    c.ry(1, 3)
    c.cnot(0, 1)
    c.ry(0, 4)
    c.ry(1, 5)
    c.cnot(0, 1)
    c.ry(0, 6)
    c.ry(1, 7)
    
    state = c.run(full_params, 'statevector')
    
    # Measure <Z_1> expectation value
    # Expectation of Z on qubit 1: <Z_1> = |a|^2 - |b|^2 marginalized
    probs = jnp.abs(state) ** 2
    prob_q1 = jnp.sum(probs, axis=0) # Marginalize over qubit 0
    return jnp.real(prob_q1[0] - prob_q1[1])

def predict_single(params, x):
    """Concatenates input x and variational parameters and runs the circuit."""
    full_params = jnp.hstack([x, params])
    return circuit_single(full_params)

# Vectorized prediction over a batch of inputs using jax.vmap
predict_batch = jax.vmap(predict_single, in_axes=(None, 0))

def loss_fn(params, X_batch, Y_batch):
    """Mean Squared Error loss between predictions and target labels in [-1, 1]."""
    preds = predict_batch(params, X_batch)
    targets = Y_batch * 2.0 - 1.0 # Map {0, 1} -> {-1, 1}
    return jnp.mean((preds - targets) ** 2)

# ==============================================================================
# JIT-compiled Optimization Step with Adam
# ==============================================================================
def adam_update(p, g, m, v, t, lr=0.03, b1=0.9, b2=0.999, eps=1e-8):
    t = t + 1
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * (g ** 2)
    mh = m / (1.0 - b1 ** t)
    vh = v / (1.0 - b2 ** t)
    return p - lr * mh / (jnp.sqrt(vh) + eps), m, v, t

@jax.jit
def step(params, m, v, t, X_batch, Y_batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, X_batch, Y_batch)
    params, m, v, t = adam_update(params, grads, m, v, t)
    return params, m, v, t, loss

# ==============================================================================
# Execute Training Loop
# ==============================================================================
def run_experiment():
    print("=" * 80)
    print(" EXPERIMENT 2: Variational Quantum Classifier (XOR) ".center(80, "="))
    print("=" * 80)
    
    # Initialize variational parameters
    params = jax.random.normal(k2, (VAR_PARAMS,)) * 0.1
    m = jnp.zeros(VAR_PARAMS)
    v = jnp.zeros(VAR_PARAMS)
    t = 0
    
    epochs = 150
    loss_history = []
    
    print(f"{'Epoch':^10} | {'MSE Loss':^18} | {'Classification Accuracy':^25}")
    print("-" * 65)
    
    t0 = time.time()
    for ep in range(1, epochs + 1):
        params, m, v, t, current_loss = step(params, m, v, t, X_data, Y_data)
        loss_history.append(float(current_loss))
        
        if ep == 1 or ep % 15 == 0:
            # Compute current training accuracy
            preds = predict_batch(params, X_data)
            pred_classes = jnp.where(preds > 0.0, 1.0, 0.0)
            accuracy = jnp.mean(pred_classes == Y_data)
            print(f"{ep:^10d} | {current_loss:^18.8f} | {accuracy:^25.2%}")
            
    total_time = time.time() - t0
    
    # Final Accuracy
    final_preds = predict_batch(params, X_data)
    final_classes = jnp.where(final_preds > 0.0, 1.0, 0.0)
    final_accuracy = jnp.mean(final_classes == Y_data)
    
    print("-" * 65)
    print(f"Final Classification Accuracy: {final_accuracy:.2%}")
    print(f"Total Compilation & Training Time: {total_time:.3f} seconds")
    print("=" * 80)
    
    # ==============================================================================
    # Plotting & Visualization
    # ==============================================================================
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0d1117')
    
    # Left Plot: Decision Boundary
    ax = axes[0]
    ax.set_facecolor('#161b22')
    
    # Generate meshgrid
    grid_size = 40
    gx = jnp.linspace(-1.8, 1.8, grid_size)
    gy = jnp.linspace(-1.8, 1.8, grid_size)
    xx, yy = jnp.meshgrid(gx, gy)
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Predict over the entire grid
    grid_preds = predict_batch(params, grid_points).reshape(grid_size, grid_size)
    
    # Plot decision boundary contour
    contour = ax.contourf(xx, yy, grid_preds, levels=50, cmap="coolwarm", alpha=0.85)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.tick_params(colors='#e6edf3')
    cbar.outline.set_edgecolor('#30363d')
    
    # Scatter plot data points
    class_0 = X_data[Y_data == 0.0]
    class_1 = X_data[Y_data == 1.0]
    ax.scatter(class_0[:, 0], class_0[:, 1], color='#58a6ff', s=25, edgecolor='#30363d', label="Class 0 (x0 * x1 >= 0)")
    ax.scatter(class_1[:, 0], class_1[:, 1], color='#f78166', s=25, edgecolor='#30363d', label="Class 1 (x0 * x1 < 0)")
    
    ax.set_title("🎯  VQC Decision Boundary (XOR)", fontsize=13, color='#e6edf3', fontweight='bold', pad=12)
    ax.set_xlabel("Feature x₀", fontsize=11, color='#8b949e')
    ax.set_ylabel("Feature x₁", fontsize=11, color='#8b949e')
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
    ax.grid(True, linestyle='--', color='#21262d', alpha=0.4)
    ax.tick_params(colors='#e6edf3')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
        
    # Right Plot: Training Loss
    ax2 = axes[1]
    ax2.set_facecolor('#161b22')
    ax2.plot(loss_history, color='#ffa657', lw=2.5)
    ax2.set_title("📉  VQC Training Loss Curve", fontsize=13, color='#e6edf3', fontweight='bold', pad=12)
    ax2.set_xlabel("Epoch / Training Step", fontsize=11, color='#8b949e')
    ax2.set_ylabel("Mean Squared Error (MSE) Loss", fontsize=11, color='#8b949e')
    ax2.grid(True, linestyle='--', color='#21262d', alpha=0.6)
    ax2.tick_params(colors='#e6edf3')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#30363d')
        
    # Super Title
    fig.suptitle("Variational Quantum Classifier (VQC) XOR Boundary - JAX GPU Engine", 
                 color='#e6edf3', fontsize=15, fontweight='bold', y=0.98)
    
    plot_path = os.path.join("results", "02_vqc_xor.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='#0d1117')
    plt.close()
    
    print(f"Plot saved successfully to: {plot_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()
