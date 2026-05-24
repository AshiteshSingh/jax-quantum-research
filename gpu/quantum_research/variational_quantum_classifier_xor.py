import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jax_qsim import Circuit
from jax_qsim.observables import PauliString, expectation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_vqc():
    print("=" * 60)
    print("  Variational Quantum Classifier (VQC): Solving the XOR Problem")
    print("=" * 60)

    # 1. Generate non-linear XOR-like dataset
    # We generate points in 2D space.
    # (+1, +1) and (-1, -1) are label 0
    # (+1, -1) and (-1, +1) are label 1
    key = jax.random.PRNGKey(24)
    num_points = 200
    
    key, subkey1, subkey2 = jax.random.split(key, 3)
    X = jax.random.uniform(subkey1, shape=(num_points, 2), minval=-1.5, maxval=1.5)
    
    # XOR logic: label is 1 if product of coordinates is negative, else 0
    Y = jnp.where(X[:, 0] * X[:, 1] < 0, 1.0, 0.0)
    
    # 2. Build VQC Circuit
    # The circuit takes 2 inputs (angle encoded) and uses 6 variational parameters
    c = Circuit(num_qubits=2)
    # Inputs: angle encoding via RX
    c.rx(0, param_index=0)
    c.rx(1, param_index=1)
    
    # Layer 1: parameterized rotations & entanglement
    c.ry(0, param_index=2)
    c.ry(1, param_index=3)
    c.cnot(0, 1)
    
    # Layer 2: parameterized rotations & entanglement
    c.ry(0, param_index=4)
    c.ry(1, param_index=5)
    c.cnot(0, 1)
    
    # Layer 3: final rotations
    c.ry(0, param_index=6)
    c.ry(1, param_index=7)
    
    obs = PauliString({1: 'Z'})
    
    # 3. Define single-point prediction
    def predict_single(params, x):
        """Execute VQC for a single input vector x = [x0, x1]."""
        # Combine inputs and trainable parameters into a single vector
        full_params = jnp.hstack([x, params])
        state = c.run(full_params)
        # Measure expectation of Z1: output in [-1.0, 1.0]
        return expectation(state, obs)
        
    # 4. Vectorize execution over a batch of data points using jax.vmap
    # This runs the simulation for all batch elements concurrently!
    predict_batch = jax.vmap(predict_single, in_axes=(None, 0))
    
    # 5. Differentiable MSE Loss
    def loss_fn(params, X_batch, Y_batch):
        predictions = predict_batch(params, X_batch)
        # Target expectations are -1.0 (for class 0) and +1.0 (for class 1)
        targets = Y_batch * 2.0 - 1.0
        return jnp.mean((predictions - targets) ** 2)

    # 6. Adam Optimizer Step
    def adam_step(params, grads, m, v, t, lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8):
        t = t + 1
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * (grads ** 2)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return params, m, v, t

    # Compile the training step
    @jax.jit
    def train_step(params, m, v, t, X_batch, Y_batch):
        loss_val, grads = jax.value_and_grad(loss_fn)(params, X_batch, Y_batch)
        params, m, v, t = adam_step(params, grads, m, v, t)
        return params, m, v, t, loss_val

    # Initialize VQC parameters (6 parameters for 3 layers of RY)
    params = jax.random.normal(subkey2, shape=(6,)) * 0.1
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    t = 0
    
    # 7. Train the Classifier
    epochs = 150
    print("\nTraining Variational Quantum Classifier...")
    for epoch in range(1, epochs + 1):
        params, m, v, t, loss_val = train_step(params, m, v, t, X, Y)
        
        if epoch == 1 or epoch % 15 == 0:
            # Calculate accuracy: predictions > 0.0 correspond to class 1
            preds = predict_batch(params, X)
            pred_classes = jnp.where(preds > 0.0, 1.0, 0.0)
            accuracy = jnp.mean(pred_classes == Y)
            print(f"Epoch {epoch:3d} | Loss: {loss_val:.6f} | Train Accuracy: {accuracy:.2%}")

    # 8. Generate Decision Boundary Plot
    print("\nGenerating decision boundary plot...")
    grid_size = 50
    grid_x = jnp.linspace(-1.8, 1.8, grid_size)
    grid_y = jnp.linspace(-1.8, 1.8, grid_size)
    xx, yy = jnp.meshgrid(grid_x, grid_y)
    
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_preds = predict_batch(params, grid_points).reshape(grid_size, grid_size)
    
    # Plotting
    plt.figure(figsize=(8, 8), facecolor='#1e1e2e')
    ax = plt.subplot(1, 1, 1)
    ax.set_facecolor('#24273a')
    
    # Contour fill prediction field
    contour = plt.contourf(xx, yy, grid_preds, levels=50, cmap='coolwarm', alpha=0.85)
    cbar = plt.colorbar(contour)
    cbar.ax.tick_params(labelsize=10, colors='#cdd6f4')
    cbar.ax.set_ylabel('Quantum Model Output (Z expectation)', color='#cdd6f4', size=11, labelpad=10)
    
    # Scatter plot data points
    scatter_0 = X[Y == 0.0]
    scatter_1 = X[Y == 1.0]
    plt.scatter(scatter_0[:, 0], scatter_0[:, 1], color='#89b4fa', label='Class 0 (XOR negative)', edgecolors='#1e1e2e', s=50, alpha=0.9)
    plt.scatter(scatter_1[:, 0], scatter_1[:, 1], color='#f38ba8', label='Class 1 (XOR positive)', edgecolors='#1e1e2e', s=50, alpha=0.9)
    
    # Label and Title
    ax.tick_params(colors='#cdd6f4', labelsize=11)
    ax.xaxis.label.set_color('#cdd6f4')
    ax.yaxis.label.set_color('#cdd6f4')
    ax.title.set_color('#cdd6f4')
    
    plt.title('Variational Quantum Classifier XOR Decision Boundary', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Feature x0', fontsize=12)
    plt.ylabel('Feature x1', fontsize=12)
    plt.legend(facecolor='#1e1e2e', edgecolor='#cba6f7', labelcolor='#cdd6f4')
    plt.grid(True, linestyle='--', color='#585b70', alpha=0.3)
    
    plot_dir = os.path.join(ROOT, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, '02_vqc_boundary.png')
    plt.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  🖼  Plot saved → {plot_path}")

if __name__ == '__main__':
    run_vqc()
