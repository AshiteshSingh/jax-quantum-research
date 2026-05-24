import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------
# CONFIG
# ----------------------------
N_QUBITS = 36
N = 2**N_QUBITS

MARKED = "1"*N_QUBITS
ZERO   = "0"*N_QUBITS

# ----------------------------
# GROVER THEORY
# ----------------------------
theta = np.arcsin(1/np.sqrt(N))
k_opt = int(np.round((np.pi/(4*theta)) - 0.5))

def prob(k):
    return np.sin((2*k + 1)*theta)**2

# ----------------------------
# RUN
# ----------------------------
t0 = time.time()

k_vals = np.arange(0, k_opt*2)
p_vals = prob(k_vals)

p_opt = prob(k_opt)

elapsed = time.time() - t0

# ----------------------------
# METRICS
# ----------------------------
accuracy = p_opt * 100
speedup = np.sqrt(N)

print("\n===== RESULTS =====")
print(f"Qubits              : {N_QUBITS}")
print(f"Search space        : {N:,}")
print(f"Optimal iterations  : {k_opt:,}")
print(f"Success probability : {p_opt:.12f}")
print(f"Accuracy (%)        : {accuracy:.8f}%")
print(f"Theoretical speedup : ~{speedup:.2f}x")
print(f"Time taken          : {elapsed:.6f} sec")

# ----------------------------
# PLOTS
# ----------------------------
fig = plt.figure(figsize=(18,10), dpi=150)
gs = fig.add_gridspec(2,2)

# 1️⃣ Full probability wave
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(k_vals, p_vals, color="royalblue", linewidth=3)

ax1.axvline(k_opt, color="#2ca02c", linestyle="--", linewidth=2)
ax1.scatter([k_opt],[p_opt], color="#2ca02c", s=100)

ax1.set_title("Grover Probability Growth (36 Qubits)", fontsize=16)
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Success Probability")
ax1.set_ylim(0,1.02)
ax1.grid(True, linestyle="--", alpha=0.5)

# 2️⃣ Zoom near peak
ax2 = fig.add_subplot(gs[0,1])
zoom_range = range(max(0,k_opt-100), k_opt+100)

ax2.plot(zoom_range, prob(np.array(list(zoom_range))), color="royalblue")
ax2.axvline(k_opt, color="#2ca02c", linestyle="--")

ax2.set_title("Zoom Near Optimal Iteration", fontsize=16)
ax2.grid(True, linestyle="--", alpha=0.5)

# 3️⃣ Final measurement
ax3 = fig.add_subplot(gs[1,0])
labels = [f"|{ZERO}⟩", f"|{MARKED}⟩"]
values = [1-p_opt, p_opt]

bars = ax3.bar(labels, values, color=["royalblue","#2ca02c"])

for bar,val in zip(bars, values):
    ax3.text(bar.get_x()+bar.get_width()/2,
             val+0.02,
             f"{val*100:.6f}%",
             ha='center')

ax3.set_title("Final Measurement Probabilities", fontsize=16)
ax3.set_ylim(0,1.02)
ax3.grid(axis="y", linestyle="--", alpha=0.5)

# 4️⃣ Accuracy vs iterations
ax4 = fig.add_subplot(gs[1,1])
ax4.plot(k_vals, p_vals*100, color="purple", linewidth=2)

ax4.set_title("Accuracy vs Iterations (%)", fontsize=16)
ax4.set_xlabel("Iterations")
ax4.set_ylabel("Accuracy (%)")
ax4.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("grover_36q_full.png")

print("Saved -> grover_36q_full.png")