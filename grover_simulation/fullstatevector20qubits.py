import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------
# CONFIG
# ----------------------------
N_QUBITS = 20
N = 2**N_QUBITS
TARGET = N - 1  # |111...1>

# ----------------------------
# INITIAL STATE (|s>)
# ----------------------------
state = np.ones(N, dtype=np.complex64) / np.sqrt(N)

# ----------------------------
# ORACLE
# ----------------------------
def oracle(s):
    s[TARGET] *= -1
    return s

# ----------------------------
# DIFFUSION
# ----------------------------
def diffusion(s):
    mean = np.mean(s)
    return 2 * mean - s

# ----------------------------
# GROVER STEP
# ----------------------------
def grover_step(s):
    s = oracle(s)
    s = diffusion(s)
    return s

# ----------------------------
# OPTIMAL ITERATIONS
# ----------------------------
theta = np.arcsin(1/np.sqrt(N))
k_opt = int(np.round((np.pi/(4*theta)) - 0.5))

print("Optimal iterations:", k_opt)

# ----------------------------
# RUN
# ----------------------------
t0 = time.time()

probs = []

for i in range(k_opt):
    state = grover_step(state)
    p = np.abs(state[TARGET])**2
    probs.append(p)

elapsed = time.time() - t0

# ----------------------------
# FINAL RESULT
# ----------------------------
p_final = probs[-1]

print("\n===== RESULTS =====")
print(f"Qubits              : {N_QUBITS}")
print(f"Search space        : {N:,}")
print(f"Iterations run      : {k_opt}")
print(f"Final probability   : {p_final:.8f}")
print(f"Accuracy (%)        : {p_final*100:.6f}%")
print(f"Time taken          : {elapsed:.4f} sec")

# ----------------------------
# PLOT
# ----------------------------
fig = plt.figure(figsize=(14,6))

# curve
plt.subplot(1,2,1)
plt.plot(probs, color="blue")
plt.axvline(k_opt, color="green", linestyle="--")
plt.title("Grover Probability Growth")
plt.xlabel("Iteration")
plt.ylabel("P(target)")
plt.grid()

# final bar
plt.subplot(1,2,2)
labels = ["|0...0⟩", "|1...1⟩"]
values = [1-p_final, p_final]

bars = plt.bar(labels, values, color=["blue","green"])

for bar,val in zip(bars, values):
    plt.text(bar.get_x()+bar.get_width()/2,
             val+0.01,
             f"{val*100:.3f}%",
             ha='center')

plt.title("Final Measurement")
plt.ylim(0,1.05)

plt.tight_layout()
plt.savefig("grover_20q_bruteforce.png")

print("Saved -> grover_20q_bruteforce.png")