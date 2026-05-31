import json
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
import os
import time

# ==========================================
# 1. Configuration & Constants
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "2dmatpedia_full.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "artifacts", "generated")
MAX_ATOMS = 40
N_LAYERS = 3
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.05
SEED = 42

# Atomic numbers map
ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
}


# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
def get_coulomb_matrix(sites, max_atoms=40):
    n_atoms = len(sites)
    coords = []
    charges = []

    for site in sites:
        coords.append(site["xyz"])
        elem = site["species"][0]["element"]
        charges.append(ATOMIC_NUMBERS.get(elem, 0))

    coords = np.array(coords)
    charges = np.array(charges)
    mat = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                mat[i, j] = 0.5 * (charges[i] ** 2.4)
            else:
                dist = np.linalg.norm(coords[i] - coords[j])
                mat[i, j] = (charges[i] * charges[j]) / dist

    padded_mat = np.zeros((max_atoms, max_atoms))
    padded_mat[:n_atoms, :n_atoms] = mat
    return padded_mat


def get_eigenspectrum(coulomb_matrix):
    eigvals = np.linalg.eigvalsh(coulomb_matrix)
    return np.sort(eigvals)[::-1]


def load_and_process_data():
    print(f"Loading data from {DATA_PATH}...")
    X, y = [], []

    count = 0
    # The file is db.json, likely a standard JSON file (list of dicts) based on user correction
    try:
        with open(DATA_PATH, "rt", encoding="utf-8") as f:
            # Check if it's one huge JSON object or JSON lines
            first_char = f.read(1)
            f.seek(0)

            if first_char == "[":
                data_list = json.load(f)
                iterator = data_list
            else:
                # Assume JSON Lines if not a list
                iterator = f

            for item in iterator:
                if isinstance(item, str):  # iterating file object yields strings
                    try:
                        data = json.loads(item)
                    except:
                        continue
                else:
                    data = item

                if "structure" not in data or "exfoliation_energy_per_atom" not in data:
                    continue

                val = data["exfoliation_energy_per_atom"]
                if val is None:
                    continue

                sites = data["structure"]["sites"]
                if len(sites) > MAX_ATOMS:
                    continue

                cm = get_coulomb_matrix(sites, MAX_ATOMS)
                feat = get_eigenspectrum(cm)

                X.append(feat)
                y.append(val)

                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count}...")

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    return np.array(X), np.array(y)


# ==========================================
# 3. QML Model Definition
# ==========================================
def create_qml_model(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs, weights):
        # inputs are normalized automatically by AmplitudeEmbedding?
        # Actually we should normalize manually to be safe, or pass normalize=True
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(n_qubits), pad_with=0.0, normalize=True
        )
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def predict_batch(circuit, inputs, weights, bias_weights):
    # Inputs: (Batch, Features)
    # Use pnp.stack to preserve autograd boxes
    res = [pnp.stack(circuit(x, weights)) for x in inputs]
    expvals = pnp.stack(res)

    # Linear Layer: dot(expvals, w) + b
    w_linear = bias_weights[:-1]
    b = bias_weights[-1]
    return pnp.dot(expvals, w_linear) + b


# ==========================================
# 4. Main Execution
# ==========================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Data
    X_raw, y_raw = load_and_process_data()
    if X_raw is None or len(X_raw) == 0:
        print("Data load failed or empty.")
        return

    # Subsample for demo speed (optional)
    if len(X_raw) > 500:
        print("Subsampling to 500 points for demonstration speed...")
        indices = np.random.choice(len(X_raw), 500, replace=False)
        X_raw = X_raw[indices]
        y_raw = y_raw[indices]

    print(f"Dataset Shape: {X_raw.shape}")

    # 2. Setup Dimensions
    n_features = X_raw.shape[1]
    n_qubits = int(np.ceil(np.log2(n_features)))
    print(f"Features: {n_features} -> Mapped to {n_qubits} Qubits")

    # 3. Train/Test Split
    np.random.seed(SEED)
    indices = np.arange(len(X_raw))
    np.random.shuffle(indices)
    split = int(0.8 * len(X_raw))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y_raw[train_idx], y_raw[test_idx]

    # PennyLane constants
    X_train_p = pnp.array(X_train, requires_grad=False)
    y_train_p = pnp.array(y_train, requires_grad=False)
    X_test_p = pnp.array(X_test, requires_grad=False)

    # 4. Initialize Weights
    weights = pnp.random.uniform(
        0, 2 * np.pi, (N_LAYERS, n_qubits, 3), requires_grad=True
    )
    bias_weights = pnp.random.normal(0, 0.1, (n_qubits + 1), requires_grad=True)

    circuit = create_qml_model(n_qubits, N_LAYERS)
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)

    # 5. Training Loop
    print("\nStarting Training...")
    train_losses = []
    val_losses = []

    start_t = time.time()
    n_batches = len(X_train) // BATCH_SIZE

    for epoch in range(EPOCHS):
        # Shuffle
        perm = np.random.permutation(len(X_train_p))
        X_train_p = X_train_p[perm]
        y_train_p = y_train_p[perm]

        epoch_loss = 0.0
        for i in range(n_batches):
            batch_X = X_train_p[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            batch_y = y_train_p[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

            def cost(w, b):
                preds = predict_batch(circuit, batch_X, w, b)
                return pnp.mean((preds - batch_y) ** 2)

            (weights, bias_weights), loss_val = opt.step_and_cost(
                cost, weights, bias_weights
            )
            epoch_loss += loss_val

        epoch_loss /= n_batches
        train_losses.append(epoch_loss)

        # Validation
        val_preds = predict_batch(circuit, X_test_p, weights, bias_weights)
        val_loss = pnp.mean((val_preds - y_test) ** 2)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

    print(f"Training finished in {time.time() - start_t:.2f}s")

    # 7. Save Model (Do this FIRST so we don't lose it if plotting fails)
    np.save(os.path.join(OUTPUT_DIR, "weights.npy"), weights)
    np.save(os.path.join(OUTPUT_DIR, "bias.npy"), bias_weights)
    print("Model saved.")

    # 6. Evaluation & Visuals
    print("\ngenerating visualizations...")

    def clean_array(arr):
        # Helper to strip autograd/pennylane wrappers
        if hasattr(arr, "_value"):
            return np.array(arr._value)
        if hasattr(arr, "numpy"):
            return arr.numpy()
        return np.array(arr)

    # Detach from autograd for plotting
    train_losses = clean_array(train_losses)
    val_losses = clean_array(val_losses)

    # Re-run prediction without grad tracking for safety
    final_preds_p = predict_batch(circuit, X_test_p, weights, bias_weights)
    final_preds = clean_array(final_preds_p)

    y_test_np = clean_array(y_test)
    residuals = final_preds - y_test_np

    # A. Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("QML Training Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()

    # B. Parity Plot (Actual vs Predicted)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_np, final_preds, alpha=0.7, color="blue")

    min_val = min(np.min(y_test_np), np.min(final_preds))
    max_val = max(np.max(y_test_np), np.max(final_preds))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")

    plt.xlabel("Actual Energy (eV/atom)")
    plt.ylabel("Predicted Energy (eV/atom)")
    plt.title("Actual vs Predicted Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "parity_plot.png"))
    plt.close()

    # C. Error Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=20, color="purple", alpha=0.7, edgecolor="black")
    plt.xlabel("Prediction Error (eV/atom)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "error_histogram.png"))
    plt.close()

    # D. Residual Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(final_preds, residuals, alpha=0.6, color="green")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Energy (eV/atom)")
    plt.ylabel("Residuals (Pred - Actual)")
    plt.title("Residuals vs Predictions")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "residual_plot.png"))
    plt.close()

    print(f"Visuals saved to {OUTPUT_DIR}")

    # Save Model
    np.save(os.path.join(OUTPUT_DIR, "weights.npy"), weights)
    np.save(os.path.join(OUTPUT_DIR, "bias.npy"), bias_weights)
    print("Model saved.")


if __name__ == "__main__":
    main()
