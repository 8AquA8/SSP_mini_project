import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import pennylane as qml
import torch
import torch.optim as optim

def cir5_PQC(device: str, nq: int, n_layers: int, shots: int):
    # Define a device
    dev = qml.device(device, wires=nq, shots=shots)
    
    # Define the quantum function
    @qml.qnode(dev, interface='torch')
    def circuit(weights, hamiltonian: str):
        
        for i in range(n_layers):
            # Apply RX and RZ gates to the 1st layer
            for q in range(nq):
                qml.RX(weights[i, q, 0], wires=q)
                qml.RZ(weights[i, q, 1], wires=q)
                
            # CNOT gates layers
            for layer in range(nq):
                for j in range(nq):
                    if j != layer:
                        qml.CNOT(wires=[layer, j])
                        
        for q in range(nq):
            qml.RX(weights[n_layers, q, 0], wires=q)
            qml.RZ(weights[n_layers, q, 1], wires=q)
                
        for i, char in enumerate(hamiltonian):
            if char == 'X':
                qml.Hadamard(wires=i)
            elif char == 'Y':
                qml.RX(-np.pi/2, wires=i)
                
        return qml.probs()
    
    return circuit

def vqe_nrg(nq, qc, weights, j=1 ,h=1):
    
    interaction = torch.tensor(0.0)
    field = torch.tensor(0.0)
    
    zz_observable = 'Z' * nq
    x_observables = ['I' * i + 'X' + 'I' * (nq - i - 1) for i in range(nq)]
    
    probs = {zz_observable: qc(weights, zz_observable)}

    for obs in x_observables:
        probs[obs] = qc(weights, obs) 

    
    for i in range(2**nq):
        state = format(i, f'0{nq}b')

        # Interaction term calculation
        for k in range(nq-1):
            if state[k] == state[k+1]:
                interaction = interaction + 1 * probs[zz_observable][i]
            else:
                interaction = interaction - 1 * probs[zz_observable][i]
        
        # Field term calculation
        for k in range(nq):
            if state[k] == '0':
                field = field + 1 * probs[x_observables[k]][i]
            else:
                field = field - 1 * probs[x_observables[k]][i]

    E = (- j * interaction - h * field)
    E.requires_grad_(True)
    return E

def calculate_basis(n_sites):
    basis = []
    for i in range(2**n_sites):
        basis.append([int(x) for x in np.binary_repr(i, width=n_sites)])
    return basis


def vqe_magnetization(nq, qc, weights):
    zz_observable = 'z' * nq
    probs = qc(weights, zz_observable)
    
    n_sites = np.log2(len(probs))
    n_sites = int(n_sites)
    basis = calculate_basis(n_sites)
    M = 0.
    for i, bstate in enumerate(basis):
        bstate_M = 0.
        for spin in bstate:
            bstate_M += (probs[i] * (1 if spin else -1)) / len(bstate)
        M += abs(bstate_M)
    return M

def vqe_ground_state_properties(nq, device, n_layers, shots, weights, j=1, h=-1):
    qc = cir5_PQC(device, nq, n_layers, shots)
    energy = vqe_nrg(nq, qc, weights, j, h).item()
    
    basis = calculate_basis(nq)
    state_probs = qc(weights, 'I' * nq).detach().numpy()
    
    magnetization = vqe_magnetization(nq, qc, weights)
    
    return energy, magnetization, state_probs

def train(device, shots, site_numbers, n_layers, h_values, j):
    magnetizations = {n_sites:[] for n_sites in site_numbers}
    energies = {n_sites: [] for n_sites in site_numbers}
    trained_weights = {n_sites: [] for n_sites in site_numbers}
    
    for n_sites in site_numbers:
        weights = torch.zeros((n_layers + 1, n_sites, 2), requires_grad=True)
        qc = cir5_PQC(device, n_sites, n_layers, shots)
        best_h_weights = []
        best_h_magnetizations = []
        best_h_energies = []
        
        for h in h_values:
            optimizer = optim.Adam([weights], lr=0.01)
            
            h_energies = []
            h_weights = []
            
            for step in range(500):
                optimizer.zero_grad()
                h_energy = vqe_nrg(n_sites, qc, weights, j, h)
                h_energy.backward()
                optimizer.step()
                
                h_energies.append(h_energy.item())
                h_weights.append(weights.clone().detach())
                # h_weights.append(weights)
                if step % 10 == 0:
                    print(f"Step {step+1} of {n_sites} sites, {h} magnetic field PQC Energy: {h_energy.item()}")
            
            min_index = h_energies.index(min(h_energies))
            best_h_weights.append(h_weights[min_index])
            
            best_h_energy, best_h_magnetization, _ = vqe_ground_state_properties(n_sites, device, n_layers, shots, h_weights[min_index], j, h)
            
            best_h_energies.append(best_h_energy)
            best_h_magnetizations.append(best_h_magnetization)
            
            # Save the best weights for each (n_sites, h)
            save_path = f"best_weights_n{n_sites}_h{h:.4f}_cir5.pt"
            torch.save(h_weights[min_index], save_path)
            print(f"Best weights saved to {save_path}")
            
            
        trained_weights[n_sites].append(best_h_weights)
        energies[n_sites].append(best_h_energies)
        magnetizations[n_sites].append(best_h_magnetizations)
        
        
    return magnetizations, energies, trained_weights 



##########main##########

torch.manual_seed(42)

device = 'default.qubit'
shots = 30000
n_layers = 2
j = 1.0

site_numbers=[3, 5, 7]
h_values = np.logspace(-2, 2, 20)

magnetizations, energies, trained_weights = train(device, shots, site_numbers, n_layers, h_values, j)

plt.figure(figsize=(10, 6))
for n_sites in site_numbers:
    plt.plot(h_values, energies[n_sites][0], 'o-', label=f'N = {n_sites}')
plt.xscale('log')
plt.xlabel('h')
plt.ylabel('Energy')
plt.title('Ground State Energy vs. Transverse Field Strength')
plt.legend()
plt.grid(True)
plt.savefig('ground_state_energy_cir5.png')  # Save the energy plot
plt.close()  # Close the plot

plt.figure(figsize=(10, 6))
for n_sites in site_numbers:
    plt.plot(h_values, magnetizations[n_sites][0], 'o-', label=f'N = {n_sites}')
plt.xscale('log')
plt.xlabel('h')
plt.ylabel('Magnetization')
plt.title('Ground State Magnetization vs. Transverse Field Strength')
plt.legend()
plt.grid(True)
plt.savefig('ground_state_magnetization_cir5.png')  # Save the magnetization plot
plt.close()  # Close the plot