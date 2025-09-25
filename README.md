
# Quantum Transduction and Lossy Channel Simulator

This repository contains a **Python simulator** built with the [QuTiP library](http://qutip.org/) for modeling quantum communication systems that involve:

- **Imperfect entangled photon sources**  
- **Lossy optical fiber channels**  
- **Noisy electro-optic (optical-to-microwave) quantum transduction**

The simulator implements **Monte Carlo trajectory simulations** and **analytical models** for fidelity estimation in hybrid quantum communication chains. Two encoding models are supported:
- **Time-bin encoding (Model 1)**
- **Single-rail Fock state encoding (Model 2)**

It provides both **numerical simulations** and **closed-form analytical predictions**, allowing performance benchmarking under realistic noise and loss conditions.  

---

## 📑 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Functionalities](#functionalities)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Contributing](#contributing)  
7. [License](#license)  

---

## Overview

Quantum networks require efficient **photon transmission** over long distances and **transduction** between optical and microwave domains to interface with superconducting qubits.  

This simulator models a **hybrid chain** where:
- An **imperfect source** generates entangled photon-qubit states with a tunable fidelity.
- Photons propagate through a **lossy optical fiber channel** (exponential attenuation with distance).
- Photons undergo **noisy electro-optic transduction** to the microwave regime, subject to thermal noise.  

The simulator evaluates the **state fidelity** between the final quantum state and the ideal entangled state under these imperfections.

---

## Features

✔️ Monte Carlo trajectory simulation of photon loss and noise  
✔️ Kraus operator formalism for loss channels  
✔️ Fidelity computation against target entangled states  
✔️ Analytical models for comparison with simulations  
✔️ Support for two encodings:
- **Time-bin encoding (Model 1)**
- **Single-rail Fock state encoding (Model 2)**  
✔️ Visualization tools:
- Fidelity vs. **transduction efficiency**  
- Fidelity vs. **fiber length**  
- Fidelity vs. **thermal photon number**

---

## Functionalities

- **Loss channel simulation**:  
  Uses Kraus operators and beam-splitter unitaries to model optical fiber loss and thermal noise.  

- **Monte Carlo trajectories**:  
  Simulates stochastic photon loss events to capture realistic experimental outcomes.  

- **Electro-optic transduction**:  
  Models imperfect conversion efficiency and coupling to a thermal environment.  

- **Analytical benchmarks**:  
  Provides closed-form fidelity expressions to compare with simulation results.  

- **Visualization**:  
  Generates high-quality LaTeX-styled plots of fidelities under varying parameters.  

---

## Installation

### Prerequisites
- Python ≥ 3.8  
- [QuTiP](http://qutip.org/)  
- NumPy  
- Matplotlib  

### Install via pip
```bash
pip install qutip numpy matplotlib
