# Time-Dependent Geomagnetic Inversion: Physics-based vs. PINN Approach

This repository contains two complementary approaches to time-dependent geomagnetic field inversion using a truncated spherical harmonic expansion:

1. **Physics-Based Inversion** using Tikhonov regularization and B-spline time basis.
2. **Physics-Informed Neural Network (PINN)** that learns the temporal variation of spherical harmonic coefficients by minimizing a physics-based loss.

---

## 1. Problem Overview

The goal is to recover time-varying spherical harmonic coefficients $( g_{l,m}(t), h_{l,m}(t) )$ of the geomagnetic field from synthetic vector magnetic field observations $( \mathbf{B}(\mathbf{r}, t) )$ at Earth's surface. The forward operator is based on the gradient of a potential field expressed in Schmidt semi-normalized spherical harmonics.



## 2. Traditional Physics-based Approach

**File**: `geomagnetic-inversion-tikhonov.ipynb`

###  Method Summary:
- Uses spherical harmonic expansion up to `L_max = 3`.
- Generates synthetic vector magnetic field data from a known time-dependent model.
- Uses B-splines to represent temporal variation of each coefficient.
- Builds a large Jacobian matrix $( G(t, \theta, \phi) )$.
- Adds second-order temporal smoothness regularization.
- Solves the Tikhonov-regularized least squares system in one step.

###  Key Steps:
1. Define the true geomagnetic model with synthetic time variation.
2. Generate synthetic noisy observations at random lat-lon points over time.
3. Construct the B-spline basis and full spatio-temporal Jacobian matrix.
4. Add spatial and temporal damping via second-order finite differences.
5. Solve the regularized linear system to estimate model coefficients.
6. Plot recovered vs. true coefficients over time.

### Output:
Plots comparing the true time-dependent coefficients with the recovered ones for each $( g_{l,m}(t) )$ and $( h_{l,m}(t) )$.

---

## 3. Physics-Informed Neural Network (PINN)

**File**: 'geomagnetic-inversion-pinn-fourierfeatures.ipynb'

### Method Summary:
- Uses a neural network to predict all spherical harmonic coefficients at any given time \( t \).
- Minimizes a composite loss:
  - Data misfit loss (between observed and predicted magnetic field components).
  - Smoothness loss via temporal derivatives of the predicted coefficients.
- Uses automatic differentiation to compute gradients of coefficients w.r.t. time.
- Suitable for sparse or irregularly sampled data and allows for continuous-time queries.

###  Key Steps:
1. Define the same synthetic true model and generate observations.
2. Build a neural network that takes time \( t \) as input and outputs the full coefficient vector.
3. Define the physics-based loss (data + regularization).
4. Train using gradient-based optimization (Here, Adam).
5. Plot the predicted coefficients against the true ones.

###  Output:
Plots comparing the predicted coefficients from the PINN and the true ones for each $( g_{l,m}(t) )$ and $( h_{l,m}(t) )$.

---

## 4. Comparison

| Feature                         | Traditional Inversion      | PINN Approach               |
|-------------------------------|----------------------------|-----------------------------|
| Time Representation           | B-splines (discrete basis) | Neural network (continuous) |
| Regularization                | Explicit finite-difference | Implicit via loss terms     |
| Flexibility                   | Moderate                   | High (can handle gaps, etc.)|
| Interpretability              | High (linear model)        | Medium                      |
| Computational Cost            | One-time solve             | Iterative training          |

---

## 5. Getting Started

### Requirements

Install dependencies via `pip`:

```bash
pip install numpy scipy matplotlib torch
```

## 6. Notes

-Both methods assume access to the same forward operator and synthetic data for fair comparison. The synthetic data is very simple and linear. This is just to check if the PINN approach works correctly and gives similar results
to the traditional Least Squares approach.

-The traditional method is better suited when problem structure is well understood.

-The PINN is advantageous for more flexible, data-driven setups (e.g., irregular time sampling).

## 7. References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). [Physics-Informed Neural Networks](https://doi.org/10.1016/j.jcp.2018.10.045). *Journal of Computational Physics*, **378**, 686–707.

-  Out, F., Schanner, M., van Grinsven, L., Korte M.,&  de Groot, L.V. (2025). [Pymaginverse: A python package for global geomagnetic field modeling](https://doi.org/10.1016/j.acags.2025.100222). *Applied Computing and Geosciences*, **25**, 100222.

- Korte, M., & Constable, C. G. (2003). [Continuous global geomagnetic field models for the past 3000 years](https://doi.org/ 10.1016/j.pepi.2003.07.013). *Physics of the Earth and Planetary Interiors*, **140** (1–3), 73-89,.

- Nilsson, A., Holme, R., Korte, M., Suttie, N., & Hill, M. (2014). [Holocene geocentric axial dipole moment variations inferred from geomagnetic field models](https://doi.org/10.1016/j.epsl.2014.01.001). *Earth and Planetary Science Letters*, **387**, 9–17.

## License

This project is licensed under the [MIT License](LICENSE).


