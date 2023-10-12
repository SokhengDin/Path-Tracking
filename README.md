# PathTracking
This repository is used for sharing all the PathTracking method.

# Dependencies
1. Casadi
2. NumPy
3. Matplotlib
4. BeizerPath

# PID Controller
1. We use PID for point to point by tracking all error between set point and the state feedback from the system.
2. We also use PID for trajectory tracking by also tracking all the error between each point from the trajectory and the state feedback system.

$$output = k_{p}e(t) + k_{i}\int_{t_{1}}^{t_{2}}e(t)dt + k_{d}\frac{d e(t)}{dt}$$

Which $k_{p}, k_{i}, k_{d}$ is the proportional, integral, derivative gain respectively.
Tracking error at each sampling time
$$e(t) = target - feedback$$
## Experiment
1. Point to Point
<img src="Figure/Figure_pid_p.png">
2. Trajectory Tracking
<img src="Figure/Figure_pid_t.png">

# Nonlinear Model Predictive Control (NMPC)
We use nonlinear optimization to solve the quadratic cost from sum of tracking error between trajectory tracking and input control.
```math
J = \phi_{N}(x_{N})+\sum_{k=0}^{N-1}(x_{k}-x_{k,ref})^{T}Q(x_{k}-x_{k,ref})+(u_{k}-u_{k, ref})^{T}R(u_{k}-u_{k, ref})
```
which
```math
\phi_{N} = (x_{N}-x_{N,ref})^{T}Q(x_{N}-x_{N,ref})
```
## Experiment (Trajectory Tracking with NMPC)
<img src="Figure/Figure_nmpc.png">


# Nonlinear Model Predictive Control with CGMRES

We use Continuation/Generalized Minimal Residual method for calculate optimal input

```math
H(x,u,\lambda,\mu) = L(x,u) + \phi(x_{N}) + \lambda^{T}f(x,u) + \mu^{T}C(x,u) + \nu^{T}h(x,u)
```
## Experiment without constraint optimization

<img src="Figure/nmpc_cgmres.gif">
