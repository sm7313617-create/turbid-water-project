# Turbidity Physics Model

Physics equations and mathematical formulations used to simulate underwater light propagation, attenuation, and scattering.

## 1. Light Attenuation (Beer-Lambert Law)

$$I(d) = I_0 \cdot e^{-\beta(\lambda) d}$$

Where:
- $I(d)$ is the light intensity at distance $d$.
- $I_0$ is the initial light intensity.
- $\beta(\lambda)$ is the wavelength-dependent attenuation coefficient.

## 2. Jaffe-McGlamery Underwater Radiance Model

$$E_{total} = E_{direct} + E_{forward\_scatter} + E_{back\_scatter}$$
