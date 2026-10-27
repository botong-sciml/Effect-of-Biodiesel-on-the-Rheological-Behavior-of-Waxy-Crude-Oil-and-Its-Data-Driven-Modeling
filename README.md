# Effect-of-Biodiesel-on-the-Rheological-Behavior-of-Waxy-Crude-Oil-and-Its-Data-Driven-Modeling
Shear-Stress-Biased Autoencoder for Biodiesel-Treated Waxy Crude Oil Rheology
This repository presents a data-driven framework for investigating the rheological behavior of sunflower-biodiesel-treated waxy crude oil using a shear-stress-biased autoencoder.
# Project Overview
This project investigates the effects of temperature, biodiesel concentration, and shear rate on the rheological behavior of Nigerian waxy crude oil. A shear-stress-biased autoencoder is developed to capture the nonlinear and non-injective relationships among operating parameters and shear stress.
By incorporating shear stress as a conditional input to the decoder, the trained model generates multiple feasible combinations of temperature, biodiesel concentration, and shear rate corresponding to a prescribed target shear stress. This approach enables the exploration of rheological parameter landscapes and provides insights into biodiesel-induced viscosity reduction, shear-thinning behavior, and low-temperature yield-stress characteristics.
# Key Features
Shear-Stress-Biased Autoencoder: Incorporates shear stress into the decoder to learn the coupled nonlinear relationships among temperature, biodiesel concentration, and shear rate.
Conditional Parameter Generation: Uses the trained decoder independently to generate multiple feasible operating-parameter combinations associated with a prescribed target shear stress.
Rheological Parameter-Space Exploration: Constructs a shear-stress-constrained parameter landscape to investigate the combined effects of temperature, biodiesel concentration, and shear rate.
Rheological Property Analysis: Identify low apparent viscosity regions of biodiesel-treated waxy crude oil, analyze its temperature-dependent shear-thinning behavior, and characterize the nonlinear variation of apparent yield stress with biodiesel concentration.
# Results & Visualizations
The main results demonstrate the capability of the shear-stress-biased autoencoder to explore the coupled rheological parameter space and characterize the effects of biodiesel addition.
Autoencoder Architecture and Reconstruction Performance:
<img width="1131" height="610" alt="image" src="https://github.com/user-attachments/assets/f6f94b87-9977-4c13-870f-e3c24d3f70a7" />
Shear-Stress-Constrained Rheological Parameter Landscape:
<img width="1135" height="595" alt="image" src="https://github.com/user-attachments/assets/82f08fde-5dbf-4ea4-aee2-65f3ed8a8061" />
Apparent Viscosity:
<img width="1145" height="640" alt="image" src="https://github.com/user-attachments/assets/d7fa400b-16d9-42c4-92d0-9e9d89f8a281" />
Shear-Thinning Behavior and Low-Temperature Yield-Stress Behavior:
<img width="1137" height="591" alt="image" src="https://github.com/user-attachments/assets/fc3d2108-a911-4e3c-b288-65edb0aa921d" />
# Reference
If you find this project useful, please refer to the associated paper:

Title: Effect of Biodiesel on the Rheological Behavior of Waxy Crude Oil and Its Data-Driven Modeling

Link: doi: 10.1063/5.0344521
