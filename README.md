# Inverse Design of 3D Nanophotonic Devices with Structural Integrity Using Auxiliary Thermal Solvers
## Abstract
3D additive manufacturing enables the fabrication of nanophotonic structures with subwavelength features that control
light across macroscopic scales. Gradient-based optimization offers an efficient approach to design these complex and
non-intuitive structures. However, expanding this methodology from 2D to 3D introduces complexities, such as the
need for structural integrity and connectivity. This work introduces a multi-objective optimization method to address
these challenges in 3D nanophotonic designs. Our method combines electromagnetic simulations with an auxiliary
heat-diffusion solver to ensure continuous material and void connectivity. By modeling material regions as heat sources
and boundaries as heat sinks, we strive to minimize the total temperature that penalizes disconnected regions within
the structure. Alongside the optical response, this temperature metric becomes part of our objective function. We
demonstrate the utility of our algorithm by designing two 3D nanophotonic devices. The first is a focusing element.
The second is a waveguide junction, which connects two incoming waveguides for two different wavelengths into two
outgoing waveguides, which are rotated by 90° to the incoming waveguides. Our approach offers a design pipeline
that generates digital blueprints for fabricable nanophotonic materials, paving the way for practical 3D nanoprinting
applications.

## Problem formulation
This code provides the full opitimization for the paper. We are interested in enforcing physical constriants to our optimization problem to enforce structural integrity.
This is done by using two auxiliary heat dissipation simulations, where we solve Poisson's Equation to calculate the heat distribution $u(x, y, z)$ using a finite element scheme.

If we use no auxiliary heat solver we end up with designs which are free floating or contain free floating parts.
For example, a focusing device, where the middle part is not connected anywhere.
![](https://github.com/OlloKuster/Structural_Integrity_3D/blob/main/Animations/Focusing_Device/material_x.gif)

If we then look at $u(x, y, z)$, we can clearly see the non-connected parts having big values (red).

![](https://github.com/OlloKuster/Structural_Integrity_3D/blob/main/Animations/Focusing_Device/heat_material_x.gif)

By using this information, we can use gradient based optimization by "building" on top of the red area until the material is connected to heat sinks, enforcing structural integrity. 

Since we are looking at 3D nanoprinting, we also don't want any cavities appearing inside our design. This means, that the void, or the absence of the material, also has to be fully connected (*i.e.* structurally integral).
So we not only try to connect the material to heat sinks using a heat dissipation solution, we also try to connect the void to their respective heat sinks with another heat dissipation simulation.

Note, that the gifs show the contour of the material for better visualization. The actual designs are filled with material.

## Result

If we optimize our structures using topology optimization, we are able to find designs which fulfil all of our desired constraints. Not only are these fully connected, but we also account for minimum feature size and binarization of the material.

Material before | Material after
:------------------------------------------------------:|:------------------------------------------------------:
![](https://github.com/OlloKuster/Structural_Integrity_3D/blob/main/Animations/Focusing_Device/material_x.gif) | ![](https://github.com/OlloKuster/Structural_Integrity_3D/blob/main/Animations/Focusing_Device/opt_material_x.gif)
![](https://github.com/OlloKuster/Structural_Integrity_3D/blob/main/Animations/Focusing_Device/heat_material_x.gif) | ![](https://github.com/OlloKuster/Structural_Integrity_3D/blob/main/Animations/Focusing_Device/opt_heat_material_x.gif)


Same thing happens if we look at the void

Void before | Void after
:------------------------------------------------------:|:------------------------------------------------------:
![](https://github.com/OlloKuster/Structural_Integrity_3D/blob/main/Animations/Focusing_Device/void_x.gif) | ![](https://github.com/OlloKuster/Structural_Integrity_3D/blob/main/Animations/Focusing_Device/opt_void_x.gif)
![](https://github.com/OlloKuster/Structural_Integrity_3D/blob/main/Animations/Focusing_Device/heat_void_x.gif) | ![](https://github.com/OlloKuster/Structural_Integrity_3D/blob/main/Animations/Focusing_Device/opt_heat_void_x.gif)


## Installation

Note that we do not provide a requirements.txt. We used Python version 3.11 for our optimization and the numpy version has to be <2. The optimization was done using jax with Cuda 11.12. All of the other installations should be fairly trivial.

## How to use the code

Simply run "main.py". By default it will optimize the focusing device with structural integrity. By changing "dispenser.LENS3D.value" to "dispenser.CROSS.value", the waveguide coupler will be optimized.
If you'd like to add your own examples, or change the settings of the existing ones, simply change the examples in the "Problem" folder.
Only one GPU can be used per optimization, but multiple different optimizations can be run in parallel. That's what "main2.py"-"main4.py" are for.
