# Perturbed satellite orbit
"""
Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.

"""

## Context
"""

"""

## Import statements
"""
The required import statements are made here, at the very beginning.

Some standard modules are first loaded. These are `numpy` and `matplotlib.pyplot`.

Then, the different modules of `tudatpy` that will be used are imported.
"""

# Load standard modules
import numpy as np
import sys
import matplotlib
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime

## Configuration

# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs
simulation_start_epoch = DateTime(2000, 1, 1).epoch()
simulation_end_epoch   = DateTime(2000, 1, 2).epoch()

## Environment setup
"""
Let's create the environment for our simulation. This setup covers the creation of (celestial) bodies, vehicle(s), and environment interfaces.

"""

### Create the bodies
"""
Bodies can be created by making a list of strings with the bodies that is to be included in the simulation.

The default body settings (such as atmosphere, body shape, rotation model) are taken from `SPICE`.

These settings can be adjusted. Please refere to the [Available Environment Models](https://tudat-space.readthedocs.io/en/latest/_src_user_guide/state_propagation/environment_setup/create_models/available.html#available-environment-models) in the user guide for more details.

Finally, the system of bodies is created using the settings. This system of bodies is stored into the variable `bodies`.
"""

# Define string names for bodies to be created from default.
bodies_to_create = ["Sun", "Moon"]

# Use "Earth"/"J2000" as global frame origin and orientation.
global_frame_origin = "Moon"
global_frame_orientation = "J2000"

# Create default body settings, usually from `spice`.
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

body_settings.add_empty_settings("GRAIL_A")

print(dir(body_settings.get("GRAIL_A").vehicle_shape_settings))

full_panelled_body_settings = environment_setup.vehicle_systems.full_panelled_body_settings(
    "tes"
)





sys.exit()
# Create system of selected celestial bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

### Create the vehicle
"""
Let's now create the GRAIL A satellite.
"""

# Create vehicle objects.
bodies.create_empty_body("GRAIL_A")

print(dir(bodies.get_body("GRAIL_A").vehicle_systems))

print(dir(environment_setup.vehicle_systems))
