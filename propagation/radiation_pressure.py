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
import os
import pandas as pd
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
simulation_end_epoch   = DateTime(2000, 1, 3).epoch()

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
global_frame_orientation = "ECLIPJ2000"

# Create default body settings, usually from `spice`.
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Create settings for the GRAIL A satellite (initialize an empyt settings object to which we will add the vehicle shape and mass information)
body_settings.add_empty_settings("GRAIL_A")

# Create settings object for each panel

# first read the panel data from input file
this_file_path = os.path.dirname(os.path.abspath(__file__))
panel_data = pd.read_csv(this_file_path + "/input/grail_macromodel.txt", delimiter=", ", engine="python")
material_data = pd.read_csv(this_file_path + "/input/grail_materials.txt", delimiter=", ", engine="python")

# initialize list to store all panel settings
all_panel_settings = []

for i, row in panel_data.iterrows():
    # create panel geometry settings
    # Options are: frame_fixed_panel_geometry, time_varying_panel_geometry, body_tracking_panel_geometry
    panel_geometry_settings = environment_setup.vehicle_systems.frame_fixed_panel_geometry(
        np.array([row["x"], row["y"], row["z"]]), # panel position in body reference frame
        row["area"] # panel area
    )    
    
    panel_material_data = material_data[material_data["material"] == row["material"]]
        
    # create panel radiation settings (for specular and diffuse reflection)
    specular_diffuse_body_panel_reflection_settings = environment_setup.radiation_pressure.specular_diffuse_body_panel_reflection(
        specular_reflectivity=float(panel_material_data["Cs"].iloc[0]), diffuse_reflectivity=float(panel_material_data["Cd"].iloc[0]), with_instantaneous_reradiation=True
    )
    
    # create settings for complete pannel (combining geometry and material properties relevant for radiation pressure calculations)
    complete_panel_settings = environment_setup.vehicle_systems.body_panel_settings(
        panel_geometry_settings,
        specular_diffuse_body_panel_reflection_settings
    )
    
    # add panel settings to list of all panel settings
    all_panel_settings.append(
        complete_panel_settings
    )

# Create settings object for complete vehicle shape
full_panelled_body_settings = environment_setup.vehicle_systems.full_panelled_body_settings(
    all_panel_settings
)

# set mass of GRAIL_A
body_settings.get("GRAIL_A").constant_mass = 202.4

# need to add an emphermeis setting to GRAIL_A in order to add synchronous rotation model
body_settings.get("GRAIL_A").ephemeris_settings = environment_setup.ephemeris.direct_spice("Moon", "ECLIPJ2000")

# need to add a roation model to GRAIL_A (synchronous) in order to create vehicle shape
body_settings.get("GRAIL_A").rotation_model_settings = environment_setup.rotation_model.synchronous(
    "Moon", "ECLIPJ2000", "VehicleFixed"
)

# add the full panelled body settings to GRAIL_A settings
body_settings.get("GRAIL_A").vehicle_shape_settings = full_panelled_body_settings

# define by which bodies the radiation pressure of GRAIL_A is to be calculated (and which bodies provide shading)
body_settings.get("GRAIL_A").radiation_pressure_target_settings = \
    environment_setup.radiation_pressure.panelled_radiation_target({
        "Sun": ["Moon"]
    })

bodies = environment_setup.create_system_of_bodies(body_settings)


# Define bodies that are propagated
bodies_to_propagate = ["GRAIL_A"]

# Define central body of propagation
central_bodies = ["Moon"]


# Define accelerations acting on GRAIL_A
acceleration_settings_GRAIL_A = dict(
    Moon=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.radiation_pressure()],
)

acceleration_settings = {"GRAIL_A": acceleration_settings_GRAIL_A}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)


# Set initial conditions for the satellite that will be
# propagated in this simulation. The initial conditions are given in
# Keplerian elements and later on converted to Cartesian elements
moon_gravitational_parameter = bodies.get("Moon").gravitational_parameter
initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=moon_gravitational_parameter,
    semi_major_axis=384748 + 500,
    eccentricity=4.03294322e-03,
    inclination=1.71065169e+00,
    argument_of_periapsis=1.31226971e+00,
    longitude_of_ascending_node=3.82958313e-01,
    true_anomaly=3.07018490e+00,
)


# Create termination settings
termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create boring numerical integrator settings
fixed_step_size = 10.0
integrator_settings = propagation_setup.integrator.runge_kutta_4(fixed_step_size)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings
)

# Create simulation object and propagate the dynamics
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Extract the resulting state history and convert it to an ndarray
states = dynamics_simulator.state_history
states_array_1 = result2array(states)



###############################################################################

# lets make it more interesting by adding the effect of the Moon's radiation pressure on the satellite
# for now lets assume a cannonball radiation pressure model, for this we can add to our previous accelearation settings


# Add Moon radiation properties
moon_surface_radiosity_models = [
    environment_setup.radiation_pressure.thermal_emission_angle_based_radiosity(
        minimum_temperature=95.0, maximum_temperature=385.0, constant_emissivity=0.95, original_source_name="Sun" ),
    
    environment_setup.radiation_pressure.variable_albedo_surface_radiosity(
        environment_setup.radiation_pressure.predefined_spherical_harmonic_surface_property_distribution( environment_setup.radiation_pressure.albedo_dlam1 ), "Sun" ) ]

# update moon settings
body_settings.get( "Moon" ).radiation_source_settings = environment_setup.radiation_pressure.panelled_extended_radiation_source(
    moon_surface_radiosity_models, [ 6, 12, 18 ] )

# create bodies with updated settings

bodies = environment_setup.create_system_of_bodies(body_settings)

# first lets add a new radiation pressure target to GRAIL_A for the moon
moon_cannonball_radiation_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    2.0, 0.5, {"Moon": []}
)

environment_setup.add_radiation_pressure_target_model(
    bodies, "GRAIL_A", moon_cannonball_radiation_settings
)

# Next specify that we want to account for this acceleration in the acceleration settings
acceleration_settings_GRAIL_A["Moon"].append(
    propagation_setup.acceleration.radiation_pressure()
)

# Re-create acceleration models
acceleration_settings = {"GRAIL_A": acceleration_settings_GRAIL_A}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

print("\n\nRunning simulation with Moon radiation pressure (cannonball model)")
# We can re-use the same propagator settings, initial state and integrator settings as before
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Extract the resulting state history and convert it to an ndarray
states = dynamics_simulator.state_history
states_array_2 = result2array(states)


###############################################################################

# lets bump up the realism abit more, lets use a panelled radiation pressure model for the effect of the moon on the s/c
# first lets recreate the bodies using the body settings we have already defined (this does not include the canonical radiation pressure model for the moon)

bodies = environment_setup.create_system_of_bodies(body_settings)

# Next we need to add the panelled radiation pressure model for the moon to the moon settings
# Create panelled radiation pressure settings for the moon

moon_panelled_radiation_settings = environment_setup.radiation_pressure.panelled_radiation_target(
    {"Moon": []}
)

environment_setup.add_radiation_pressure_target_model(
    bodies, "GRAIL_A", moon_panelled_radiation_settings
)

print("\n\nRunning simulation with Moon radiation pressure (panelled model)")

dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Extract the resulting state history and convert it to an ndarray
states = dynamics_simulator.state_history
states_array_3 = result2array(states)


# print(body_settings.get("GRAIL_A").radiation_pressure_target_settings)

print("Done!")

sys.exit()


plt.figure()


plt.plot(states_array_2[:, 0] - states_array_1[:, 0], states_array_2[:, 1] - states_array_1[:, 1])
plt.plot(states_array_3[:, 0] - states_array_2[:, 0], states_array_3[:, 1] - states_array_2[:, 1])

plt.show()




# create 3D plot of orbit

# ax = plt.figure().add_subplot(projection='3d')

# ax.plot(states_array_1[:, 1], states_array_1[:, 2], states_array_1[:, 3])
# ax.plot(states_array_2[:, 1], states_array_2[:, 2], states_array_2[:, 3])
# ax.plot(states_array_3[:, 1], states_array_3[:, 2], states_array_3[:, 3])

# plt.show()



