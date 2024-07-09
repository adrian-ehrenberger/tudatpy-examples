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
sys.path.insert(0, "/home/tudat-bundle/build/tudatpy")

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
simulation_end_epoch   = simulation_start_epoch + 2.0 * 24 * 60**2

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

# add the full panelled body settings to GRAIL_A settings
body_settings.get("GRAIL_A").vehicle_shape_settings = full_panelled_body_settings

# set mass of GRAIL_A
body_settings.get("GRAIL_A").constant_mass = 202.4

# need to add an emphermeis setting to GRAIL_A in order to add synchronous rotation model
body_settings.get("GRAIL_A").ephemeris_settings = environment_setup.ephemeris.direct_spice("Moon", "ECLIPJ2000")

# need to add a roation model to GRAIL_A (synchronous) in order to create vehicle shape
body_settings.get("GRAIL_A").rotation_model_settings = environment_setup.rotation_model.synchronous(
    "Moon", "ECLIPJ2000", "VehicleFixed"
)


# Define bodies that are propagated
bodies_to_propagate = ["GRAIL_A"]

# Define central body of propagation
central_bodies = ["Moon"]


moon_gravitational_parameter = spice.get_body_gravitational_parameter("Moon")

initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=moon_gravitational_parameter,
    semi_major_axis=1788000,
    eccentricity=4.03294322e-03,
    inclination=np.rad2deg(90),
    argument_of_periapsis=1.31226971e+00,
    longitude_of_ascending_node=3.82958313e-01,
    true_anomaly=3.07018490e+00,
)

# Create termination settings
termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create boring numerical integrator settings
fixed_step_size = 1
integrator_settings = propagation_setup.integrator.runge_kutta_4(fixed_step_size)


#####################################################
# start with model A

dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("GRAIL_A"),
    propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.radiation_pressure_type, "GRAIL_A", "Sun")
]



bodies_model_A = environment_setup.create_system_of_bodies(body_settings)

sun_panelled_radiation_target_settings = \
    environment_setup.radiation_pressure.panelled_radiation_target({
        "Sun": ["Moon"]
    })

environment_setup.add_radiation_pressure_target_model(
    bodies_model_A, "GRAIL_A", sun_panelled_radiation_target_settings
)

# Define accelerations acting on GRAIL_A
acceleration_settings_GRAIL_A_model_A = dict(
    Moon=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.radiation_pressure()],
)


acceleration_settings_model_A = {"GRAIL_A": acceleration_settings_GRAIL_A_model_A}

# Create acceleration models
acceleration_models_model_A = propagation_setup.create_acceleration_models(
    bodies_model_A, acceleration_settings_model_A, bodies_to_propagate, central_bodies
)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models_model_A,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings,
    output_variables=dependent_variables_to_save
)

# Create simulation object and propagate the dynamics
dynamics_simulator_model_A = numerical_simulation.create_dynamics_simulator(
    bodies_model_A, propagator_settings
)



#####################################################
# Model B


dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("GRAIL_A"),
    propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.radiation_pressure_type, "GRAIL_A", "Sun"),
    propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.radiation_pressure_type, "GRAIL_A", "Moon")
]


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

bodies_model_B = environment_setup.create_system_of_bodies(body_settings)

# first lets add a new radiation pressure target to GRAIL_A for the moon
moon_cannonball_radiation_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    2.0, 0.5, {"Moon": []}
)

environment_setup.add_radiation_pressure_target_model(
    bodies_model_B, "GRAIL_A", moon_cannonball_radiation_settings
)

sun_cannonball_radiation_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    2.0, 0.5, {"Sun": ["Moon"]}
)
environment_setup.add_radiation_pressure_target_model(
    bodies_model_B, "GRAIL_A", sun_cannonball_radiation_settings
)

acceleration_settings_GRAIL_A_model_B = dict(
    Moon=[propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.radiation_pressure()],
    Sun=[propagation_setup.acceleration.radiation_pressure()],
)

# Re-create acceleration models
acceleration_settings_model_B = {"GRAIL_A": acceleration_settings_GRAIL_A_model_B}

acceleration_models_model_B = propagation_setup.create_acceleration_models(
    bodies_model_B, acceleration_settings_model_B, bodies_to_propagate, central_bodies
)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models_model_B,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings,
    output_variables=dependent_variables_to_save
)

print("\n\nRunning simulation with Moon radiation pressure (cannonball model)")
# We can re-use the same propagator settings, initial state and integrator settings as before
dynamics_simulator_model_B = numerical_simulation.create_dynamics_simulator(
    bodies_model_B, propagator_settings
)


################################################
# Model C

dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("GRAIL_A"),
    propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.radiation_pressure_type, "GRAIL_A", "Sun"),
    propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.radiation_pressure_type, "GRAIL_A", "Moon")
]
    

bodies_model_C = environment_setup.create_system_of_bodies(body_settings)

# Next we need to add the panelled radiation pressure model for the moon to the moon settings
# Create panelled radiation pressure settings for the moon

moon_panelled_radiation_settings = environment_setup.radiation_pressure.panelled_radiation_target(
    {"Moon": []}
)

sun_panelled_radiation_settings = environment_setup.radiation_pressure.panelled_radiation_target(
    {"Sun": ["Moon"]}
)

environment_setup.add_radiation_pressure_target_model(
    bodies_model_C, "GRAIL_A", moon_panelled_radiation_settings
)
environment_setup.add_radiation_pressure_target_model(
    bodies_model_C, "GRAIL_A", sun_panelled_radiation_settings
)

# Define accelerations acting on GRAIL_A
acceleration_settings_GRAIL_A = dict(
    Moon=[propagation_setup.acceleration.point_mass_gravity(),
          propagation_setup.acceleration.radiation_pressure()],
    Sun=[propagation_setup.acceleration.radiation_pressure()],
)

# Re-create acceleration models
acceleration_settings = {"GRAIL_A": acceleration_settings_GRAIL_A}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies_model_C, acceleration_settings, bodies_to_propagate, central_bodies
)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings,
    output_variables=dependent_variables_to_save
)


print("\n\nRunning simulation with Moon radiation pressure (panelled model)")

dynamics_simulator_model_C = numerical_simulation.create_dynamics_simulator(
    bodies_model_C, propagator_settings
)


# PLOTTING


states_A = result2array(dynamics_simulator_model_A.state_history)
states_B = result2array(dynamics_simulator_model_B.state_history)
states_C = result2array(dynamics_simulator_model_C.state_history)


dep_vars_A = result2array(dynamics_simulator_model_A.dependent_variable_history)
dep_vars_B = result2array(dynamics_simulator_model_B.dependent_variable_history)
dep_vars_C = result2array(dynamics_simulator_model_C.dependent_variable_history)

time_array = states_A[:, 0]

states_A_C = states_A - states_C
states_B_C = states_B - states_C

dep_vars_A_C = dep_vars_A[:, 0:4] - dep_vars_C[:, 0:4]
dep_vars_B_C = dep_vars_B[:, 0:4] - dep_vars_C[:, 0:4]

orbital_period = 2 * np.pi * ( 1788000**3 / moon_gravitational_parameter ) ** (1/2)

no_orbits_array = time_array / orbital_period


# plot radiation pressures

plt.figure(figsize=(8, 6))

plt.plot(no_orbits_array, dep_vars_A[:, 4], label="Model A, Sun x", lw=6)
plt.plot(no_orbits_array, dep_vars_A[:, 5], label="Model A, Sun y", lw=6)
plt.plot(no_orbits_array, dep_vars_A[:, 6], label="Model A, Sun z", lw=6)

plt.plot(no_orbits_array, dep_vars_B[:, 4], label="Model B, Sun x", lw=3)
plt.plot(no_orbits_array, dep_vars_B[:, 5], label="Model B, Sun y", lw=3)
plt.plot(no_orbits_array, dep_vars_B[:, 6], label="Model B, Sun z", lw=3)


plt.plot(no_orbits_array, dep_vars_C[:, 4], label="Model C, Sun x")
plt.plot(no_orbits_array, dep_vars_C[:, 5], label="Model C, Sun y")
plt.plot(no_orbits_array, dep_vars_C[:, 6], label="Model C, Sun z")

plt.title("Radiation pressure acceleration on GRAIL_A due to Sun in different models")

plt.xlabel("No. of orbits [-]")
plt.ylabel(r"Radiation pressure acceleration [$m/s^2$]")
plt.grid()

plt.xlim(0, 2)
plt.tight_layout()

plt.legend()

plt.savefig("pressures.png")

plt.figure(figsize=(8, 6))

plt.plot(no_orbits_array, np.linalg.norm(dep_vars_B[:, 7:10], axis=1), label="Model B", color="navy", alpha=0.8)
plt.plot(no_orbits_array, np.linalg.norm(dep_vars_C[:, 7:10], axis=1), label="Model C", color="royalblue", ls="--")

plt.yscale("log")

plt.title("L-2 norm of radiation pressure acceleration on GRAIL_A due to Moon in different models")

plt.xlabel("No. of orbits [-]")
plt.ylabel(r"Radiation pressure acceleration [$m/s^2$]")
plt.grid()

plt.xlim(0, 2)
plt.tight_layout()

plt.legend()
plt.savefig("Norm of rad pressures.png")

plt.figure(figsize=(8, 6))

plt.plot(no_orbits_array, dep_vars_B[:, 7], label="Model B, Moon x", color="navy", alpha=0.8)
plt.plot(no_orbits_array, dep_vars_B[:, 8], label="Model B, Moon y", color="darkred", alpha=0.8)
plt.plot(no_orbits_array, dep_vars_B[:, 9], label="Model B, Moon z", color="darkgreen", alpha=0.8)


plt.plot(no_orbits_array, dep_vars_C[:, 7], label="Model C, Moon x", color="royalblue", ls="--")
plt.plot(no_orbits_array, dep_vars_C[:, 8], label="Model C, Moon y", color="indianred", ls="--")
plt.plot(no_orbits_array, dep_vars_C[:, 9], label="Model C, Moon z", color="seagreen", ls="--")

plt.title("Radiation pressure acceleration on GRAIL_A due to Moon in different models")

plt.xlim(0, 2)

plt.xlabel("No. of orbits [-]")
plt.ylabel(r"Radiation pressure acceleration [$m/s^2$]")
plt.grid()

plt.tight_layout()

# print("Max acceleration due to on Moon radiation pressure in model B:", np.max(np.linalg.norm(dep_vars_B[:, 7:10], axis=1)), "m/s2")
# print("Max acceleration due to on Moon radiation pressure in model C:", np.max(np.linalg.norm(dep_vars_C[:, 7:10], axis=1)), "m/s2")

plt.legend()

plt.savefig("radiation_pressure_diff_models.png")



### pos vel accell

fig, axs = plt.subplots(3, 1, figsize=(8, 6))
axs = axs.flatten()

axs[0].set_title("Position error between models")
axs[0].plot(no_orbits_array, np.linalg.norm(states_A_C[:, 1:4], axis=1), label="A w.r.t. C")
axs[0].plot(no_orbits_array, np.linalg.norm(states_B_C[:, 1:4], axis=1), label="B w.r.t. C")
axs[0].set_ylabel("Error [m]")

axs[1].set_title("Velocity error between models")
axs[1].plot(no_orbits_array, np.linalg.norm(states_A_C[:, 5:7], axis=1), label="A w.r.t. C")
axs[1].plot(no_orbits_array, np.linalg.norm(states_B_C[:, 4:7], axis=1), label="B w.r.t. C")
axs[1].set_ylabel("Error [m/s]")

axs[2].set_title("Acceleration error between models")
axs[2].plot(no_orbits_array, np.linalg.norm(dep_vars_A_C[:, 1:4], axis=1), label="A w.r.t. C")
axs[2].plot(no_orbits_array, np.linalg.norm(dep_vars_B_C[:, 1:4], axis=1), label="B w.r.t. C")
axs[2].set_ylabel("Error [m/s^2]")

for ax in axs:
    ax.legend()
    ax.set_xlabel("No. of orbits [-]")
    ax.grid()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # ax.set_yscale("log")
    
fig.tight_layout()

fig.savefig("pos_vel_acc_diff_models.png")

fig, axs = plt.subplots(3, 2, figsize=(8, 6))
axs = axs.flatten()

axs[0].set_title("Position error model A w.r.t. model C")
axs[0].plot(no_orbits_array, states_A_C[:, 1], label="x")
axs[0].plot(no_orbits_array, states_A_C[:, 2], label="y")
axs[0].plot(no_orbits_array, states_A_C[:, 3], label="z")
axs[0].set_ylabel("Error [m]")

axs[1].set_title("Position error model B w.r.t. model C")
axs[1].plot(no_orbits_array, states_B_C[:, 1], label="x")
axs[1].plot(no_orbits_array, states_B_C[:, 2], label="y")
axs[1].plot(no_orbits_array, states_B_C[:, 3], label="z")
axs[1].set_ylabel("Error [m]")

axs[2].set_title("Velocity error model A w.r.t. model C")
axs[2].plot(no_orbits_array, states_A_C[:, 4], label="x")
axs[2].plot(no_orbits_array, states_A_C[:, 5], label="y")
axs[2].plot(no_orbits_array, states_A_C[:, 6], label="z")
axs[2].set_ylabel("Error [m/s]")

axs[3].set_title("Velocity error model B w.r.t. model C")
axs[3].plot(no_orbits_array, states_B_C[:, 4], label="x")
axs[3].plot(no_orbits_array, states_B_C[:, 5], label="y")
axs[3].plot(no_orbits_array, states_B_C[:, 6], label="z")
axs[3].set_ylabel("Error [m/s]")

axs[4].set_title("Acceleration error model A w.r.t. model C")
axs[4].plot(no_orbits_array, dep_vars_A_C[:, 1], label="x")
axs[4].plot(no_orbits_array, dep_vars_A_C[:, 2], label="y")
axs[4].plot(no_orbits_array, dep_vars_A_C[:, 3], label="z")
axs[4].set_ylabel("Error [m/s^2]")

axs[5].set_title("Acceleration error model B w.r.t. model C")
axs[5].plot(no_orbits_array, dep_vars_B_C[:, 1], label="x")
axs[5].plot(no_orbits_array, dep_vars_B_C[:, 2], label="y")
axs[5].plot(no_orbits_array, dep_vars_B_C[:, 3], label="z")
axs[5].set_ylabel("Error [m/s^2]")

for ax in axs:
    ax.legend()
    ax.set_xlabel("No. of orbits [-]")
    ax.grid()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # ax.set_yscale("log")
    
fig.tight_layout()


fig.savefig("pos_vel_acc_diff_models_xyz.png")


