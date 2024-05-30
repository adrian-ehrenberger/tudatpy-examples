import numpy as np
import matplotlib.pyplot as plt
import os, sys
import scipy.interpolate
import scipy.interpolate.interpnd

rerun_sims = False

if rerun_sims:
    import radiation_pressure_A
    import radiation_pressure_B
    import radiation_pressure_C


# read data
this_filepath = os.path.dirname(os.path.abspath(__file__))

states_A = np.genfromtxt(os.path.join(this_filepath, "output", "radiation_pressure", "states_A.dat"))
states_B = np.genfromtxt(os.path.join(this_filepath, "output", "radiation_pressure", "states_B.dat"))
states_C = np.genfromtxt(os.path.join(this_filepath, "output", "radiation_pressure", "states_C.dat"))


dep_vars_A = np.genfromtxt(os.path.join(this_filepath, "output", "radiation_pressure", "dep_vars_A.dat"))
dep_vars_B = np.genfromtxt(os.path.join(this_filepath, "output", "radiation_pressure", "dep_vars_B.dat"))
dep_vars_C = np.genfromtxt(os.path.join(this_filepath, "output", "radiation_pressure", "dep_vars_C.dat"))


time_array = states_A[:, 0]

states_A_C = states_A - states_C
states_B_C = states_B - states_C

dep_vars_A_C = dep_vars_A[:, 0:4] - dep_vars_C[:, 0:4]
dep_vars_B_C = dep_vars_B[:, 0:4] - dep_vars_C[:, 0:4]

orbital_period = 2 * np.pi * ( 1788000**3 / 4902800121846.8 ) ** (1/2)

no_orbits_array = time_array / orbital_period

# plot x, y, z position, velocity and acceleration error of GRAIL_A w.r.t. model C (the most accurate model)

def integrator_error():
    

    states_A_high_accuracy = np.genfromtxt(os.path.join(this_filepath, "output", "radiation_pressure", "states_A_high_accuracy.dat"))
    states_B_high_accuracy = np.genfromtxt(os.path.join(this_filepath, "output", "radiation_pressure", "states_B_high_accuracy.dat"))
    states_C_high_accuracy = np.genfromtxt(os.path.join(this_filepath, "output", "radiation_pressure", "states_C_high_accuracy.dat"))

    dep_vars_A_high_accuracy = np.genfromtxt(os.path.join(this_filepath, "output", "radiation_pressure", "dep_vars_A_high_accuracy.dat"))
    dep_vars_B_high_accuracy = np.genfromtxt(os.path.join(this_filepath, "output", "radiation_pressure", "dep_vars_B_high_accuracy.dat"))
    dep_vars_C_high_accuracy = np.genfromtxt(os.path.join(this_filepath, "output", "radiation_pressure", "dep_vars_C_high_accuracy.dat"))
    # create interpolators
    states_A_interpolator = scipy.interpolate.interp1d(states_A[:, 0], states_A.T)
    states_B_interpolator = scipy.interpolate.interp1d(states_B[:, 0], states_B.T)
    states_C_interpolator = scipy.interpolate.interp1d(states_C[:, 0], states_C.T)
    
    states_A_high_accuracy_interpolator = scipy.interpolate.interp1d(states_A_high_accuracy[:, 0], states_A_high_accuracy.T)
    states_B_high_accuracy_interpolator = scipy.interpolate.interp1d(states_B_high_accuracy[:, 0], states_B_high_accuracy.T)
    states_C_high_accuracy_interpolator = scipy.interpolate.interp1d(states_C_high_accuracy[:, 0], states_C_high_accuracy.T)
    
    error_A = states_A_high_accuracy_interpolator(time_array).T - states_A_interpolator(time_array).T
    error_B = states_B_high_accuracy_interpolator(time_array).T - states_B_interpolator(time_array).T
    error_C = states_C_high_accuracy_interpolator(time_array).T - states_C_interpolator(time_array).T
    
    pos_error_A = np.linalg.norm(error_A[:, 1:4], axis=1)
    pos_error_B = np.linalg.norm(error_B[:, 1:4], axis=1)
    pos_error_C = np.linalg.norm(error_C[:, 1:4], axis=1)
    
    
    vel_error_A = np.linalg.norm(error_A[:, 4:7], axis=1)
    vel_error_B = np.linalg.norm(error_B[:, 4:7], axis=1)
    vel_error_C = np.linalg.norm(error_C[:, 4:7], axis=1)
    
     
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs = axs.flatten()
    
    fig.suptitle("Estimate of numerical error in position and velocity of GRAIL_A\n(obtained by comparing to integration using half the time step size)")
    
    axs[0].plot(no_orbits_array, pos_error_A, label="Model A")
    axs[0].plot(no_orbits_array, pos_error_B, label="Model B")
    axs[0].plot(no_orbits_array, pos_error_C, label="Model C")
    
    axs[0].legend()
    axs[0].grid()
    
    axs[0].set_ylabel("Position error [m]")    
    axs[0].set_xlabel("No. of orbits [-]")
    
    
    axs[1].plot(no_orbits_array, vel_error_A, label="Model A")
    axs[1].plot(no_orbits_array, vel_error_B, label="Model B")
    axs[1].plot(no_orbits_array, vel_error_C, label="Model C")
    
    axs[1].legend()
    axs[1].grid()
    
    axs[1].set_ylabel("Velocity error [m/s]")    
    axs[1].set_xlabel("No. of orbits [-]")
    
    fig.tight_layout()
    
def pos_vel_accel_model_comparison():
    
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

def radiation_pressure_plots():

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

    plt.figure(figsize=(8, 6))

    plt.plot(no_orbits_array, dep_vars_B[:, 7], label="Model B, Moon x", color="navy", alpha=0.8)
    plt.plot(no_orbits_array, dep_vars_B[:, 8], label="Model B, Moon y", color="darkred", alpha=0.8)
    plt.plot(no_orbits_array, dep_vars_B[:, 9], label="Model B, Moon z", color="darkgreen", alpha=0.8)


    plt.plot(no_orbits_array, dep_vars_C[:, 7], label="Model C, Moon x", color="royalblue", ls="--")
    plt.plot(no_orbits_array, dep_vars_C[:, 8], label="Model C, Moon y", color="indianred", ls="--")
    plt.plot(no_orbits_array, dep_vars_C[:, 9], label="Model C, Moon z", color="seagreen", ls="--")

    plt.title("Radiation pressure acceleration on GRAIL_A due to Moon in different models")

    # plt.yscale("symlog")

    plt.xlim(0, 2)

    plt.xlabel("No. of orbits [-]")
    plt.ylabel(r"Radiation pressure acceleration [$m/s^2$]")
    plt.grid()

    plt.tight_layout()

    print("Max acceleration due to on Moon radiation pressure in model B:", np.max(np.linalg.norm(dep_vars_B[:, 7:10], axis=1)), "m/s2")
    print("Max acceleration due to on Moon radiation pressure in model C:", np.max(np.linalg.norm(dep_vars_C[:, 7:10], axis=1)), "m/s2")


    plt.legend()
    
    
if __name__ == "__main__":

    integrator_error()
    pos_vel_accel_model_comparison()
    radiation_pressure_plots()

    plt.show()
