import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd

from pydrake.systems.framework import DiagramBuilder

from pydrake.multibody.plant import (AddMultibodyPlant, MultibodyPlantConfig)
from pydrake.geometry import SceneGraphConfig
from pydrake.all import Simulator, LogVectorOutput, AddDefaultVisualization, Meshcat, Demultiplexer, Multiplexer, ConstantVectorSource

from utilities.world_features import add_plate
from utilities.compare_models import Pravecek_2025_Model

from ball_plant.create_ball_plant import add_RoboBall_plant, update_bedliner_properties
from ball_plant.joint_modifiers import StictionModel_Majd
from ball_plant.data.plot_these_files import add_avg_data_to_plot


def plot_prav_model(ax, init_conditions, tf, configs=(None,)):
    # set up the drake sim
    
    # Create a simple block diagram containing our system
    sys = Pravecek_2025_Model()

    builder = DiagramBuilder()
    mySys = builder.AddSystem(sys) 
    # wire loggers for data logging 
    logger_output = LogVectorOutput(mySys.get_output_port(), builder) 
    
    diagram = builder.Build()

    # set ic's for pravecek model
    q0_p = [init_conditions[0], 
            init_conditions[0] + init_conditions[2],
            init_conditions[1], 
            init_conditions[1] + init_conditions[3]] # [phi, theta_g, dphi, dtheta_g]
    context = diagram.CreateDefaultContext()
    context.SetContinuousState(q0_p)

    # create the simulator, the modified context with the ICs must be included
    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(tf)

    # Grab output results from Logger:
    log = logger_output.FindLog(context) # find output log with that context
    time = log.sample_times()
    data = log.data().transpose()

    # Grab input results from Logger:
    ax[0].plot(time, data[:, 0], label=f'Pravecek et al. - {configs}')
    ax[1].plot(time, data[:, 1], label=f'Pravecek et al. - {configs}')

    return ax

def plot_drake_model(ax, initi_conditions, tf, config, new_proximity=None, meshcat=None):
    builder = DiagramBuilder()
    
    sceneConfig = SceneGraphConfig()
    sceneConfig.default_proximity_properties.compliance_type = "compliant"

    # check config conflicts:
    if ("soft" in config) and ("point" in config):
        raise ValueError("Cannot model both point and soft models")
    
    # check configs to set up sim
    if "soft" in config:
        plant, scene_graph = AddMultibodyPlant(
            MultibodyPlantConfig(
                time_step=0.001,
                penetration_allowance=0.001,
                contact_surface_representation="polygon",
                contact_model="hydroelastic"
                ), 
                sceneConfig, 
            builder)
    elif "point" in config:
        plant, scene_graph = AddMultibodyPlant(
            MultibodyPlantConfig(
                time_step=0.001,
                contact_model="point"
                ), builder)

    # insert a table (plant, angle [deg])
    plant = add_plate(plant, 0.0, visible=False)

    plant, _ = add_RoboBall_plant(plant, place_in_stand="steer", lumpy_bedliner=("lumpy" in config))
    
    try:
        if new_proximity:
            update_bedliner_properties(scene_graph, new_proximity)
    except ValueError:
         if new_proximity.any():
            update_bedliner_properties(scene_graph, new_proximity)

    drake_logger = LogVectorOutput(plant.get_state_output_port(), builder)
     
    if "stiction" in config:
        steer_friction = builder.AddSystem(StictionModel_Majd([0.204, 0.45, 10, 0.1, 1]))
        steer_w_idx = plant.GetStateNames().index("RoboBall_URDF_steer_w")
        
        state_demuxer = builder.AddSystem(Demultiplexer(plant.num_multibody_states()))
        # demux the states
        builder.Connect(plant.get_state_output_port(), # split the state vector
                        state_demuxer.get_input_port())
        
        builder.Connect(state_demuxer.get_output_port(steer_w_idx),  # connect the steer speed
                        steer_friction.velocity_input_port)

        # Apply the calculated friction torque back to the joint
        builder.Connect(steer_friction.torque_output_port, plant.get_actuation_input_port())
    if meshcat:
        AddDefaultVisualization(builder, meshcat)
    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    
    drake_model_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    
    # set initial positions for drake model, test held constant so velocities are zero
    q0 = [00, 0.304, initi_conditions[0], initi_conditions[2]]
    plant.SetPositions(drake_model_context, q0)
   
    simulator = Simulator(diagram, diagram_context)

    if meshcat:
        meshcat.StartRecording()
    simulator.AdvanceTo(tf)
    if meshcat:
        meshcat.PublishRecording()


    # plot the models
    drake_log = drake_logger.FindLog(simulator.get_context())
    drake_data = drake_log.data().transpose()

    config_str = f""
    if "point" in config:
        config_str = f"{config} "
    elif "soft" in config:
        try:
            if new_proximity == [1.2e5, 0.75]:
                config_str = f"{config} with Hand-Tuned Parameters"
            elif new_proximity == [1.55201e5, 0.405]:
                config_str = f"{config} with Optimized Parameters"
            else:
                config_str = f"{config} with Estimated Contact Parameters"
        except ValueError:
            config_str = "optimizing"

    data = {"time": drake_log.sample_times(),
            "pipe": drake_data[:, 2],
            "pend": drake_data[:, 3]}
    
    try:
        # plot phi
        ax[0].plot(drake_log.sample_times(), drake_data[:, 2], label=f'Drake - '+ config_str)
        # plot theta
        ax[1].plot(drake_log.sample_times(), drake_data[:, 3], label=f'Drake - '+ config_str)
        
        return ax, data
    except TypeError:
        # if ax is None return it
        return None, data
   


if __name__=="__main__":
    meshcat = Meshcat()
    
    fig, ax_point = plt.subplots(2,1, sharex=True, figsize=(6,8))
    fig, ax_soft =  plt.subplots(2,1, sharex=True, figsize=(6,8))
    fig, ax_prav =  plt.subplots(2,1, sharex=True, figsize=(6,8))
    # make sure ax are in the form:
    # ax = [phi, dphi, theta, dtheta]
    # pull the initial condition from the datafile
    ax_prav,_,_ = add_avg_data_to_plot(ax_prav)
    ax_soft, pipe0, pend0 = add_avg_data_to_plot(ax_soft)
    # make sure IC's are in the form:
    # [phi, dphi, theta, dtheta]
    q0 = [pipe0["mean"][0], 0, pend0["mean"][0], 0 ]
    tf = pipe0["time"][-1] # final integrating time
    tuned_proximity = [1.2e5, 0.75]

    ax_prav = plot_prav_model(ax_prav, q0, tf, ("tau_flat",))
    ax_prav, _ = plot_drake_model(ax_prav, q0, tf, ("point",))
    ax_prav, _ = plot_drake_model(ax_prav, q0, tf, ("soft", "stiction"), [1.55201e5, 0.405])


    ax_point =    plot_prav_model(ax_point, [0.0, 0, 1, 0], tf, ("point", ))
    ax_point, _ = plot_drake_model(ax_point, [0.0, 0, 1, 0], tf, ("point",))
    ax_point, _ = plot_drake_model(ax_point, [0.0, 0, 1, 0], tf, ("point", "stiction"))
    ax_point, _ = plot_drake_model(ax_point, [0.0, 0, 1, 0], tf, ("point", "stiction", "lumpy"))


    ax_soft, _ = plot_drake_model(ax_soft, q0, tf, ("soft", "stiction"))
    ax_soft, _ = plot_drake_model(ax_soft, q0, tf, ("soft", "stiction"), tuned_proximity)
    ax_soft, _ = plot_drake_model(ax_soft, q0, tf, ("soft", "stiction"), [1.55201e5, 0.405])
    ax_soft,_ = plot_drake_model(ax_soft, q0, tf,  ("soft", "stiction", "lumpy"), [1.55201e5, 0.405])

    ax_soft[0].set_title("Responses of Models with Hydroelastic Contact")
    ax_point[0].set_title("Responses of Models with Point Contact")
    ax_prav[0].set_title("Pravecek's Models Compared with Modular Model")


    new_axes = [ax_point, ax_soft, ax_prav]

    for new_ax in new_axes:
        # set all the fun stuff
        new_ax[1].set_xlabel("time (s)")
        new_ax[0].set_ylabel("pipe angle: $\phi$ (rad)")
        new_ax[0].set_xlim(left=0, right=tf)
        new_ax[0].grid()
        new_ax[1].set_ylabel("pend angle: $\\theta$ (rad)")
        new_ax[1].set_xlim(left=0, right=tf)
        new_ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        new_ax[1].grid()

    plt.show()
