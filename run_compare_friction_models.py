from pydrake.all import (
    Demultiplexer, 
    Multiplexer, 
    DiagramBuilder,
    SceneGraphConfig,
    AddMultibodyPlant,
    MultibodyPlantConfig,
    Simulator,
    AddDefaultVisualization
)
from pydrake.systems.primitives import LogVectorOutput

import matplotlib.pyplot as plt

from ball_plant.create_ball_plant import add_RoboBall_plant
from ball_plant.joint_modifiers import StictionModel, StictionModel_Majd
from ball_plant.ballParams import RoboBall2Params
from ball_plant.data.plot_these_files import (
    plot_and_export_average_with_std, 
    get_csv_files
    )

def plot_test_data(ax,log_directory, return_data=False):
       
    csvs = get_csv_files(log_directory)

    out_pend = plot_and_export_average_with_std(ax, csvs, time_column='timestamp', value_column="roll_joint.position", label="Average Robot Data")
    first_time = out_pend["time"][0]
    peaks_vals = out_pend["mean"][0]

    return out_pend["ax"], first_time, peaks_vals
    

def run_test(meshcat, starting_angle, friction_model_class, friction_params):
   
    builder = DiagramBuilder()
    sceneConfig = SceneGraphConfig()
    sceneConfig.default_proximity_properties.compliance_type = "compliant"
    plant, scene_graph = AddMultibodyPlant(
        MultibodyPlantConfig(
            time_step=0.001,
            penetration_allowance=0.001,
            contact_surface_representation="polygon",
            contact_model="hydroelastic"
            ), sceneConfig, 
        builder)

    plant, _ = add_RoboBall_plant(plant, 
                                          place_in_stand="hanging")
    
    steer_q_idx = plant.GetStateNames().index("RoboBall_URDF_steer_q")
    steer_w_idx = plant.GetStateNames().index("RoboBall_URDF_steer_w")
    drive_w_idx = plant.GetStateNames().index("RoboBall_URDF_drive_w")

    drive_motor_idx = plant.GetActuatorNames().index("RoboBall_URDF_drive")
    steer_motor_idx = plant.GetActuatorNames().index("RoboBall_URDF_steer")
    
    drive_friction = builder.AddSystem(friction_model_class(friction_params))
    steer_friction = builder.AddSystem(friction_model_class(friction_params))
    # set up meshcat visualizer
    if meshcat != None:
        AddDefaultVisualization(builder, meshcat)

    state_demuxer = builder.AddSystem(Demultiplexer(plant.num_multibody_states()))
    control_muxer = builder.AddSystem(Multiplexer(2))
    
    # demux the states
    builder.Connect(plant.get_state_output_port(), # split the state vector
                    state_demuxer.get_input_port())
    builder.Connect(state_demuxer.get_output_port(steer_w_idx),  # connect the steer speed
                    steer_friction.velocity_input_port)
    builder.Connect(state_demuxer.get_output_port(drive_w_idx),  # connect the drive speed
                    drive_friction.velocity_input_port)
    # mux the forces
    builder.Connect(steer_friction.torque_output_port, control_muxer.get_input_port(steer_motor_idx))
    builder.Connect(drive_friction.torque_output_port, control_muxer.get_input_port(drive_motor_idx))
    
    # Apply the calculated friction torque back to the joint
    builder.Connect(control_muxer.get_output_port(), plant.get_actuation_input_port())

    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    diagram = builder.Build()
    
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    # set initial consitoins to match the data file
    plant.SetPositions(plant_context, [0, starting_angle])
    
    simulator = simulator = Simulator(diagram, diagram_context)
    if meshcat !=None:
        meshcat.StartRecording()
    simulator.AdvanceTo(5)
    if meshcat !=None:
        meshcat.PublishRecording()
    
    # plot_logger_data(logger, simulator, plant.GetStateNames())
    log = logger.FindLog(simulator.get_context())
    times = log.sample_times()
    data = log.data().transpose()
    return times, data[:,steer_q_idx] # return the steer data

if __name__=="__main__":
    
    data_logs = "./ball_plant/data/stiction_data/"
    
    meshcat = None # import and declare an intance if desired

    fig, ax = plt.subplots()
    ax, time_offset, starting_angle = plot_test_data(ax, data_logs)
    
    stiction_params =  [RoboBall2Params().steer_dynamic_friction,
                        RoboBall2Params().steer_static_friction,
                        RoboBall2Params().steer_viscous_damping]

    viscous_params = [0,0,RoboBall2Params().steer_viscous_damping]
    print(f"Running Test for stiction")
    times, data = run_test(meshcat, starting_angle, StictionModel, stiction_params)
    print(f"Running Test for  Madj_et_al")
    times_maj, data_maj =  run_test(meshcat, starting_angle, StictionModel_Majd, RoboBall2Params().majd_friction_params)
    print("Running Test for Viscous Damping")
    times_visc, data_visc = run_test(meshcat, starting_angle, StictionModel, viscous_params)
    print(f"finishing tests, plotting ... ")
    ax.plot(times + time_offset, data, label="Coulomb + Viscous Friction Model")
    ax.plot(times + time_offset, data_visc, label="Viscous Model")
    ax.plot(times_maj + time_offset, data_maj, label="Majd et al.")
    ax.legend()
    ax.grid()
    ax.set_title(f"Robot Data vs Friction Models")
    ax.set_xlabel("time(s)")
    ax.set_ylabel("Angle (rad)")
    plt.show()