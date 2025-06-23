class RoboBall2Params():
    """
    Important Measurable Values for the Ball Plant
    not found in the URDF
    """
    def __init__(self):
        self.steer_dynamic_friction = 0.7
        self.steer_static_friction = 0.65
        self.steer_viscous_damping = 0.104

        self.majd_friction_params = [0.204, 0.45, 10, 0.1, 1] # [f_w, f_c, sigma, w_c, n ]
        # robot urdf
        self.package_path = "./ball_plant/RoboBall_URDF/package.xml"
        self.robot_file = "./ball_plant/RoboBall_URDF/urdf/RoboBall_URDF.urdf"
        self.lumpy_robot_file = "./ball_plant/RoboBall_URDF/urdf/RoboBall_URDF_lumpy.urdf"
      