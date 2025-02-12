import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgDreamer

class A1DreamerCfg(LeggedRobotCfg):
    
    class env( LeggedRobotCfg.env ): 
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 48 + 187 + 24 #+ 16 #+ 11 * 17
        num_privileged_obs = 48 + 187 + 24 #+ 16 #+ 11 * 17
        privileged_dim = 24 + 3  # privileged_obs[:,:privileged_dim] is the privileged information in privileged_obs, include 3-dim base linear vel
        height_dim = 187  # privileged_obs[:,-height_dim:] is the heightmap in privileged_obs
    
    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 50 # [m]  change 25 to 50
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 0 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [wave, rough slope, stairs up, stairs down, discrete, rough_flat]
        terrain_proportions = [0.1, 0.1, 0.30, 0.25, 0.15, 0.1]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,  # [rad]
            'RL_hip_joint': 0.0,  # [rad]
            'FR_hip_joint': 0.0,  # [rad]
            'RR_hip_joint': 0.0,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.0,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.0,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 28.}  # [N*m/rad]
        damping = {'joint': 0.7}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    # class terrain( LeggedRobotCfg.terrain ):
    #     mesh_type = 'plane'
    #     measure_heights = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        # terminate_after_contacts_on = [
        #     "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
        #     "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        # self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand:
        randomize_friction = True
        friction_range = [0.05, 2.75]
        randomize_restitution = True
        restitution_range = [0, 1.0]

        randomize_base_mass = True
        added_mass_range = [0., 3.]  #kg
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_com_pos = True
        com_pos_range = [-0.05, 0.05]

        push_robots = True
        push_interval_s = 15
        min_push_interval_s = 15
        max_push_vel_xy = 1.0

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_action_latency = True
        randomize_obs_latency = False
        latency_range = [0.00, 0.02]

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            # lin_vel = 0.1
            lin_vel = 0 # set lin_vel as privileged information
            ang_vel = 0.3
            gravity = 0.05
            # height_measurements = 0.1
            height_measurements = 0 #only for critic

    class rewards( LeggedRobotCfg.rewards ):
        reward_curriculum = True
        reward_curriculum_term = ["lin_vel_z"]
        reward_curriculum_schedule = [0, 1000, 1.0, 0]  #from iter 0 to iter 1000, decrease from 1 to 0
        # reward_curriculum_term = ["torques","dof_acc","feet_air_time","collision", "action_rate"]
        # reward_curriculum_schedule = [0, 1000, 0.1, 1.0]  #from iter o to iter 1000, decrease from 1 to 0
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        foot_height_target = 0.15
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            torques = -0.0001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time =  0.5
            collision = -0.1
            feet_stumble = -0.0
            action_rate = -0.01

            lin_vel_z = -2.0
            ang_vel_xy = 0#-0.05
            orientation = 0#-2.0
            action_magnitude = 0#-0.3


            # clearance = -0.2
            # dof_pos_dif = 0#-0.1
            # stand_still = -0.
            # smoothness = 0#-0.01
            # power = -5.0e-4
            # power_distribution = -1.0e-4

    class commands:
        curriculum = True
        max_lin_vel_forward_x_curriculum = 1.0
        max_lin_vel_backward_x_curriculum = 1.0
        max_lin_vel_y_curriculum = 0.2
        max_ang_vel_yaw_curriculum = 0.5

        max_flat_lin_vel_forward_x_curriculum = 2.0
        max_flat_lin_vel_backward_x_curriculum = 1.0
        max_flat_lin_vel_y_curriculum = 1.0
        max_flat_ang_vel_yaw_curriculum = 4.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.4, 0.4] # min max [m/s]
            lin_vel_y = [-0.2, 0.2]   # min max [m/s]
            ang_vel_yaw = [-0.2, 0.2]    # min max [rad/s]
            heading = [-3.14 / 4, 3.14 / 4]

            flat_lin_vel_x = [-0.4, 0.4] # min max [m/s]
            flat_lin_vel_y = [-0.2, 0.2]   # min max [m/s]
            flat_ang_vel_yaw = [-0.2, 0.2]    # min max [rad/s]
            flat_heading = [-3.14 / 4, 3.14 / 4]
        
class A1DreamerCfgDreamer(LeggedRobotCfgDreamer):
    class runner:
        run_name = 'test'
        experiment_name = 'a1_dreamer'
        
