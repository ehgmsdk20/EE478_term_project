import os

import rospy
from geometry_msgs.msg import Twist
from isaacgym import gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict, get_args, parse_sim_params, update_cfg_from_args
from rsl_rl.runners import OnPolicyRunner

class Tester:
    def __init__(self, args):

        self.env_cfg, self.alg_cfg = task_registry.get_cfgs(args.task)
        self.env_cfg, self.alg_cfg = update_cfg_from_args(self.env_cfg, self.alg_cfg, args)
        self.env_cfg.terrain.num_rows = 5
        self.env_cfg.terrain.num_cols = 5
        self.env_cfg.terrain.curriculum = False
        self.env_cfg.terrain.max_init_terrain_level = 5
        self.env_cfg.noise.add_noise = False
        self.env_cfg.noise.noise_scales.dof_pos = 0.1
        self.env_cfg.domain_rand.randomize_friction = True
        self.env_cfg.domain_rand.push_robots = False

        self.env_cfg.env.num_envs = 100

        self.env_cfg.commands.resampling_time = 3

        terrain_dict = {
            "smooth_slope": 0.0,
            "rough_slope": 0.0,
            "stairs_up": 1.0,
            "stairs_down": 0.0,
            "discrete": 0.0,
        }

        # convert dictionary to list
        self.env_cfg.terrain.terrain_proportions = list(terrain_dict.values())

        sim_params = {"sim": class_to_dict(self.env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)

        physics_engine = gymapi.SIM_PHYSX
        self.sim_device = "cuda:0"
        env_class = task_registry.get_task_class(args.task)
        self.env = env_class(self.env_cfg, sim_params, physics_engine, self.sim_device, headless=args.headless)

        # get the path 2 levels up from URL_GYM_ROOT_DIR
        up_two_levels = os.path.join(LEGGED_GYM_ROOT_DIR, "../..")
        log_root = os.path.join(up_two_levels, "logs", self.alg_cfg.runner.experiment_name)
        log_dir = os.path.join(log_root, args.load_run)
        log_file = os.path.join(log_dir, "model_" + str(args.checkpoint) + ".pt")
        print(log_file)

        alg_cfg_dict = class_to_dict(self.alg_cfg)
        self.runner = OnPolicyRunner(self.env, alg_cfg_dict, log_dir)
        self.runner.load(log_file)
        self.policy = self.runner.get_inference_policy(device="cuda:0")

        self.lin_speed = [0, 0, 0]
        self.ang_speed = 0

        self._teleop_cmd_sub = rospy.Subscriber("/cmd_vel", Twist, self.TeleopCmdCallback)
        rospy.init_node("policy_controller", anonymous=True)

    def TeleopCmdCallback(self, msg) -> None:
        """Subscriber callback to receive any teleop command

        Args:
            msg (rospy.msg.Twist): Twist command message. Sent from any node outside.
        """
        self.lin_speed = [msg.linear.x, msg.linear.y, msg.linear.z]
        self.ang_speed = msg.angular.z
        print(self.lin_speed, self.ang_speed)

    def run(self) -> None:
        obs = self.env.obs_buf

        while not rospy.is_shutdown():
            robot_pos = self.env.root_states[0][:3]
            cam_pos = [robot_pos[0], robot_pos[1] + 3, robot_pos[2] + 2]
            self.env.set_camera(cam_pos, robot_pos)
            self.env.commands[:, 0] = 0.5
            self.env.commands[:, 1] = 0
            self.env.commands[:, 2] = 0

            actions = self.policy(obs)

            obs, _, _, _, _ = self.env.step(actions.detach())


if __name__ == "__main__":
    args = get_args()
    agent = Tester(args)
    agent.run()
