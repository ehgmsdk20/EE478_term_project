import os

import rospy
from geometry_msgs.msg import Twist
from isaacgym import gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa
from legged_gym.utils import task_registry

from legged_gym.utils.helpers import get_args
from ee478_utils.tests.eval_freewalk import Tester
import torch
from typing import Tuple

class TesterTrackingError(Tester):
    def __init__(self, args):
        super().__init__(args)

    def run(self) -> None:
        obs = self.env.obs_buf

        tracking_error_record = torch.zeros(self.env_cfg.env.num_envs, dtype=torch.float, device=self.env.device)
        max_time = 500
        i = 0

        while not rospy.is_shutdown() and i < max_time:
            robot_pos = self.env.root_states[0][:3]
            cam_pos = [robot_pos[0], robot_pos[1] + 3, robot_pos[2] + 2]
            self.env.set_camera(cam_pos, robot_pos)
            self.env.commands[:, 0] = 0.5
            self.env.commands[:, 1] = 0
            self.env.commands[:, 2] = 0

            tracking_error_record = self.tr_logger(tracking_error_record)
            actions = self.policy(obs)

            obs, _, _, _, _ = self.env.step(actions.detach())

            i += 1
        
        velocity_tracking_error = torch.mean(tracking_error_record.float()) / max_time
        print("Velocity tracking error:", velocity_tracking_error)

    def tr_logger(self, tracking_error_record: torch.Tensor) -> torch.Tensor:

        velocity_tracking_error = torch.norm(self.env.base_lin_vel[:, :2] - self.env.commands[:, :2], dim=1)
        tracking_error_record += velocity_tracking_error

        return tracking_error_record



if __name__ == "__main__":
    args = get_args()
    agent = TesterTrackingError(args)
    agent.run()
