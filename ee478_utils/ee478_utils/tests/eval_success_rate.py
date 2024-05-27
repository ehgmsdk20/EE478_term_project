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

class TesterSuccessRate(Tester):
    def __init__(self, args):
        super().__init__(args)

    def run(self) -> None:
        obs = self.env.obs_buf

        failure_record = torch.zeros(self.env_cfg.env.num_envs, dtype=torch.bool, device=self.env.device)
        success_record = torch.zeros(self.env_cfg.env.num_envs, dtype=torch.bool, device=self.env.device)
        max_time = 500
        i = 0

        while not rospy.is_shutdown() and i < max_time:
            robot_pos = self.env.root_states[0][:3]
            cam_pos = [robot_pos[0], robot_pos[1] + 3, robot_pos[2] + 2]
            self.env.set_camera(cam_pos, robot_pos)
            self.env.commands[:, 0] = 0.5
            self.env.commands[:, 1] = 0
            self.env.commands[:, 2] = 0

            success_record, failure_record = self.sr_logger(success_record, failure_record)
            actions = self.policy(obs)

            obs, _, _, _, _ = self.env.step(actions.detach())

            i += 1
            print("Time step:", i, "/500")
        
        record = (1-failure_record.float()) * success_record.float()
        success_rate = 100 * torch.mean(record)
        print("Success rate:", success_rate)

    def sr_logger(self, success_record: torch.Tensor, failure_record: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the success-failure statistics

        Args:
            success_record (torch.Tensor): The success instance record
            failure_record (torch.Tensor): The failure instance record

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated record tuple
        """
        distance = torch.norm(self.env.root_states[:, :2] - self.env.env_origins[:, :2], dim=1)
        success = distance > (self.env.terrain.env_length / 2)
        failure = torch.any(
            torch.norm(self.env.contact_forces[:, self.env.termination_contact_indices, :], dim=-1) > 1.0, dim=1
        )
        failure_record += failure
        success_record += success

        return success_record, failure_record



if __name__ == "__main__":
    args = get_args()
    agent = TesterSuccessRate(args)
    agent.run()
