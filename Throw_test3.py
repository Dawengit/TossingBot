import os
import pybullet as p
import pybullet_data
import numpy as np
import math
import gymnasium as gym
import time
from gymnasium import spaces
#--------------------------------
from typing import Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.callbacks import BaseCallback

# Define helper functions
def normalized_angle(x: float) -> float:
    """Map angle to (-π, π) range."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def get_joint_limits(body_id, joint_ids):
    """Query the joint limits of the specified joints, return a tuple of (lower limit, upper limit)."""
    joint_limits = []
    for joint_id in joint_ids:
        joint_info = p.getJointInfo(body_id, joint_id)
        joint_limit = joint_info[8], joint_info[9]
        joint_limits.append(joint_limit)
    joint_limits = np.transpose(joint_limits)  # Dx2 -> 2xD
    return joint_limits

# Define the Robot class
class Robot:
    """
    Robot configuration and motion control class
    """
    def __init__(self, uid, control_dt):
        # Simulation configuration
        self.uid = uid
        self.control_dt = control_dt

        # Kinematics configuration
        self.joint_index_last = 13
        self.joint_index_endeffector_base = 7
        self.joint_indices_arm = np.array(range(1, self.joint_index_endeffector_base))  # Arm joint indices (excluding base)
        self.joint_limits = get_joint_limits(uid, range(self.joint_index_last+1))  # Get joint limits
        self.joint_range = (np.array(self.joint_limits[1]) - np.array(self.joint_limits[0]))  # Joint range

        self.rest_pose = np.array((
            0.0,    # Base (fixed)
            0.0,    # Joint 1
            -2.094, # Joint 2
            1.57,   # Joint 3
            -1.047, # Joint 4
            -1.57,  # Joint 5
            0,      # Joint 6
            0.0,    # End effector base (fixed)
            0.785,  # End effector finger
        ))

        # Motion configuration
        self.reset()  # Reset robot state

    def reset(self):
        # Initialize motion
        self.current_pose = np.copy(self.rest_pose)  # Current target pose

        for i in self.joint_indices_arm:
            p.resetJointState(self.uid, i, self.current_pose[i])
        self._finger_control(self.current_pose[self.joint_index_endeffector_base+1])

    def _finger_control(self, target):
        """
        Control the finger joints of the gripper, simulating mimic joints in ROS.
        """
        # Directly control the main finger joints
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+1, p.POSITION_CONTROL, 
                                targetPosition=target)
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+4, p.POSITION_CONTROL, 
                                targetPosition=target)
        # Get current finger joint positions and velocities
        finger_left = p.getJointState(self.uid, self.joint_index_endeffector_base+1)
        finger_right = p.getJointState(self.uid, self.joint_index_endeffector_base+4)
        # Propagate main finger joint positions and velocities to the mimic joints
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+2, p.POSITION_CONTROL, 
                                targetPosition=finger_left[0], 
                                targetVelocity=finger_left[1],
                                positionGain=1.2)
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+3, p.POSITION_CONTROL, 
                                targetPosition=finger_left[0], 
                                targetVelocity=finger_left[1],
                                positionGain=1.2)
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+5, p.POSITION_CONTROL, 
                                targetPosition=finger_right[0], 
                                targetVelocity=finger_right[1],
                                positionGain=1.2)
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+6, p.POSITION_CONTROL, 
                                targetPosition=finger_right[0], 
                                targetVelocity=finger_right[1],
                                positionGain=1.2)

    def apply_action(self, action):
        # Ensure action is a numpy array
        action = np.array(action)

        # Clip the action to be within -1 and 1
        action = np.clip(action, -1, 1)

        # Define the maximum change in joint angles per time step
        max_delta = 0.05  # Adjust this value as needed

        # Compute the desired change in joint angles
        delta_angles = action * max_delta

        # Update the current pose
        self.current_pose[self.joint_indices_arm] += delta_angles

        # Clip the updated joint angles to joint limits
        lower_limits = self.joint_limits[0][self.joint_indices_arm]
        upper_limits = self.joint_limits[1][self.joint_indices_arm]
        self.current_pose[self.joint_indices_arm] = np.clip(
            self.current_pose[self.joint_indices_arm],
            lower_limits,
            upper_limits
        )

        # Apply the joint positions using POSITION_CONTROL
        for idx, joint_idx in enumerate(self.joint_indices_arm):
            p.setJointMotorControl2(
                bodyIndex=self.uid,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.current_pose[joint_idx],
                positionGain=0.2  # Adjust gain as needed
            )

# Define the ThrowBallEnv class
class ThrowBallEnv(gym.Env):
    def __init__(self):
        super(ThrowBallEnv, self).__init__()
        
        # Connect to PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # load the plane
        self.plane_uid = p.loadURDF("plane.urdf")
        
        # Load the robot
        UR5_URDF_PATH = "./urdf/ur5_rg2.urdf"  # Update this path to your URDF file
        self.robot_uid = p.loadURDF(UR5_URDF_PATH, useFixedBase=True)
        self.robot = Robot(self.robot_uid, control_dt=0.01)
        
        # Initialize the ball and bucket
        self.ball_start_pos = [0.5, 0, 0.5]
        self.bucket_pos = [1.5, 0, 0]  # Adjust as needed
        
        # Create the ball
        self._create_ball()
        
        # Create the bucket
        self.create_bucket()
        
        # Define action and observation spaces
        num_joints = len(self.robot.joint_indices_arm)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_joints,), dtype=np.float32)
        
        # Observations: joint positions, joint velocities, end-effector position, ball position and velocity
        obs_dim = num_joints * 2 + 3 + 6  # joints, end-effector pos, ball pos and vel
        obs_high = np.ones(obs_dim) * np.finfo(np.float32).max
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        
        self.time_step = 0
        self.max_steps = 200

    def _create_ball(self):
        ball_radius = 0.05  # Adjust as needed
        ball_mass = 0.1  # Adjust as needed

        # Create the ball using a sphere shape
        ball_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        ball_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 0, 0, 1])

        # Create the ball as a multi-body
        self.ball_uid = p.createMultiBody(
            baseMass=ball_mass,
            baseCollisionShapeIndex=ball_collision_shape,
            baseVisualShapeIndex=ball_visual_shape,
            basePosition=self.ball_start_pos
        )

    def create_bucket(self):
        # Define bucket parameters
        bucket_radius = 0.2
        bucket_height = 0.3
        wall_thickness = 0.01

        # Positions and orientations for the bucket walls
        wall_positions = [
            [self.bucket_pos[0], self.bucket_pos[1] + bucket_radius - wall_thickness / 2, self.bucket_pos[2] + bucket_height / 2],
            [self.bucket_pos[0], self.bucket_pos[1] - bucket_radius + wall_thickness / 2, self.bucket_pos[2] + bucket_height / 2],
            [self.bucket_pos[0] + bucket_radius - wall_thickness / 2, self.bucket_pos[1], self.bucket_pos[2] + bucket_height / 2],
            [self.bucket_pos[0] - bucket_radius + wall_thickness / 2, self.bucket_pos[1], self.bucket_pos[2] + bucket_height / 2],
        ]
        wall_orientations = [
            p.getQuaternionFromEuler([0, 0, 0]),
            p.getQuaternionFromEuler([0, 0, 0]),
            p.getQuaternionFromEuler([0, 0, np.pi / 2]),
            p.getQuaternionFromEuler([0, 0, np.pi / 2]),
        ]

        # Create walls
        for pos, orn in zip(wall_positions, wall_orientations):
            wall_collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[bucket_radius, wall_thickness, bucket_height / 2]
            )
            wall_visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[bucket_radius, wall_thickness, bucket_height / 2],
                rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_collision_shape,
                baseVisualShapeIndex=wall_visual_shape,
                basePosition=pos,
                baseOrientation=orn
            )

        # Create the bottom of the bucket
        bottom_collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[bucket_radius, bucket_radius, wall_thickness]
        )
        bottom_visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[bucket_radius, bucket_radius, wall_thickness],
            rgbaColor=[0.5, 0.5, 0.5, 1]
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=bottom_collision_shape,
            baseVisualShapeIndex=bottom_visual_shape,
            basePosition=[self.bucket_pos[0], self.bucket_pos[1], self.bucket_pos[2] + wall_thickness]
        )

    def reset(self, *, seed=None, options=None):
        # Set the seed if provided
        super().reset(seed=seed)
        
        # Reset the simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Reload the robot
        UR5_URDF_PATH = "./urdf/ur5_rg2.urdf"  # Update this path to your URDF file
        self.robot_uid = p.loadURDF(UR5_URDF_PATH, useFixedBase=True)
        self.robot = Robot(self.robot_uid, control_dt=0.01)
        self.robot.reset()
        
        # Reload the ball
        self._create_ball()

        # Recreate the bucket
        self.create_bucket()
        
        self.time_step = 0

        # Attach the ball to the robot's end-effector
        self.constraint_uid = p.createConstraint(
            parentBodyUniqueId=self.robot_uid,
            parentLinkIndex=self.robot.joint_indices_arm[-1],
            childBodyUniqueId=self.ball_uid,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )

        # Get initial observation
        observation = self._get_observation()
        info = {}
        return observation, info


    def step(self, action):
        # Apply action to robot
        self.robot.apply_action(action)
        
        # Release the ball after a certain condition (e.g., time step)
        if self.time_step == 50:
            p.removeConstraint(self.constraint_uid)
        
        # Step simulation
        for _ in range(int(self.robot.control_dt / p.getPhysicsEngineParameters()['fixedTimeStep'])):
            p.stepSimulation()
            time.sleep(1./240.)  # Match simulation step time
        
        self.time_step += 1
        
        # Get observation
        observation = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check if episode is terminated or truncated
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        info = {}
        
        return observation, reward, terminated, truncated, info

    # def _check_terminated(self):
    #     # Termination due to success or failure
    #     if self._ball_in_bucket():
    #         return True
    #     elif self._ball_on_ground():
    #         return True
    #     else:
    #         return False

    # def _check_truncated(self):
    #     # Truncation due to exceeding max steps
    #     if self.time_step >= self.max_steps:
    #         return True
    #     else:
    #         return False

    def _get_observation(self):
        # Get joint states
        joint_states = p.getJointStates(self.robot_uid, self.robot.joint_indices_arm)
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
        joint_velocities = np.array([state[1] for state in joint_states], dtype=np.float32)

        # Get end-effector position
        end_effector_state = p.getLinkState(self.robot_uid, self.robot.joint_indices_arm[-1])
        end_effector_pos = np.array(end_effector_state[4], dtype=np.float32)

        # Get ball position and velocity
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_uid)
        ball_vel, _ = p.getBaseVelocity(self.ball_uid)
        ball_state = np.array(ball_pos + ball_vel, dtype=np.float32)

        # Concatenate all observations
        observation = np.concatenate([joint_positions, joint_velocities, end_effector_pos, ball_state])
        return observation

    def _compute_reward(self):
        # Get ball and bucket positions
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_uid)
        bucket_pos = np.array(self.bucket_pos)

        # Parameters for reward shaping
        distance_weight = -1.0
        success_reward = 100.0
        time_penalty = -0.01
        control_penalty_weight = -0.1

        # Calculate horizontal distance to bucket
        horizontal_distance = np.linalg.norm(np.array(ball_pos[:2]) - bucket_pos[:2])

        # Reward for minimizing distance to bucket
        distance_reward = distance_weight * horizontal_distance

        # Success reward if ball is in the bucket
        if self._ball_in_bucket():
            success = success_reward
        else:
            success = 0.0

        # Penalty for time steps to encourage faster completion
        time_penalty = time_penalty

        # Control penalty to discourage excessive joint movements
        joint_velocities = np.array([state[1] for state in p.getJointStates(self.robot_uid, self.robot.joint_indices_arm)])
        control_penalty = control_penalty_weight * np.linalg.norm(joint_velocities)

        # Total reward
        reward = distance_reward + success + time_penalty + control_penalty

        return reward

#------------------------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    def _ball_in_bucket(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_uid)
        bucket_pos = np.array(self.bucket_pos)

        # Define bucket dimensions (assuming known)
        bucket_radius = 0.2
        bucket_height = 0.3

        # Check if ball is within bucket boundaries
        horizontal_distance = np.linalg.norm(np.array(ball_pos[:2]) - bucket_pos[:2])
        vertical_distance = ball_pos[2] - bucket_pos[2]

        if horizontal_distance < bucket_radius and 0 < vertical_distance < bucket_height:
            return True
        else:
            return False

    def _ball_on_ground(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_uid)
        if ball_pos[2] <= 0.0:
            return True
        else:
            return False

    def _check_done(self):
        # Check if ball is in bucket or on ground, or if max steps exceeded
        if self._ball_in_bucket() or self._ball_on_ground() or self.time_step >= self.max_steps:
            return True
        else:
            return False
        
    def _check_terminated(self):
    # 如果球进入桶中，则成功，episode 终止
        if self._ball_in_bucket():
            return True
        # 如果球掉到地面上，则失败，episode 终止
        elif self._ball_on_ground():
            return True
        else:
            return False

    def _check_truncated(self):
    # 如果达到最大步数，则截断
        if self.time_step >= self.max_steps:
            return True
        else:
            return False

    def render(self):
        pass  # Rendering is handled by PyBullet's GUI

    def close(self):
        p.disconnect()

#     # 创建自定义回调函数
# class EpisodeLimitCallback(BaseCallback):
#     def __init__(self, max_episodes, verbose=0):
#         super(EpisodeLimitCallback, self).__init__(verbose)
#         self.max_episodes = max_episodes
#         self.num_episodes = 0

#     def _on_step(self) -> bool:
#         # 获取当前时间步的 dones 信息
#         dones = self.locals.get('dones')
#         if dones is not None:
#             # 累计完成的 episode 数量
#             self.num_episodes += sum(dones)
#             if self.verbose > 0:
#                 print(f"Total episodes: {self.num_episodes}")
#             # 如果达到最大 episode 数量，停止训练
#             if self.num_episodes >= self.max_episodes:
#                 print(f"Reached maximum episodes of {self.max_episodes}. Training stopped.")
#                 return False
#         return True


# Main code to test or train the environment
# if __name__ == "__main__":
#     # Create the environment
#     env = ThrowBallEnv()

#     # Check the environment
#     check_env(env)

#     # Uncomment the following lines to train the agent using PPO
#     """
#     # Set up the PPO model
#     model = PPO('MlpPolicy', env, verbose=1)

#     # Train the model
#     model.learn(total_timesteps=100000)

#     # Save the trained model
#     model.save("ppo_throw_ball")
#     """

#     # Test the environment with random actions or the trained model
#     obs = env.reset()
#     done = False

#     while not done:
#         # For testing with random actions
#         action = env.action_space.sample()

#         # Uncomment the following lines to test with the trained model
#         """
#         # Load the trained model
#         model = PPO.load("ppo_throw_ball")

#         # Get action from the model
#         action, _states = model.predict(obs, deterministic=True)
#         """

#         observation, reward, terminated, truncated, info = env.step(action)
#         env.render()
#         time.sleep(1./240.)  # Adjust sleep time as needed

#     env.close()

if __name__ == "__main__":
    env = ThrowBallEnv()
    obs, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        time.sleep(1./240.)

    env.close()
