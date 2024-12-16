import os
import pybullet as p
import pybullet_data
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import matplotlib.pyplot as plt

def normalized_angle(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def get_joint_limits(body_id, joint_ids):
    joint_limits = []
    for joint_id in joint_ids:
        joint_info = p.getJointInfo(body_id, joint_id)
        joint_limit = joint_info[8], joint_info[9]
        joint_limits.append(joint_limit)
    joint_limits = np.transpose(joint_limits)  # Dx2 -> 2xD
    return joint_limits

class Robot:
    def __init__(self, uid, control_dt):
        self.uid = uid
        self.control_dt = control_dt

        self.joint_index_last             = 13
        self.joint_index_endeffector_base = 7
        self.joint_indices_arm:np.ndarray = np.array(range(1, self.joint_index_endeffector_base))
        self.joint_limits:np.ndarray      = get_joint_limits(uid, range(self.joint_index_last+1))
        self.joint_range:np.ndarray       = (np.array(self.joint_limits[1]) - np.array(self.joint_limits[0]))

        # 修改这里：将末端手指的初始位置改小，使夹爪半闭合
        self.rest_pose:np.ndarray = np.array((
            0.0,
            0.0,
            -2.094,
            1.57,
            -1.047,
            -1.57,
            0.0,
            0.0,   # 手指相关关节初始值
            0.3    # 原本是0.785，改为0.3使夹爪微微闭合
        ))

        self.reset()

    def reset(self):
        self.current_pose:np.ndarray = np.copy(self.rest_pose)
        for i in self.joint_indices_arm:
            p.resetJointState(self.uid, i, self.current_pose[i])
        self._finger_control(self.current_pose[self.joint_index_endeffector_base+1])

    def _finger_control(self, target):
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+1, p.POSITION_CONTROL, 
                                targetPosition=target)
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+4, p.POSITION_CONTROL, 
                                targetPosition=target)
        finger_left = p.getJointState(self.uid, self.joint_index_endeffector_base+1)
        finger_right = p.getJointState(self.uid, self.joint_index_endeffector_base+4)
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

class ThrowingEnv:
    def __init__(self, render=False):
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)

        self._load_models()

        self.action_space_shape = (6,)
        obs_dim = 6 + 3 + 3  
        self.observation_space_shape = (obs_dim,)

        self.max_steps = 200
        self.current_step = 0

    def _load_models(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb_data_path = pybullet_data.getDataPath()
        project_path = os.path.dirname(os.path.abspath(__file__))

        self.robot_uid = p.loadURDF(os.path.join(project_path, "urdf/ur5_rg2.urdf"), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        self.table_uid = p.loadURDF(os.path.join(pb_data_path, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
        bucket_distance = random.uniform(0.8, 1.2)
        self.bucket_uid = p.loadURDF(os.path.join(pb_data_path, "tray/traybox.urdf"), basePosition=[bucket_distance, 0, 0])
        
        # 将球放在略低于末端夹爪高度且在爪子范围内
        self.object_uid = p.loadURDF("sphere2.urdf", basePosition=[0.5, 0, 0.12], globalScaling=0.1)

        self.bucket_pos = p.getBasePositionAndOrientation(self.bucket_uid)[0]
        self.robot = Robot(self.robot_uid, 1./240.)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(4./240.)
        self._load_models()

        # 执行几步仿真，让球和夹爪之间的接触稳定
        for _ in range(50):
            p.stepSimulation()

        self.current_step = 0
        obs = self._get_observation()
        return obs

    def step(self, action):
        self._apply_action(action)
        for _ in range(4):
            p.stepSimulation()
            time.sleep(1./240.) # 若不需要可视化可注释掉
        self.current_step += 1

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._is_done()
        info = {}
        return obs, reward, done, info

    def close(self):
        p.disconnect()

    def _apply_action(self, action):
        max_joint_velocity = 1.0
        scaled_action = action * max_joint_velocity
        p.setJointMotorControlArray(self.robot.uid, self.robot.joint_indices_arm,
                                    p.VELOCITY_CONTROL,
                                    targetVelocities=scaled_action)

    def _get_observation(self):
        joint_states = p.getJointStates(self.robot.uid, self.robot.joint_indices_arm)
        joint_positions = [state[0] for state in joint_states]
        object_pos = p.getBasePositionAndOrientation(self.object_uid)[0]
        bucket_pos = self.bucket_pos
        obs = np.array(joint_positions + list(object_pos) + list(bucket_pos), dtype=np.float32)
        return obs

    def _compute_reward(self):
        object_pos = p.getBasePositionAndOrientation(self.object_uid)[0]
        bucket_pos = self.bucket_pos
        distance = np.linalg.norm(np.array(object_pos[:2]) - np.array(bucket_pos[:2]))
        if self._is_object_in_bucket():
            reward = 100.0
        else:
            reward = -distance
        reward -= 0.1
        return reward

    def _is_object_in_bucket(self):
        object_pos = p.getBasePositionAndOrientation(self.object_uid)[0]
        bucket_pos = self.bucket_pos
        distance = np.linalg.norm(np.array(object_pos[:2]) - np.array(bucket_pos[:2]))
        if distance < 0.1 and object_pos[2] < bucket_pos[2] + 0.2:
            return True
        return False

    def _is_done(self):
        if self.current_step >= self.max_steps or self._is_object_in_bucket():
            return True
        return False

    def render(self):
        pass

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        self.policy_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )
        self.value_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        shared = self.shared_layers(x)
        policy = self.policy_layers(shared)
        value = self.value_layers(shared)
        return policy, value


#-----------------------------------------------------------------------    
class PPOAgent:
    def __init__(self, obs_dim, action_dim):
        self.actor_critic = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=1e-4)
        self.clip_param = 0.2
        self.num_epochs = 10
        self.batch_size = 64
        self.gamma = 0.99
        self.lam = 0.95

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_mean, _ = self.actor_critic(state)
        action = action_mean.numpy()
        action = np.clip(action + np.random.normal(0, 0.1, size=action.shape), -1, 1)
        return action

    def update(self, trajectories):
        states = torch.tensor(np.array([t[0] for t in trajectories]), dtype=torch.float32)
        actions = torch.tensor(np.array([t[1] for t in trajectories]), dtype=torch.float32)
        rewards = np.array([t[2] for t in trajectories], dtype=np.float32)
        dones = np.array([t[3] for t in trajectories], dtype=np.float32)
        next_states = torch.tensor(np.array([t[4] for t in trajectories]), dtype=torch.float32)

        _, values = self.actor_critic(states)
        values = values.detach().numpy().flatten()

        _, next_values = self.actor_critic(next_states)
        next_values = next_values.detach().numpy().flatten()
        values = np.append(values, next_values[-1])

        advantages, returns = self.compute_gae(rewards, values, dones)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        old_actions = actions.clone().detach()
        old_states = states.clone().detach()

        for _ in range(self.num_epochs):
            perm = np.random.permutation(len(states))
            for i in range(0, len(states), self.batch_size):
                idx = perm[i:i+self.batch_size]
                sampled_states = states[idx]
                sampled_actions = actions[idx]
                sampled_advantages = advantages[idx]
                sampled_returns = returns[idx]
                sampled_old_actions = old_actions[idx]

                action_means, values_ = self.actor_critic(sampled_states)
                values_ = values_.squeeze()
                dist = torch.distributions.Normal(action_means, torch.tensor(0.1))
                log_probs = dist.log_prob(sampled_actions).sum(axis=-1)
                old_dist = torch.distributions.Normal(sampled_old_actions, torch.tensor(0.1))
                old_log_probs = old_dist.log_prob(sampled_actions).sum(axis=-1).detach()
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * sampled_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * sampled_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values_, sampled_returns)
                loss = policy_loss + 0.5 * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages[i] = gae
        returns = advantages + values[:-1]
        return advantages, returns

def train():
    env = ThrowingEnv(render=True)
    obs_dim = env.observation_space_shape[0]
    action_dim = env.action_space_shape[0]
    agent = PPOAgent(obs_dim, action_dim)

    num_episodes = 1000
    max_timesteps = 1000

    rewards_history = []

    for episode in range(num_episodes):
        state = env.reset()
        trajectories = []
        total_reward = 0
        for t in range(max_timesteps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            trajectories.append((state, action, reward, done, next_state))
            state = next_state
            total_reward += reward
            if done:
                break
        agent.update(trajectories)
        rewards_history.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}")
        if episode % 10 == 0:
            torch.save(agent.actor_critic.state_dict(), f"model_episode_{episode}.pth")

    env.close()

    plt.figure(figsize=(10,5))
    plt.plot(rewards_history, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()

if __name__ == "__main__":
    train()
