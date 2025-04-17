import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RoboticArmEnv:
    def __init__(self, arm_lengths=[1.0, 0.8], target_pos=None):
        self.arm_lengths = np.array(arm_lengths)
        self.num_joints = len(arm_lengths)
        self.angle_step = 0.1
        self.max_angle = np.pi
        self.min_angle = -np.pi
        self.current_angles = np.zeros(self.num_joints)
        self.target_pos = np.array([1.0, 0.5]) if target_pos is None else np.array(target_pos)
        self.max_distance = sum(self.arm_lengths) * 2
        
    def reset(self):
        self.current_angles = np.random.uniform(self.min_angle, self.max_angle, self.num_joints)
        return self._get_state()
    
    def step(self, action):
        angle_changes = np.zeros(self.num_joints)
        for i in range(self.num_joints):
            if action == i*2:
                angle_changes[i] = self.angle_step
            elif action == i*2 + 1:
                angle_changes[i] = -self.angle_step
        
        new_angles = self.current_angles + angle_changes
        new_angles = np.clip(new_angles, self.min_angle, self.max_angle)
        self.current_angles = new_angles
        
        end_pos = self._forward_kinematics(self.current_angles)
        distance = np.linalg.norm(end_pos - self.target_pos)
        
        reward = -distance
        done = distance < 0.1
        if done:
            reward += 10
            
        return self._get_state(), reward, done
    
    def _get_state(self):
        state = tuple(np.round(self.current_angles / 0.2).astype(int))
        return state
    
    def _forward_kinematics(self, angles):
        pos = np.zeros(2)
        angle_sum = 0
        
        for i in range(self.num_joints):
            angle_sum += angles[i]
            pos[0] += self.arm_lengths[i] * np.cos(angle_sum)
            pos[1] += self.arm_lengths[i] * np.sin(angle_sum)
            
        return pos
    
    def render(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        pos = np.zeros(2)
        angle_sum = 0
        points = [pos.copy()]
        
        for i in range(self.num_joints):
            angle_sum += self.current_angles[i]
            pos[0] += self.arm_lengths[i] * np.cos(angle_sum)
            pos[1] += self.arm_lengths[i] * np.sin(angle_sum)
            points.append(pos.copy())
        
        points = np.array(points)
        ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=3)
        ax.plot(points[:, 0], points[:, 1], 'ro', markersize=10)
        ax.plot(self.target_pos[0], self.target_pos[1], 'g*', markersize=15)
        
        ax.set_xlim([-sum(self.arm_lengths), sum(self.arm_lengths)])
        ax.set_ylim([-sum(self.arm_lengths), sum(self.arm_lengths)])
        ax.grid(True)
        plt.title('Robotic Arm')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

class MonteCarloRL:
    def __init__(self, env, epsilon=0.1, gamma=0.9):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_actions = env.num_joints * 2
        self.Q = {}
        self.returns = {}
        self.policy = {}
        
    def get_action(self, state, explore=True):
        if state not in self.policy:
            self.policy[state] = np.ones(self.num_actions) / self.num_actions
            self.Q[state] = np.zeros(self.num_actions)
        
        if explore and np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.Q[state])
    
    def generate_episode(self, max_steps=200, explore=True):
        episode = []
        state = self.env.reset()
        for _ in range(max_steps):
            action = self.get_action(state, explore)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode
    
    def update_policy(self, state):
        best_action = np.argmax(self.Q[state])
        self.policy[state] = np.eye(self.num_actions)[best_action]
    
    def mc_prediction(self, num_episodes=1000):
        for _ in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            visited_state_actions = set()
            
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                if (state, action) not in visited_state_actions:
                    visited_state_actions.add((state, action))
                    
                    if (state, action) not in self.returns:
                        self.returns[(state, action)] = []
                    
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.returns[(state, action)])
    
    def mc_control(self, num_episodes=5000):
        for i in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            visited_state_actions = set()
            
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                if (state, action) not in visited_state_actions:
                    visited_state_actions.add((state, action))
                    
                    if (state, action) not in self.returns:
                        self.returns[(state, action)] = []
                    
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.returns[(state, action)])
                    self.update_policy(state)
            
            if (i+1) % 500 == 0:
                print(f"Episode {i+1}/{num_episodes}")
                self.test_policy()
    
    def test_policy(self, num_tests=10, max_steps=100, render_final=False):
        total_rewards = 0
        success_count = 0
        
        for _ in range(num_tests):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            for step in range(max_steps):
                action = self.get_action(state, explore=False)
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    success_count += 1
                    break
            
            total_rewards += episode_reward
        
        if render_final:
            self.env.render()
            
        avg_reward = total_rewards / num_tests
        success_rate = success_count / num_tests
        print(f"Average reward: {avg_reward:.2f}, Success rate: {success_rate:.2f}")
        
        return avg_reward, success_rate

def main():
    arm_env = RoboticArmEnv(arm_lengths=[1.0, 0.8], target_pos=[0.8, 0.6])
    mc_agent = MonteCarloRL(arm_env, epsilon=0.2, gamma=0.95)
    
    print("Training Monte Carlo agent...")
    mc_agent.mc_control(num_episodes=5000)
    
    print("\nFinal evaluation:")
    mc_agent.test_policy(num_tests=10, render_final=True)

if __name__ == "__main__":
    main()