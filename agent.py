import numpy as np
from task import Task
from agents.actor import Actor
from agents.critic import Critic
from noise import Noise
from memory import ReplayBuffer

class Christophers_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range
        
        self.actor = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.critic = Critic(self.state_size, self.action_size)
        
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        self.gamma = 0.95
        self.tau = 0.001

        self.best_w = None
        self.best_score = -np.inf

        self.exploration_mu = 0.5
        self.exploration_theta = 0.2
        self.exploration_sigma = 0.4
        self.noise = Noise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        
        self.buffer_size = 100000
        self.batch_size = 32
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        self.best_score = -np.inf
        self.num_steps = 0

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        if self.get_score() > self.best_score:
            self.best_score = self.get_score()
        self.total_reward = 0.0
        self.num_steps = 0
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        self.total_reward += reward
        self.num_steps += 1
        
        self.memory.add(self.last_state, action, reward, next_state, done)
        
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
  
        self.last_state = next_state

    def act(self, state):
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor.model.predict(state)[0]
        action = list(action + self.noise.sample())  # add some noise for exploration
        return action
    
    def get_score(self):
        return -np.inf if self.num_steps == 0 else self.total_reward / self.num_steps

    def learn(self, experiences):
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        done = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
    
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - done)

        self.critic.model.train_on_batch(x=[states, actions], y=Q_targets)
        
        action_gradients = np.reshape(self.critic.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor.train_fn([states, action_gradients, 1])
        
        self.soft_update(self.critic.model, self.critic_target.model)
        self.soft_update(self.actor.model, self.actor_target.model)
        
    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights)

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        
        
        
        
        
        