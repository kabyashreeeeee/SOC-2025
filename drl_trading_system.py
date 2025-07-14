# Deep Reinforcement Learning Trading System - FIXED VERSION
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Deep Learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
import json
import os
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")

print("Deep Reinforcement Learning Trading System - FIXED VERSION")
print("=" * 60)

# ============================================================================
# 1. DATA COLLECTION AND PREPROCESSING (FIXED)
# ============================================================================

class DataPipeline:
    def __init__(self, assets=['AAPL', 'MSFT', 'GOOGL', 'TSLA'], period='3y'):
        self.assets = assets
        self.period = period
        self.raw_data = {}
        self.processed_data = {}
        self.scalers = {}

    def fetch_data(self):
        """Fetch historical data for all assets"""
        print("Fetching historical data...")
        for symbol in self.assets:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=self.period)
                if not df.empty and len(df) >= 500:
                    self.raw_data[symbol] = df
                    print(f"✓ {symbol}: {len(df)} records")
                else:
                    print(f"✗ Insufficient data for {symbol}")
            except Exception as e:
                print(f"✗ Error fetching {symbol}: {str(e)}")

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators with proper NaN handling"""
        data = df.copy()
        
        # Ensure we have enough data
        if len(data) < 60:
            print("Warning: Not enough data for technical indicators")
            return data

        # Moving Averages
        data['SMA_5'] = data['Close'].rolling(window=5, min_periods=1).mean()
        data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

        # RSI with proper handling
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
            rs = gain / (loss + 1e-8)  # Avoid division by zero
            return 100 - (100 / (1 + rs))

        data['RSI'] = calculate_rsi(data['Close'])

        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_hist'] = data['MACD'] - data['MACD_signal']

        # Bollinger Bands
        data['BB_middle'] = data['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = data['Close'].rolling(window=20, min_periods=1).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        data['BB_width'] = data['BB_upper'] - data['BB_lower']
        
        # Avoid division by zero in BB position
        bb_range = data['BB_upper'] - data['BB_lower']
        data['BB_position'] = np.where(bb_range > 0, 
                                     (data['Close'] - data['BB_lower']) / bb_range, 
                                     0.5)

        # ATR (Average True Range)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['ATR'] = true_range.rolling(14, min_periods=1).mean()

        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20, min_periods=1).mean()
        data['Volume_ratio'] = data['Volume'] / (data['Volume_SMA'] + 1e-8)

        # Price-based features
        data['Price_change'] = data['Close'].pct_change()
        data['High_Low_ratio'] = data['High'] / (data['Low'] + 1e-8)
        data['Close_Open_ratio'] = data['Close'] / (data['Open'] + 1e-8)

        # Momentum indicators
        data['momentum_5'] = data['Close'].pct_change(5)
        data['momentum_10'] = data['Close'].pct_change(10)

        # Fill any remaining NaN values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Replace any infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(0)

        return data

    def preprocess_data(self, df):
        """Preprocess and normalize data with better handling"""
        # Remove first 30 rows to avoid NaN issues from indicators
        df_clean = df.iloc[30:].copy()
        
        if len(df_clean) < 100:
            print("Warning: Not enough data after cleaning")
            return df_clean, df_clean, None, None

        # Separate features
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df_clean.columns if col not in price_cols]

        # Scale features
        scaler = StandardScaler()
        
        # Only scale feature columns
        df_normalized = df_clean.copy()
        if len(feature_cols) > 0:
            df_normalized[feature_cols] = scaler.fit_transform(df_clean[feature_cols])

        # Clip extreme values
        df_normalized = df_normalized.clip(-5, 5)

        return df_normalized, df_clean, scaler, None

    def process_all_assets(self):
        """Process all assets with feature engineering"""
        print("Processing assets with feature engineering...")

        for symbol, df in self.raw_data.items():
            print(f"Processing {symbol}...")

            # Calculate technical indicators
            engineered_df = self.calculate_technical_indicators(df)

            # Preprocess data
            normalized, clean, scaler, _ = self.preprocess_data(engineered_df)

            self.processed_data[symbol] = {
                'normalized': normalized,
                'clean': clean,
                'raw': df
            }

            self.scalers[symbol] = {
                'scaler': scaler
            }

            print(f"✓ {symbol}: {len(clean)} samples, {len(clean.columns)} features")

# ============================================================================
# 2. IMPROVED TRADING ENVIRONMENT (FIXED)
# ============================================================================

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001, lookback_window=10):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.max_steps = len(data) - 1
        
        # State features (excluding OHLCV for state representation) - FIXED: Initialize before reset
        self.feature_columns = [col for col in data.columns
                               if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = 3
        self.observation_space = len(self.feature_columns) * lookback_window + 4  # +4 for portfolio state
        
        # Initialize current step
        self.current_step = lookback_window
        
        # Initialize portfolio
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        self.portfolio_history = []
        
        return self._get_observation()

    def _get_observation(self):
        """Get current observation state with lookback window"""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space)

        # Get lookback window of technical indicators
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        tech_features = []
        for i in range(start_idx, end_idx):
            if i < len(self.data):
                row_features = self.data[self.feature_columns].iloc[i].values
                row_features = np.nan_to_num(row_features, nan=0.0)
                tech_features.extend(row_features)
        
        # Pad if necessary
        expected_length = len(self.feature_columns) * self.lookback_window
        while len(tech_features) < expected_length:
            tech_features.extend([0.0] * len(self.feature_columns))
        
        tech_features = np.array(tech_features[:expected_length])

        # Portfolio state
        current_price = self.data['Close'].iloc[self.current_step]
        portfolio_value = self.balance + self.shares_held * current_price
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held * current_price / self.initial_balance,  # Normalized holdings
            portfolio_value / self.initial_balance,  # Normalized total value
            self.shares_held / 100.0  # Normalized shares held
        ])

        return np.concatenate([tech_features, portfolio_state])

    def step(self, action):
        """Execute action and return next state, reward, done"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, {}

        current_price = self.data['Close'].iloc[self.current_step]
        prev_net_worth = self.net_worth
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > current_price * (1 + self.transaction_cost):
                shares_to_buy = int(self.balance / (current_price * (1 + self.transaction_cost)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    self.balance -= cost
                    self.shares_held += shares_to_buy
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost
                    })

        elif action == 2:  # Sell
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.trades.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'shares': self.shares_held,
                    'price': current_price,
                    'revenue': revenue
                })
                self.shares_held = 0

        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1

        # Calculate new net worth
        new_price = self.data['Close'].iloc[self.current_step]
        self.net_worth = self.balance + self.shares_held * new_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Calculate reward - combination of return and risk adjustment
        portfolio_return = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
        
        # Scale reward to be more meaningful
        reward = portfolio_return * 100  # Scale up the reward
        
        # Add small penalty for holding cash (encourage trading)
        if action == 0 and self.balance > self.initial_balance * 0.9:
            reward -= 0.01

        # Track portfolio history
        self.portfolio_history.append({
            'step': self.current_step,
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'price': new_price
        })

        # Check if done
        done = self.current_step >= self.max_steps - 1

        return self._get_observation(), reward, done, {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_return': portfolio_return
        }

    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if len(self.portfolio_history) < 2:
            return {
                'total_return': 0,
                'num_trades': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'final_balance': self.balance,
                'final_shares': self.shares_held,
                'net_worth': self.net_worth
            }

        total_return = (self.net_worth - self.initial_balance) / self.initial_balance

        # Calculate daily returns
        returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_val = self.portfolio_history[i-1]['net_worth']
            curr_val = self.portfolio_history[i]['net_worth']
            if prev_val > 0:
                returns.append((curr_val - prev_val) / prev_val)

        if len(returns) > 1:
            returns = np.array(returns)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Max drawdown
        portfolio_values = [p['net_worth'] for p in self.portfolio_history]
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (running_max - portfolio_values) / running_max
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        return {
            'total_return': total_return,
            'num_trades': len(self.trades),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_balance': self.balance,
            'final_shares': self.shares_held,
            'net_worth': self.net_worth
        }

# ============================================================================
# 3. IMPROVED DRL AGENTS
# ============================================================================

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.95
        self.update_target_freq = 100
        self.learn_step = 0

        # Neural Network
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Update target network
        self.update_target_network()

    def _build_network(self):
        """Build DQN neural network"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.update_target_freq == 0:
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0003):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Actor-Critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.ppo_epochs = 4
        self.entropy_coef = 0.01
        self.gamma = 0.99

    def _build_actor(self):
        """Build actor network"""
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
            nn.Softmax(dim=-1)
        )

    def _build_critic(self):
        """Build critic network"""
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def act(self, state):
        """Choose action using policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            action_probs = torch.clamp(action_probs, 1e-8, 1.0)  # Prevent zero probabilities
            
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def evaluate(self, states, actions):
        """Evaluate state-action pairs"""
        action_probs = self.actor(states)
        action_probs = torch.clamp(action_probs, 1e-8, 1.0)
        
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        state_values = self.critic(states).squeeze()

        return action_log_probs, state_values, entropy

    def update(self, states, actions, rewards, log_probs, values, next_values):
        """Update policy using PPO"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_log_probs = torch.FloatTensor(log_probs)
        old_values = torch.FloatTensor(values)
        next_values = torch.FloatTensor(next_values)

        # Calculate returns and advantages
        returns = []
        advantages = []
        
        for i in range(len(rewards)):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = next_values[i]
            
            returns.append(rewards[i] + self.gamma * next_value)
            advantages.append(returns[i] - old_values[i])

        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.ppo_epochs):
            log_probs_new, values_new, entropy = self.evaluate(states, actions)

            # Actor loss
            ratio = torch.exp(log_probs_new - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            # Critic loss
            critic_loss = F.mse_loss(values_new, returns)

            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

# ============================================================================
# 4. TRAINING FRAMEWORK
# ============================================================================

class TradingTrainer:
    def __init__(self, data_pipeline, agent_type='DQN'):
        self.data_pipeline = data_pipeline
        self.agent_type = agent_type
        self.results = {}

    def train_agent(self, symbol, episodes=500):
        """Train agent on specific symbol"""
        print(f"\nTraining {self.agent_type} agent on {symbol}...")

        # Prepare data
        data = self.data_pipeline.processed_data[symbol]['normalized']
        
        if len(data) < 200:
            print(f"Not enough data for {symbol}")
            return None, None

        # Split data
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        # Create environment
        env = TradingEnvironment(train_data)

        # Initialize agent
        if self.agent_type == 'DQN':
            agent = DQNAgent(env.observation_space, env.action_space)
        elif self.agent_type == 'PPO':
            agent = PPOAgent(env.observation_space, env.action_space)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        # Training loop
        episode_rewards = []
        episode_returns = []
        best_return = -np.inf

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            # For PPO
            if self.agent_type == 'PPO':
                states, actions, rewards, log_probs, values = [], [], [], [], []

            while not done and step_count < 1000:  # Prevent infinite loops
                if self.agent_type == 'DQN':
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                    step_count += 1

                    if len(agent.memory) > 100:
                        agent.replay()

                elif self.agent_type == 'PPO':
                    action, log_prob = agent.act(state)
                    next_state, reward, done, info = env.step(action)

                    # Store transition
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    log_probs.append(log_prob)
                    
                    # Get value
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        value = agent.critic(state_tensor).item()
                    values.append(value)

                    state = next_state
                    episode_reward += reward
                    step_count += 1

            # Update PPO agent
            if self.agent_type == 'PPO' and len(states) > 10:
                next_values = values[1:] + [0]  # Bootstrap with 0 for terminal state
                agent.update(states, actions, rewards, log_probs, values, next_values)

            # Track progress
            episode_rewards.append(episode_reward)
            performance = env.get_performance_metrics()
            episode_returns.append(performance['total_return'])

            # Save best model
            if performance['total_return'] > best_return:
                best_return = performance['total_return']

            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_return = np.mean(episode_returns[-50:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Avg Return: {avg_return:.4f}")

        # Test on unseen data
        print(f"Testing {self.agent_type} agent on {symbol}...")
        test_env = TradingEnvironment(test_data)
        test_state = test_env.reset()
        test_done = False
        test_steps = 0

        while not test_done and test_steps < 1000:
            if self.agent_type == 'DQN':
                agent.epsilon = 0  # No exploration during testing
                test_action = agent.act(test_state)
            else:  # PPO
                test_action, _ = agent.act(test_state)

            test_state, _, test_done, _ = test_env.step(test_action)
            test_steps += 1

        # Store results
        test_performance = test_env.get_performance_metrics()
        self.results[symbol] = {
            'agent_type': self.agent_type,
            'training_rewards': episode_rewards,
            'training_returns': episode_returns,
            'test_performance': test_performance,
            'test_env': test_env,
            'agent': agent
        }

        print(f"✓ {symbol} - Test Return: {test_performance['total_return']:.4f}")
        return agent, test_performance

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    # Initialize data pipeline
    pipeline = DataPipeline()

    # Fetch and process data
    pipeline.fetch_data()
    if not pipeline.raw_data:
        print("No data fetched. Please check your internet connection.")
        return None, None
        
    pipeline.process_all_assets()

    # Train agents for each asset
    results = {}

    for agent_type in ['DQN', 'PPO']:
        print(f"\n{'='*60}")
        print(f"Training {agent_type} agents")
        print('='*60)

        trainer = TradingTrainer(pipeline, agent_type)

        for symbol in pipeline.assets:
            if symbol in pipeline.processed_data:
                try:
                    agent, performance = trainer.train_agent(symbol, episodes=200)
                    if agent and performance:
                        results[f"{agent_type}_{symbol}"] = {
                            'agent': agent,
                            'performance': performance,
                            'trainer': trainer
                        }
                except Exception as e:
                    print(f"Error training {agent_type} on {symbol}: {e}")

    # Display results summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print('='*60)

    for key, result in results.items():
        agent_type, symbol = key.split('_', 1)
        performance = result['performance']
        print(f"{agent_type} - {symbol}:")
        print(f"  Total Return: {performance['total_return']:.4f}")
        print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.4f}")
        print(f"  Number of Trades: {performance['num_trades']}")
        print(f"  Max Drawdown: {performance['max_drawdown']:.4f}")
        print(f"  Final Net Worth: ${performance['net_worth']:.2f}")
        print()

    # Find best performing agent
    if results:
        best_agent = max(results.items(), key=lambda x: x[1]['performance']['total_return'])
        print(f"Best performing agent: {best_agent[0]} with {best_agent[1]['performance']['total_return']:.4f} return")

    return pipeline, results

# ============================================================================
# 6. VISUALIZATION AND ANALYSIS
# ============================================================================

def plot_training_results(results):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (key, result) in enumerate(results.items()):
        if i >= 4:  # Only plot first 4 results
            break
            
        row = i // 2
        col = i % 2
        
        # Plot training rewards
        rewards = result['trainer'].results[key.split('_', 1)[1]]['training_rewards']
        axes[row, col].plot(rewards)
        axes[row, col].set_title(f'{key} - Training Rewards')
        axes[row, col].set_xlabel('Episode')
        axes[row, col].set_ylabel('Reward')
        axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_portfolio_performance(results):
    """Plot portfolio performance"""
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4*len(results)))
    
    if len(results) == 1:
        axes = [axes]
    
    for i, (key, result) in enumerate(results.items()):
        test_env = result['trainer'].results[key.split('_', 1)[1]]['test_env']
        
        # Get portfolio history
        portfolio_history = test_env.portfolio_history
        if portfolio_history:
            steps = [p['step'] for p in portfolio_history]
            net_worths = [p['net_worth'] for p in portfolio_history]
            
            axes[i].plot(steps, net_worths, label='Portfolio Value')
            axes[i].axhline(y=test_env.initial_balance, color='r', linestyle='--', label='Initial Balance')
            axes[i].set_title(f'{key} - Portfolio Performance')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Portfolio Value ($)')
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. BACKTESTING FRAMEWORK
# ============================================================================

class Backtester:
    def __init__(self, agent, data, initial_balance=10000):
        self.agent = agent
        self.data = data
        self.initial_balance = initial_balance
        
    def run_backtest(self):
        """Run comprehensive backtest"""
        env = TradingEnvironment(self.data, self.initial_balance)
        state = env.reset()
        done = False
        
        # Disable exploration for DQN
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = 0
        
        actions_taken = []
        portfolio_values = []
        
        while not done:
            if hasattr(self.agent, 'act'):
                if isinstance(self.agent, PPOAgent):
                    action, _ = self.agent.act(state)
                else:
                    action = self.agent.act(state)
            else:
                action = 0  # Default to hold
                
            state, reward, done, info = env.step(action)
            actions_taken.append(action)
            portfolio_values.append(info['net_worth'])
        
        return {
            'final_performance': env.get_performance_metrics(),
            'actions': actions_taken,
            'portfolio_values': portfolio_values,
            'trades': env.trades
        }

# ============================================================================
# 8. STRATEGY COMPARISON
# ============================================================================

def compare_strategies(results, pipeline):
    """Compare different strategies"""
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON")
    print('='*60)
    
    comparison_data = []
    
    for key, result in results.items():
        agent_type, symbol = key.split('_', 1)
        performance = result['performance']
        
        comparison_data.append({
            'Strategy': f"{agent_type}_{symbol}",
            'Agent': agent_type,
            'Symbol': symbol,
            'Total Return': performance['total_return'],
            'Sharpe Ratio': performance['sharpe_ratio'],
            'Max Drawdown': performance['max_drawdown'],
            'Trades': performance['num_trades'],
            'Final Value': performance['net_worth']
        })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Summary statistics
        print(f"\n{'='*40}")
        print("SUMMARY STATISTICS")
        print('='*40)
        
        print(f"Best Total Return: {comparison_df['Total Return'].max():.4f}")
        print(f"Best Sharpe Ratio: {comparison_df['Sharpe Ratio'].max():.4f}")
        print(f"Lowest Max Drawdown: {comparison_df['Max Drawdown'].min():.4f}")
        print(f"Average Return: {comparison_df['Total Return'].mean():.4f}")
        
        return comparison_df
    
    return None

# ============================================================================
# 9. RISK MANAGEMENT
# ============================================================================

class RiskManager:
    def __init__(self, max_position_size=0.1, stop_loss=0.05, take_profit=0.1):
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
    def apply_risk_rules(self, action, current_price, entry_price, portfolio_value):
        """Apply risk management rules"""
        if entry_price > 0:
            # Calculate unrealized P&L
            unrealized_pnl = (current_price - entry_price) / entry_price
            
            # Stop loss
            if unrealized_pnl < -self.stop_loss:
                return 2  # Force sell
            
            # Take profit
            if unrealized_pnl > self.take_profit:
                return 2  # Force sell
        
        return action

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        pipeline, results = main()
        
        if pipeline and results:
            print("\n" + "="*60)
            print("ADDITIONAL ANALYSIS")
            print("="*60)
            
            # Compare strategies
            comparison_df = compare_strategies(results, pipeline)
            
            # Plot results if matplotlib is available
            try:
                plot_training_results(results)
                plot_portfolio_performance(results)
            except Exception as e:
                print(f"Plotting error: {e}")
            
            # Run additional backtests
            print("\nRunning additional backtests...")
            for key, result in results.items():
                try:
                    agent = result['agent']
                    symbol = key.split('_', 1)[1]
                    data = pipeline.processed_data[symbol]['normalized']
                    
                    backtester = Backtester(agent, data)
                    backtest_results = backtester.run_backtest()
                    
                    print(f"\n{key} Backtest Results:")
                    print(f"  Final Return: {backtest_results['final_performance']['total_return']:.4f}")
                    print(f"  Total Trades: {len(backtest_results['trades'])}")
                    
                except Exception as e:
                    print(f"Backtest error for {key}: {e}")
            
            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Results stored in 'results' variable.")
            print("Pipeline stored in 'pipeline' variable.")
            
        else:
            print("Training failed. Please check the data and try again.")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
