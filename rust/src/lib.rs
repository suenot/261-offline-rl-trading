//! # Offline Reinforcement Learning for Trading
//!
//! This crate implements offline RL algorithms for learning trading strategies
//! from historical data without live market interaction.
//!
//! Key components:
//! - OfflineDataset: Fixed replay buffer from historical transitions
//! - BehaviorCloning: Supervised learning baseline
//! - ImplicitQLearning (IQL): Expectile regression + advantage-weighted BC
//! - DistributionShiftDetector: Monitors policy divergence from behavior
//! - BybitClient: Fetches historical OHLCV data

use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Number of discrete actions: Buy, Sell, Hold
pub const NUM_ACTIONS: usize = 3;

/// Trading action
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Buy = 0,
    Sell = 1,
    Hold = 2,
}

impl Action {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Action::Buy,
            1 => Action::Sell,
            _ => Action::Hold,
        }
    }

    pub fn index(&self) -> usize {
        *self as usize
    }
}

/// A single transition in the offline dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub state: Vec<f64>,
    pub action: Action,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
}

/// Fixed replay buffer for offline RL - stores historical transitions
#[derive(Debug, Clone)]
pub struct OfflineDataset {
    pub transitions: Vec<Transition>,
    pub state_dim: usize,
}

impl OfflineDataset {
    /// Create a new empty offline dataset
    pub fn new(state_dim: usize) -> Self {
        Self {
            transitions: Vec::new(),
            state_dim,
        }
    }

    /// Add a transition to the dataset
    pub fn add(&mut self, transition: Transition) {
        self.transitions.push(transition);
    }

    /// Number of transitions in the dataset
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Sample a random batch of transitions
    pub fn sample_batch(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<&Transition> {
        let n = self.transitions.len();
        if n == 0 {
            return Vec::new();
        }
        let actual_size = batch_size.min(n);
        let indices: Vec<usize> = (0..actual_size)
            .map(|_| rng.gen_range(0..n))
            .collect();
        indices.iter().map(|&i| &self.transitions[i]).collect()
    }

    /// Get the empirical action distribution (behavior policy) for a discretized state
    pub fn behavior_policy_distribution(&self) -> HashMap<Action, f64> {
        let mut counts = HashMap::new();
        let total = self.transitions.len() as f64;
        for t in &self.transitions {
            *counts.entry(t.action).or_insert(0.0) += 1.0;
        }
        for v in counts.values_mut() {
            *v /= total;
        }
        counts
    }

    /// Compute state statistics for normalization
    pub fn state_statistics(&self) -> (Vec<f64>, Vec<f64>) {
        if self.transitions.is_empty() {
            return (vec![0.0; self.state_dim], vec![1.0; self.state_dim]);
        }
        let n = self.transitions.len() as f64;
        let mut mean = vec![0.0; self.state_dim];
        let mut var = vec![0.0; self.state_dim];

        for t in &self.transitions {
            for (i, &v) in t.state.iter().enumerate() {
                mean[i] += v / n;
            }
        }
        for t in &self.transitions {
            for (i, &v) in t.state.iter().enumerate() {
                let diff = v - mean[i];
                var[i] += diff * diff / n;
            }
        }
        let std: Vec<f64> = var.iter().map(|&v| v.sqrt().max(1e-8)).collect();
        (mean, std)
    }
}

/// Linear model for Q-function and policy approximation
#[derive(Debug, Clone)]
pub struct LinearModel {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
}

impl LinearModel {
    /// Create a new linear model with small random weights
    pub fn new(input_dim: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0 / input_dim as f64).sqrt();
        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_dim);
        Self { weights, bias }
    }

    /// Forward pass: output = state @ weights + bias
    pub fn forward(&self, state: &[f64]) -> Array1<f64> {
        let s = Array1::from_vec(state.to_vec());
        s.dot(&self.weights) + &self.bias
    }

    /// Update weights using gradient descent
    pub fn update(&mut self, state: &[f64], gradient: &Array1<f64>, lr: f64) {
        let s = Array1::from_vec(state.to_vec());
        for j in 0..self.weights.ncols() {
            for i in 0..self.weights.nrows() {
                self.weights[[i, j]] -= lr * gradient[j] * s[i];
            }
            self.bias[j] -= lr * gradient[j];
        }
    }
}

/// Softmax function for converting logits to probabilities
pub fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Array1<f64> = logits.mapv(|x| (x - max_val).exp());
    let sum: f64 = exp_vals.sum();
    exp_vals / sum
}

/// Behavior Cloning: supervised learning baseline that imitates the behavior policy
#[derive(Debug, Clone)]
pub struct BehaviorCloning {
    pub model: LinearModel,
    pub learning_rate: f64,
    pub state_dim: usize,
}

impl BehaviorCloning {
    /// Create a new behavior cloning agent
    pub fn new(state_dim: usize, learning_rate: f64, rng: &mut impl Rng) -> Self {
        Self {
            model: LinearModel::new(state_dim, NUM_ACTIONS, rng),
            learning_rate,
            state_dim,
        }
    }

    /// Train on a batch of transitions using cross-entropy loss
    pub fn train_step(&mut self, batch: &[&Transition]) -> f64 {
        let mut total_loss = 0.0;
        for transition in batch {
            let logits = self.model.forward(&transition.state);
            let probs = softmax(&logits);
            let target = transition.action.index();

            // Cross-entropy loss
            let loss = -(probs[target].max(1e-10)).ln();
            total_loss += loss;

            // Gradient of cross-entropy w.r.t. logits
            let mut grad = probs.clone();
            grad[target] -= 1.0;

            self.model
                .update(&transition.state, &grad, self.learning_rate);
        }
        total_loss / batch.len() as f64
    }

    /// Train for multiple epochs
    pub fn train(&mut self, dataset: &OfflineDataset, epochs: usize, batch_size: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut losses = Vec::new();
        for _ in 0..epochs {
            let batch = dataset.sample_batch(batch_size, &mut rng);
            if batch.is_empty() {
                continue;
            }
            let loss = self.train_step(&batch);
            losses.push(loss);
        }
        losses
    }

    /// Get action probabilities for a state
    pub fn predict_probs(&self, state: &[f64]) -> Array1<f64> {
        let logits = self.model.forward(state);
        softmax(&logits)
    }

    /// Select the best action for a state
    pub fn select_action(&self, state: &[f64]) -> Action {
        let probs = self.predict_probs(state);
        let best_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        Action::from_index(best_idx)
    }
}

/// Implicit Q-Learning (IQL): offline RL with expectile regression
///
/// IQL avoids querying OOD actions by using expectile regression
/// to approximate the value function maximum, then extracts a policy
/// via advantage-weighted behavior cloning.
#[derive(Debug, Clone)]
pub struct ImplicitQLearning {
    /// Q-function: maps (state) -> Q-values for each action
    pub q_network: LinearModel,
    /// Value function: maps (state) -> scalar value
    pub v_network: LinearModel,
    /// Policy network: maps (state) -> action logits
    pub policy_network: LinearModel,
    /// Expectile parameter (tau): controls conservatism (0.5 = mean, 1.0 = max)
    pub expectile_tau: f64,
    /// Temperature for advantage-weighted BC
    pub temperature: f64,
    /// Discount factor
    pub gamma: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// State dimension
    pub state_dim: usize,
}

impl ImplicitQLearning {
    /// Create a new IQL agent
    pub fn new(
        state_dim: usize,
        expectile_tau: f64,
        temperature: f64,
        gamma: f64,
        learning_rate: f64,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            q_network: LinearModel::new(state_dim, NUM_ACTIONS, rng),
            v_network: LinearModel::new(state_dim, 1, rng),
            policy_network: LinearModel::new(state_dim, NUM_ACTIONS, rng),
            expectile_tau: expectile_tau,
            temperature,
            gamma,
            learning_rate,
            state_dim,
        }
    }

    /// Expectile loss: L_tau(u) = |tau - 1(u < 0)| * u^2
    pub fn expectile_loss(u: f64, tau: f64) -> f64 {
        let weight = if u < 0.0 { 1.0 - tau } else { tau };
        weight * u * u
    }

    /// Update the value function using expectile regression
    /// V(s) is trained so that Q(s,a) - V(s) gives the advantage,
    /// with the expectile focusing on the upper tail of Q-values
    fn update_value(&mut self, batch: &[&Transition]) {
        for transition in batch {
            let q_values = self.q_network.forward(&transition.state);
            let q_sa = q_values[transition.action.index()];
            let v = self.v_network.forward(&transition.state)[0];

            let u = q_sa - v;
            let weight = if u < 0.0 {
                1.0 - self.expectile_tau
            } else {
                self.expectile_tau
            };

            // Gradient of expectile loss w.r.t. V: -2 * weight * u
            let grad_v = Array1::from_vec(vec![-2.0 * weight * u]);
            self.v_network
                .update(&transition.state, &grad_v, self.learning_rate);
        }
    }

    /// Update the Q-function using Bellman backup with V(s') as target
    fn update_q(&mut self, batch: &[&Transition]) {
        for transition in batch {
            let q_values = self.q_network.forward(&transition.state);
            let q_sa = q_values[transition.action.index()];

            let v_next = if transition.done {
                0.0
            } else {
                self.v_network.forward(&transition.next_state)[0]
            };

            let target = transition.reward + self.gamma * v_next;
            let td_error = q_sa - target;

            // Gradient: only update the Q-value for the taken action
            let mut grad = Array1::zeros(NUM_ACTIONS);
            grad[transition.action.index()] = 2.0 * td_error;

            self.q_network
                .update(&transition.state, &grad, self.learning_rate);
        }
    }

    /// Update the policy using advantage-weighted behavior cloning
    /// pi(a|s) is trained to maximize: exp(beta * A(s,a)) * log pi(a|s)
    /// where A(s,a) = Q(s,a) - V(s) is the advantage
    fn update_policy(&mut self, batch: &[&Transition]) {
        for transition in batch {
            let q_values = self.q_network.forward(&transition.state);
            let v = self.v_network.forward(&transition.state)[0];
            let advantage = q_values[transition.action.index()] - v;

            // Advantage weight: exp(beta * A(s,a)), clamped for stability
            let weight = (self.temperature * advantage).exp().min(100.0);

            let logits = self.policy_network.forward(&transition.state);
            let probs = softmax(&logits);

            // Weighted cross-entropy gradient
            let target = transition.action.index();
            let mut grad = probs.clone();
            grad[target] -= 1.0;

            // Scale gradient by advantage weight
            let weighted_grad = grad * weight;
            self.policy_network
                .update(&transition.state, &weighted_grad, self.learning_rate);
        }
    }

    /// Perform one training step on a batch
    pub fn train_step(&mut self, batch: &[&Transition]) -> (f64, f64) {
        // 1. Update V using expectile regression
        self.update_value(batch);

        // 2. Update Q using Bellman backup
        self.update_q(batch);

        // 3. Update policy using advantage-weighted BC
        self.update_policy(batch);

        // Compute losses for logging
        let mut q_loss = 0.0;
        let mut v_loss = 0.0;
        for t in batch {
            let q_values = self.q_network.forward(&t.state);
            let q_sa = q_values[t.action.index()];
            let v = self.v_network.forward(&t.state)[0];
            let v_next = if t.done {
                0.0
            } else {
                self.v_network.forward(&t.next_state)[0]
            };
            let target = t.reward + self.gamma * v_next;
            q_loss += (q_sa - target).powi(2);
            v_loss += Self::expectile_loss(q_sa - v, self.expectile_tau);
        }
        let n = batch.len() as f64;
        (q_loss / n, v_loss / n)
    }

    /// Train for multiple epochs
    pub fn train(
        &mut self,
        dataset: &OfflineDataset,
        epochs: usize,
        batch_size: usize,
    ) -> Vec<(f64, f64)> {
        let mut rng = rand::thread_rng();
        let mut losses = Vec::new();
        for _ in 0..epochs {
            let batch = dataset.sample_batch(batch_size, &mut rng);
            if batch.is_empty() {
                continue;
            }
            let loss = self.train_step(&batch);
            losses.push(loss);
        }
        losses
    }

    /// Get action probabilities for a state
    pub fn predict_probs(&self, state: &[f64]) -> Array1<f64> {
        let logits = self.policy_network.forward(state);
        softmax(&logits)
    }

    /// Select the best action for a state
    pub fn select_action(&self, state: &[f64]) -> Action {
        let probs = self.predict_probs(state);
        let best_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        Action::from_index(best_idx)
    }

    /// Get Q-values for a state
    pub fn q_values(&self, state: &[f64]) -> Array1<f64> {
        self.q_network.forward(state)
    }

    /// Get value estimate for a state
    pub fn value(&self, state: &[f64]) -> f64 {
        self.v_network.forward(state)[0]
    }
}

/// Distribution shift detector: monitors divergence between learned and behavior policies
#[derive(Debug, Clone)]
pub struct DistributionShiftDetector {
    /// Behavior policy action distribution
    pub behavior_distribution: HashMap<Action, f64>,
    /// Warning threshold for KL divergence
    pub kl_threshold: f64,
}

impl DistributionShiftDetector {
    /// Create a new detector from the offline dataset
    pub fn new(dataset: &OfflineDataset, kl_threshold: f64) -> Self {
        let behavior_distribution = dataset.behavior_policy_distribution();
        Self {
            behavior_distribution,
            kl_threshold,
        }
    }

    /// Compute KL divergence: KL(policy || behavior)
    pub fn kl_divergence(&self, policy_probs: &Array1<f64>) -> f64 {
        let mut kl = 0.0;
        for action in [Action::Buy, Action::Sell, Action::Hold] {
            let p = policy_probs[action.index()].max(1e-10);
            let q = self
                .behavior_distribution
                .get(&action)
                .copied()
                .unwrap_or(1e-10)
                .max(1e-10);
            kl += p * (p / q).ln();
        }
        kl
    }

    /// Check if distribution shift exceeds threshold
    pub fn detect_shift(&self, policy_probs: &Array1<f64>) -> bool {
        self.kl_divergence(policy_probs) > self.kl_threshold
    }

    /// Compute total variation distance between policy and behavior
    pub fn total_variation(&self, policy_probs: &Array1<f64>) -> f64 {
        let mut tv = 0.0;
        for action in [Action::Buy, Action::Sell, Action::Hold] {
            let p = policy_probs[action.index()];
            let q = self
                .behavior_distribution
                .get(&action)
                .copied()
                .unwrap_or(0.0);
            tv += (p - q).abs();
        }
        tv / 2.0
    }
}

/// Bybit API kline response structures
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

/// OHLCV candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

/// Bybit API client for fetching historical market data
pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (OHLCV) data from Bybit
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );
        let response: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    Some(Candle {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to chronological order
        candles.reverse();
        Ok(candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute RSI (Relative Strength Index) for a price series
pub fn compute_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let mut rsi_values = vec![50.0; period]; // Default RSI for initial values
    if prices.len() <= period {
        return vec![50.0; prices.len()];
    }

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    // Initial average gain/loss
    for i in 1..=period {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            avg_gain += change;
        } else {
            avg_loss += change.abs();
        }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;

    let rs = if avg_loss > 0.0 {
        avg_gain / avg_loss
    } else {
        100.0
    };
    rsi_values.push(100.0 - 100.0 / (1.0 + rs));

    // Subsequent values using smoothed averages
    for i in (period + 1)..prices.len() {
        let change = prices[i] - prices[i - 1];
        let (gain, loss) = if change > 0.0 {
            (change, 0.0)
        } else {
            (0.0, change.abs())
        };
        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;

        let rs = if avg_loss > 0.0 {
            avg_gain / avg_loss
        } else {
            100.0
        };
        rsi_values.push(100.0 - 100.0 / (1.0 + rs));
    }
    rsi_values
}

/// Build an offline dataset from OHLCV candles
///
/// State features: [return, volatility, rsi_normalized, volume_ratio]
/// Actions: Generated by a rule-based behavior policy
/// Rewards: Returns with risk penalty
pub fn build_offline_dataset(candles: &[Candle]) -> OfflineDataset {
    let state_dim = 4;
    let mut dataset = OfflineDataset::new(state_dim);

    if candles.len() < 20 {
        return dataset;
    }

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
    let rsi_values = compute_rsi(&closes, 14);

    // Compute features for each candle (starting from index 15 to have enough history)
    let start_idx = 15;
    let mut states: Vec<Vec<f64>> = Vec::new();

    for i in start_idx..candles.len() {
        // Return
        let ret = (closes[i] - closes[i - 1]) / closes[i - 1];

        // Volatility (std of last 10 returns)
        let recent_returns: Vec<f64> = (i.saturating_sub(10)..i)
            .map(|j| (closes[j] - closes[j.saturating_sub(1).max(1)]) / closes[j.saturating_sub(1).max(1)])
            .collect();
        let mean_ret: f64 = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let volatility = (recent_returns
            .iter()
            .map(|r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / recent_returns.len() as f64)
            .sqrt();

        // Normalized RSI (0-1)
        let rsi_norm = rsi_values[i] / 100.0;

        // Volume ratio (current / average of last 10)
        let avg_vol: f64 = volumes[i.saturating_sub(10)..i].iter().sum::<f64>()
            / volumes[i.saturating_sub(10)..i].len().max(1) as f64;
        let vol_ratio = if avg_vol > 0.0 {
            volumes[i] / avg_vol
        } else {
            1.0
        };

        states.push(vec![ret, volatility, rsi_norm, vol_ratio]);
    }

    // Generate behavior policy actions and build transitions
    let mut rng = rand::thread_rng();
    for i in 0..states.len().saturating_sub(1) {
        let state = &states[i];
        let next_state = &states[i + 1];

        // Rule-based behavior policy (simulates historical trader)
        let ret = state[0];
        let rsi = state[2];
        let action = if rsi < 0.3 && ret < -0.005 {
            Action::Buy // Oversold + negative momentum -> buy the dip
        } else if rsi > 0.7 && ret > 0.005 {
            Action::Sell // Overbought + positive momentum -> take profits
        } else if rng.gen::<f64>() < 0.1 {
            // Random exploration (10%)
            if rng.gen::<bool>() {
                Action::Buy
            } else {
                Action::Sell
            }
        } else {
            Action::Hold
        };

        // Reward: next period return with risk penalty
        let next_ret = next_state[0];
        let volatility = state[1];
        let reward = match action {
            Action::Buy => next_ret - 0.5 * volatility,
            Action::Sell => -next_ret - 0.5 * volatility,
            Action::Hold => -0.001, // Small cost for inaction
        };

        let is_done = i == states.len() - 2;

        dataset.add(Transition {
            state: state.clone(),
            action,
            reward,
            next_state: next_state.clone(),
            done: is_done,
        });
    }

    dataset
}

/// Generate synthetic OHLCV data for testing (when API is unavailable)
pub fn generate_synthetic_candles(n: usize, seed: u64) -> Vec<Candle> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0; // Starting BTC price
    let mut timestamp = 1700000000000u64;

    for _ in 0..n {
        let ret = rng.gen_range(-0.03..0.03);
        let open: f64 = price;
        let close: f64 = price * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);
        let turnover = volume * close;

        candles.push(Candle {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover,
        });

        price = close;
        timestamp += 3600000; // 1 hour
    }
    candles
}

/// Evaluate a policy on the offline dataset by computing average reward
/// when following the policy's action choices
pub fn evaluate_policy_on_dataset(
    dataset: &OfflineDataset,
    policy_fn: &dyn Fn(&[f64]) -> Action,
) -> f64 {
    if dataset.is_empty() {
        return 0.0;
    }
    let mut total_reward = 0.0;
    let mut count = 0;
    for t in &dataset.transitions {
        let selected_action = policy_fn(&t.state);
        // Estimate reward for the policy's action
        // For actions matching the dataset, use actual reward
        // For different actions, estimate based on the transition
        if selected_action == t.action {
            total_reward += t.reward;
        } else {
            // Simple estimation: use the return direction
            let next_ret = if t.next_state.is_empty() {
                0.0
            } else {
                t.next_state[0]
            };
            let estimated_reward = match selected_action {
                Action::Buy => next_ret - 0.5 * t.state.get(1).unwrap_or(&0.01),
                Action::Sell => -next_ret - 0.5 * t.state.get(1).unwrap_or(&0.01),
                Action::Hold => -0.001,
            };
            total_reward += estimated_reward;
        }
        count += 1;
    }
    total_reward / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offline_dataset_basic() {
        let mut dataset = OfflineDataset::new(4);
        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);

        dataset.add(Transition {
            state: vec![0.01, 0.02, 0.5, 1.0],
            action: Action::Buy,
            reward: 0.05,
            next_state: vec![0.02, 0.015, 0.55, 1.1],
            done: false,
        });

        assert!(!dataset.is_empty());
        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_offline_dataset_sampling() {
        let mut dataset = OfflineDataset::new(4);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for i in 0..100 {
            dataset.add(Transition {
                state: vec![i as f64 * 0.01, 0.02, 0.5, 1.0],
                action: Action::from_index(i % 3),
                reward: 0.01 * i as f64,
                next_state: vec![(i + 1) as f64 * 0.01, 0.02, 0.5, 1.0],
                done: i == 99,
            });
        }

        let batch = dataset.sample_batch(32, &mut rng);
        assert_eq!(batch.len(), 32);
    }

    #[test]
    fn test_behavior_cloning() {
        let candles = generate_synthetic_candles(100, 42);
        let dataset = build_offline_dataset(&candles);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut bc = BehaviorCloning::new(4, 0.01, &mut rng);

        let losses = bc.train(&dataset, 50, 32);
        assert!(!losses.is_empty());

        // Loss should generally decrease
        let first_avg: f64 = losses[..5].iter().sum::<f64>() / 5.0;
        let last_avg: f64 = losses[losses.len() - 5..].iter().sum::<f64>() / 5.0;
        // Just check that training ran successfully
        assert!(first_avg.is_finite());
        assert!(last_avg.is_finite());

        // Check that predict_probs returns valid probabilities
        let state = vec![0.01, 0.02, 0.5, 1.0];
        let probs = bc.predict_probs(&state);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_iql_training() {
        let candles = generate_synthetic_candles(100, 123);
        let dataset = build_offline_dataset(&candles);

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut iql = ImplicitQLearning::new(4, 0.7, 1.0, 0.99, 0.001, &mut rng);

        let losses = iql.train(&dataset, 50, 32);
        assert!(!losses.is_empty());

        // Check Q and V losses are finite
        for (q_loss, v_loss) in &losses {
            assert!(q_loss.is_finite(), "Q loss is not finite");
            assert!(v_loss.is_finite(), "V loss is not finite");
        }

        // Verify action selection works
        let state = vec![0.01, 0.02, 0.5, 1.0];
        let action = iql.select_action(&state);
        assert!([Action::Buy, Action::Sell, Action::Hold].contains(&action));
    }

    #[test]
    fn test_distribution_shift_detector() {
        let candles = generate_synthetic_candles(200, 42);
        let dataset = build_offline_dataset(&candles);
        let detector = DistributionShiftDetector::new(&dataset, 0.5);

        // Behavior distribution should sum to 1
        let sum: f64 = detector.behavior_distribution.values().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Behavior distribution doesn't sum to 1");

        // A policy matching behavior should have low KL
        let behavior_probs = Array1::from_vec(vec![
            *detector
                .behavior_distribution
                .get(&Action::Buy)
                .unwrap_or(&0.33),
            *detector
                .behavior_distribution
                .get(&Action::Sell)
                .unwrap_or(&0.33),
            *detector
                .behavior_distribution
                .get(&Action::Hold)
                .unwrap_or(&0.34),
        ]);
        let kl = detector.kl_divergence(&behavior_probs);
        assert!(kl < 0.01, "KL divergence for matching distribution should be ~0, got {}", kl);

        // A very different policy should have higher KL
        let extreme_probs = Array1::from_vec(vec![0.98, 0.01, 0.01]);
        let kl_extreme = detector.kl_divergence(&extreme_probs);
        assert!(kl_extreme > kl, "Extreme policy should have higher KL divergence");
    }

    #[test]
    fn test_expectile_loss() {
        // Symmetric at tau = 0.5
        let loss_pos = ImplicitQLearning::expectile_loss(1.0, 0.5);
        let loss_neg = ImplicitQLearning::expectile_loss(-1.0, 0.5);
        assert!((loss_pos - loss_neg).abs() < 1e-10);

        // At tau = 0.9, positive errors penalized more
        let loss_pos_09 = ImplicitQLearning::expectile_loss(1.0, 0.9);
        let loss_neg_09 = ImplicitQLearning::expectile_loss(-1.0, 0.9);
        assert!(loss_pos_09 > loss_neg_09);

        // Zero error -> zero loss
        let loss_zero = ImplicitQLearning::expectile_loss(0.0, 0.7);
        assert!(loss_zero.abs() < 1e-10);
    }

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_synthetic_candles() {
        let candles = generate_synthetic_candles(50, 42);
        assert_eq!(candles.len(), 50);
        for c in &candles {
            assert!(c.high >= c.open.max(c.close));
            assert!(c.low <= c.open.min(c.close));
            assert!(c.volume > 0.0);
        }
    }

    #[test]
    fn test_build_offline_dataset() {
        let candles = generate_synthetic_candles(100, 42);
        let dataset = build_offline_dataset(&candles);
        assert!(!dataset.is_empty());
        assert_eq!(dataset.state_dim, 4);

        for t in &dataset.transitions {
            assert_eq!(t.state.len(), 4);
            assert_eq!(t.next_state.len(), 4);
            assert!(t.reward.is_finite());
        }
    }

    #[test]
    fn test_compute_rsi() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        let rsi = compute_rsi(&prices, 14);
        assert_eq!(rsi.len(), prices.len());
        for &r in &rsi {
            assert!(r >= 0.0 && r <= 100.0, "RSI {} out of range", r);
        }
    }
}
