# Chapter 313: Offline Reinforcement Learning for Trading

## Introduction

Offline reinforcement learning (also called batch RL) represents a paradigm shift in how we apply RL to financial markets. Unlike traditional online RL, which requires continuous interaction with an environment (and thus exposure to real financial risk), offline RL learns entirely from a fixed dataset of previously collected transitions. This is particularly compelling for trading because:

1. **No live risk during training**: The agent never places a real trade while learning, eliminating the catastrophic losses that can occur when an untrained policy explores in live markets.
2. **Leveraging historical data**: Firms accumulate years of trading logs, order executions, and market data. Offline RL can extract optimal policies from this historical record.
3. **Regulatory and compliance benefits**: Regulators prefer strategies that can be validated on historical data before deployment, and offline RL naturally fits this requirement.
4. **Reproducibility**: Training on a fixed dataset means experiments are fully reproducible, unlike online RL where market conditions change between runs.

The central challenge of offline RL is **distribution shift**: the learned policy may want to take actions that are poorly represented in the historical dataset, leading to overestimated Q-values and catastrophic real-world performance. This chapter covers the mathematical foundations, key algorithms (BCQ, BEAR, IQL), and a complete Rust implementation with Bybit market data integration.

## Mathematical Foundations

### The Offline RL Problem

In standard RL, we have a Markov Decision Process (MDP) defined by the tuple $(S, A, P, R, \gamma)$:

- $S$: State space (market features: prices, volumes, indicators)
- $A$: Action space (buy, sell, hold, position sizes)
- $P(s'|s,a)$: Transition dynamics (market evolution)
- $R(s,a)$: Reward function (trading P&L, risk-adjusted returns)
- $\gamma$: Discount factor

In offline RL, we are given a fixed dataset $\mathcal{D} = \{(s_i, a_i, r_i, s_i')\}_{i=1}^{N}$ collected by some **behavior policy** $\beta(a|s)$. Our goal is to learn a policy $\pi(a|s)$ that maximizes the expected cumulative reward, using only $\mathcal{D}$ without any additional environment interaction.

### The Distribution Shift Problem

The fundamental difficulty arises from the mismatch between the state-action distribution induced by our learned policy $\pi$ and the distribution in the dataset $\mathcal{D}$. When we use the Bellman equation for policy evaluation:

$$Q^{\pi}(s,a) = R(s,a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)} [V^{\pi}(s')]$$

where $V^{\pi}(s') = \mathbb{E}_{a' \sim \pi(\cdot|s')}[Q^{\pi}(s', a')]$, the value $Q^{\pi}(s', a')$ may be queried at state-action pairs $(s', a')$ that are out-of-distribution (OOD). For OOD pairs, the Q-function can produce arbitrarily overestimated values because it was never corrected by real transitions at those points.

In trading, this manifests when the offline policy decides to take a large leveraged position in a volatile asset, but the historical dataset only contains conservative trades. The Q-function might assign high values to these unseen aggressive actions, leading to disastrous real deployment.

### Behavior Cloning Baseline

The simplest offline approach is **behavior cloning** (BC): treat the problem as supervised learning and directly imitate the behavior policy:

$$\pi_{BC} = \arg\max_{\pi} \mathbb{E}_{(s,a) \sim \mathcal{D}} [\log \pi(a|s)]$$

BC avoids distribution shift entirely since it only selects actions seen in the data. However, it is limited to at most matching the performance of the behavior policy and cannot improve upon it. If the historical trades were suboptimal, BC will faithfully replicate their suboptimality.

### The Pessimism Principle

Modern offline RL algorithms address distribution shift through the **pessimism principle**: be conservative about actions not well-represented in the data. This can be implemented via:

1. **Support constraint**: Only consider actions within the support of $\beta(a|s)$ (BCQ)
2. **Distribution matching**: Constrain $\pi$ to be close to $\beta$ via MMD or KL divergence (BEAR)
3. **Implicit pessimism**: Use expectile regression to learn a conservative value function (IQL)

Formally, the pessimistic Bellman operator is:

$$\hat{Q}^{\pi}(s,a) = R(s,a) + \gamma \mathbb{E}_{s'}[\max_{a': a' \in \text{supp}(\beta(\cdot|s'))} Q^{\pi}(s', a') - \lambda \cdot u(s', a')]$$

where $u(s',a')$ is an uncertainty penalty that increases for actions far from the data distribution.

## Key Algorithms

### BCQ (Batch-Constrained deep Q-learning)

BCQ (Fujimoto et al., 2019) constrains the policy to only select actions similar to those in the dataset. It trains a generative model (VAE) of the behavior policy and only considers actions within a threshold of generated samples:

1. Train a conditional VAE to model $\beta(a|s)$
2. Sample $n$ candidate actions from the VAE for each state
3. Select the action with the highest Q-value among candidates
4. Perturb the selected action with a learned perturbation network

**Trading application**: BCQ ensures the agent only considers trade sizes and timings similar to historical executions, preventing extreme positions.

### BEAR (Bootstrapping Error Accumulation Reduction)

BEAR (Kumar et al., 2019) constrains the learned policy to have a bounded Maximum Mean Discrepancy (MMD) from the behavior policy:

$$\pi^* = \arg\max_{\pi} \mathbb{E}_{s \sim \mathcal{D}} [\mathbb{E}_{a \sim \pi(\cdot|s)}[Q(s,a)]]$$
$$\text{s.t. } \text{MMD}(\pi(\cdot|s) \| \beta(\cdot|s)) \leq \epsilon$$

This is more flexible than BCQ because it allows actions not exactly in the dataset, as long as the overall distribution stays close.

**Trading application**: BEAR allows the policy to discover slightly different trading strategies than historical ones while maintaining a safety bound on how far it can deviate.

### IQL (Implicit Q-Learning)

IQL (Kostrikov et al., 2022) avoids querying OOD actions entirely by using **expectile regression** to learn the value function. Instead of taking a max over actions:

$$V(s) = \max_a Q(s,a)$$

IQL uses the expectile loss:

$$L_{\tau}(u) = |\tau - \mathbb{1}(u < 0)| \cdot u^2$$

to approximate the maximum with high expectile $\tau \to 1$:

$$V_{\psi} = \arg\min_{V} \mathbb{E}_{(s,a) \sim \mathcal{D}} [L_{\tau}(Q_{\theta}(s,a) - V(s))]$$

The policy is then extracted via advantage-weighted regression:

$$\pi^* = \arg\max_{\pi} \mathbb{E}_{(s,a) \sim \mathcal{D}} [\exp(\beta \cdot (Q(s,a) - V(s))) \cdot \log \pi(a|s)]$$

**Trading application**: IQL is particularly well-suited for trading because it never evaluates Q-values on unseen actions. It extracts the best behavior from historical data using only in-sample computations, making it robust to the noisy, non-stationary nature of financial markets.

### Algorithm Comparison

| Feature | BCQ | BEAR | IQL |
|---------|-----|------|-----|
| Constraint type | Support | Distribution (MMD) | Implicit (expectile) |
| Needs behavior model | Yes (VAE) | Yes (for MMD) | No |
| Action space | Continuous | Continuous | Both |
| Conservatism control | Threshold | MMD bound | Expectile $\tau$ |
| Computational cost | High | Medium | Low |
| Trading suitability | Good | Good | Excellent |

## Applications: Learning from Historical Trading Logs

### Building the Offline Dataset

The offline dataset for trading is constructed from historical OHLCV data and trading logs:

1. **State features**: Normalized price returns, volume ratios, technical indicators (RSI, MACD, Bollinger Bands), order book imbalance, volatility estimates
2. **Actions**: Discrete (buy/sell/hold) or continuous (position size from -1 to +1)
3. **Rewards**: Period returns, Sharpe ratio contribution, drawdown penalties
4. **Transitions**: Sequential market states connected by time steps

### Advantages Over Online RL for Trading

- **Safety**: No risk of the agent blowing up an account during exploration
- **Data efficiency**: Can reuse years of historical data without needing a simulator
- **Backtesting integration**: Offline RL naturally produces backtestable policies
- **Multiple strategy extraction**: Different expectile/constraint parameters yield different risk-return profiles from the same dataset

### Practical Considerations

1. **Dataset quality matters**: Garbage in, garbage out. If the historical trades were consistently losing money, offline RL can at best learn to lose less.
2. **Non-stationarity**: Financial markets change. Policies trained on 2020 data may fail in 2024. Periodic retraining on recent data is essential.
3. **Action discretization**: For simplicity and robustness, discretizing the action space (buy/sell/hold) often works better than continuous actions in trading.

## Rust Implementation

The implementation in `rust/src/lib.rs` provides:

- **OfflineDataset**: A fixed replay buffer storing transitions from historical data, with methods for batch sampling
- **BehaviorCloning**: Supervised learning baseline that imitates the behavior policy using cross-entropy loss
- **ImplicitQLearning (IQL)**: Full implementation with expectile regression for value function and advantage-weighted behavior cloning for policy extraction
- **DistributionShiftDetector**: Monitors the divergence between learned policy and behavior policy to flag potential OOD issues
- **BybitClient**: Fetches historical OHLCV data from the Bybit API for building offline datasets

### Key Design Decisions

- **Discrete actions**: We use {Buy, Sell, Hold} for robustness and interpretability
- **Feature engineering**: Returns, volatility, RSI, and volume ratio as state features
- **Expectile loss**: Implemented with configurable $\tau$ parameter (default 0.7) for controlling conservatism
- **Temperature parameter**: Controls how aggressively the policy exploits advantages (default $\beta = 1.0$)

## Bybit Data Integration

The implementation fetches real market data from Bybit's public API:

```
GET /v5/market/kline?symbol=BTCUSDT&interval=60&limit=200
```

Each kline provides open, high, low, close, volume, and turnover. The data pipeline:

1. Fetch raw OHLCV candles from Bybit
2. Compute state features (returns, volatility, RSI, volume ratio)
3. Generate actions using a rule-based behavior policy (simulating historical trader decisions)
4. Calculate rewards (returns with risk penalty)
5. Package into `Transition` structs for the offline dataset

This allows users to quickly build realistic offline datasets for experimentation without needing proprietary trading logs.

## Key Takeaways

1. **Offline RL eliminates live risk during training** by learning entirely from historical data, making it ideal for financial applications where exploration costs are prohibitive.

2. **Distribution shift is the core challenge**: naive application of off-policy RL to fixed datasets leads to value overestimation and poor real-world performance. The pessimism principle (being conservative about unseen actions) is essential.

3. **IQL is particularly well-suited for trading** because it never evaluates Q-values on out-of-distribution actions, uses only in-sample computations, and naturally produces conservative policies.

4. **Behavior cloning provides a simple but limited baseline**: it can only match the behavior policy's performance, while offline RL methods like IQL can improve upon it.

5. **Dataset quality and recency are critical**: offline RL cannot create alpha from noise. High-quality historical data and periodic retraining are necessary for practical deployment.

6. **The expectile parameter $\tau$ controls the risk-return tradeoff**: higher values extract more aggressive policies, lower values produce more conservative ones. This provides a natural knob for risk management.

7. **Practical deployment requires distribution shift monitoring**: even after training, the agent should track how different its actions are from the training data and flag potential issues before they cause losses.
