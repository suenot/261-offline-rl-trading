//! Offline RL Trading Example
//!
//! Demonstrates:
//! 1. Fetching BTCUSDT data from Bybit (with synthetic fallback)
//! 2. Building an offline dataset from historical OHLCV
//! 3. Training behavior cloning and IQL
//! 4. Comparing offline policies

use offline_rl_trading::*;
use rand::SeedableRng;

fn main() {
    println!("=== Offline RL Trading Example ===\n");

    // Step 1: Fetch market data
    println!("Step 1: Fetching market data...");
    let candles = match fetch_bybit_data() {
        Some(c) => {
            println!("  Fetched {} candles from Bybit API", c.len());
            c
        }
        None => {
            println!("  Bybit API unavailable, using synthetic data");
            generate_synthetic_candles(200, 42)
        }
    };

    println!(
        "  Price range: {:.2} - {:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max),
    );

    // Step 2: Build offline dataset
    println!("\nStep 2: Building offline dataset...");
    let dataset = build_offline_dataset(&candles);
    println!("  Dataset size: {} transitions", dataset.len());

    let behavior_dist = dataset.behavior_policy_distribution();
    println!("  Behavior policy distribution:");
    for action in [Action::Buy, Action::Sell, Action::Hold] {
        let prob = behavior_dist.get(&action).unwrap_or(&0.0);
        println!("    {:?}: {:.1}%", action, prob * 100.0);
    }

    let (mean, std) = dataset.state_statistics();
    println!("  State feature means: [return={:.6}, vol={:.6}, rsi={:.3}, vol_ratio={:.3}]",
             mean[0], mean[1], mean[2], mean[3]);
    println!("  State feature stds:  [return={:.6}, vol={:.6}, rsi={:.3}, vol_ratio={:.3}]",
             std[0], std[1], std[2], std[3]);

    // Step 3: Train Behavior Cloning
    println!("\nStep 3: Training Behavior Cloning...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut bc = BehaviorCloning::new(4, 0.01, &mut rng);
    let bc_losses = bc.train(&dataset, 200, 32);

    let bc_first_loss: f64 = bc_losses[..5].iter().sum::<f64>() / 5.0;
    let bc_last_loss: f64 = bc_losses[bc_losses.len() - 5..].iter().sum::<f64>() / 5.0;
    println!("  Initial loss: {:.4}", bc_first_loss);
    println!("  Final loss:   {:.4}", bc_last_loss);

    // Step 4: Train IQL
    println!("\nStep 4: Training Implicit Q-Learning (IQL)...");
    let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
    let mut iql = ImplicitQLearning::new(
        4,    // state_dim
        0.7,  // expectile_tau
        1.0,  // temperature
        0.99, // gamma
        0.001, // learning_rate
        &mut rng2,
    );
    let iql_losses = iql.train(&dataset, 200, 32);

    let iql_first_q: f64 = iql_losses[..5].iter().map(|(q, _)| q).sum::<f64>() / 5.0;
    let iql_last_q: f64 = iql_losses[iql_losses.len() - 5..].iter().map(|(q, _)| q).sum::<f64>() / 5.0;
    let iql_first_v: f64 = iql_losses[..5].iter().map(|(_, v)| v).sum::<f64>() / 5.0;
    let iql_last_v: f64 = iql_losses[iql_losses.len() - 5..].iter().map(|(_, v)| v).sum::<f64>() / 5.0;
    println!("  Q-loss: {:.6} -> {:.6}", iql_first_q, iql_last_q);
    println!("  V-loss: {:.6} -> {:.6}", iql_first_v, iql_last_v);

    // Step 5: Compare policies
    println!("\nStep 5: Comparing offline policies...");

    // Evaluate on a few sample states
    let sample_states = vec![
        vec![0.02, 0.01, 0.3, 1.5],   // Positive return, low RSI, high volume
        vec![-0.02, 0.03, 0.8, 0.5],  // Negative return, high RSI, low volume
        vec![0.001, 0.005, 0.5, 1.0], // Flat, neutral RSI, normal volume
        vec![0.05, 0.02, 0.9, 2.0],   // Strong positive, overbought, spike volume
        vec![-0.03, 0.04, 0.2, 1.2],  // Strong negative, oversold
    ];

    println!("\n  {:<45} {:<10} {:<10}", "State [ret, vol, rsi, vol_ratio]", "BC", "IQL");
    println!("  {}", "-".repeat(65));

    for state in &sample_states {
        let bc_action = bc.select_action(state);
        let iql_action = iql.select_action(state);
        println!(
            "  [{:>7.4}, {:>6.4}, {:>4.2}, {:>4.2}]                {:?}{:<5} {:?}",
            state[0], state[1], state[2], state[3],
            bc_action, "", iql_action
        );
    }

    // Step 6: Distribution shift analysis
    println!("\nStep 6: Distribution shift analysis...");
    let detector = DistributionShiftDetector::new(&dataset, 0.5);

    for state in &sample_states {
        let bc_probs = bc.predict_probs(state);
        let iql_probs = iql.predict_probs(state);

        let bc_kl = detector.kl_divergence(&bc_probs);
        let iql_kl = detector.kl_divergence(&iql_probs);
        let bc_shift = detector.detect_shift(&bc_probs);
        let iql_shift = detector.detect_shift(&iql_probs);

        println!(
            "  State [{:>7.4}, ...]: BC KL={:.4} {} | IQL KL={:.4} {}",
            state[0],
            bc_kl,
            if bc_shift { "[SHIFT!]" } else { "[OK]    " },
            iql_kl,
            if iql_shift { "[SHIFT!]" } else { "[OK]    " },
        );
    }

    // Step 7: Policy evaluation
    println!("\nStep 7: Policy evaluation on dataset...");
    let bc_reward = evaluate_policy_on_dataset(&dataset, &|s: &[f64]| bc.select_action(s));
    let iql_reward = evaluate_policy_on_dataset(&dataset, &|s: &[f64]| iql.select_action(s));

    // Random baseline
    let random_reward = evaluate_policy_on_dataset(&dataset, &|_: &[f64]| {
        Action::Hold
    });

    println!("  Hold-only baseline:  avg reward = {:.6}", random_reward);
    println!("  Behavior Cloning:    avg reward = {:.6}", bc_reward);
    println!("  IQL:                 avg reward = {:.6}", iql_reward);

    // Step 8: IQL Q-value and advantage analysis
    println!("\nStep 8: IQL Q-value analysis...");
    for state in &sample_states[..3] {
        let q_values = iql.q_values(state);
        let v = iql.value(state);
        println!("  State [{:>7.4}, ...]: Q=[Buy:{:.4}, Sell:{:.4}, Hold:{:.4}] V={:.4}",
                 state[0], q_values[0], q_values[1], q_values[2], v);
        println!("    Advantages: Buy={:.4}, Sell={:.4}, Hold={:.4}",
                 q_values[0] - v, q_values[1] - v, q_values[2] - v);
    }

    println!("\n=== Offline RL Trading Example Complete ===");
}

/// Attempt to fetch data from Bybit API
fn fetch_bybit_data() -> Option<Vec<Candle>> {
    let client = BybitClient::new();
    match client.fetch_klines("BTCUSDT", "60", 200) {
        Ok(candles) if !candles.is_empty() => Some(candles),
        _ => None,
    }
}
