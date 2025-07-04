import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.environment import SmartGridEnv


def run_simulation(env, model=None):
    obs, _ = env.reset()
    done = False

    history = []

    while not done:
        if model:
            # Use the trained agent to pick an action
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Baseline "dumb" strategy:
            # If demand > renewables, discharge battery. If still not enough, buy.
            # If renewables > demand, charge battery with surplus. If still surplus, sell.
            current_obs = obs
            demand = current_obs[3]
            solar = current_obs[1]
            wind = current_obs[2]
            soc = current_obs[5]

            renewable_gen = solar + wind
            net_demand = demand - renewable_gen

            if net_demand > 0:  # Need more power
                if soc > 0.1:  # Discharge if battery has meaningful charge
                    action = np.array([-1.0])  # Max discharge
                else:
                    action = np.array([0.0])  # Do nothing with battery, will buy from grid
            else:  # Have surplus power
                if soc < 0.9:  # Charge if battery is not full
                    action = np.array([1.0])  # Max charge
                else:
                    action = np.array([0.0])  # Do nothing, will sell to grid

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store data for this step
        current_data = env.df.iloc[env.current_step - 1]
        history.append({
            'timestamp': current_data.name,
            'demand_kw': current_data['household_demand_kw'],
            'solar_kw': current_data['solar_generation_kw'],
            'wind_kw': current_data['wind_generation_kw'],
            'grid_price': current_data['grid_price_cents_kwh'],
            'battery_soc': env.battery.get_state_of_charge(),
            'action': action[0],
            'reward': reward
        })

    return pd.DataFrame(history)


def plot_results(smart_history, baseline_history):
    """Plots the comparison between the smart agent and the baseline."""

    # Calculate cumulative cost (negative of cumulative reward)
    smart_history['cumulative_cost_cents'] = -smart_history['reward'].cumsum()
    baseline_history['cumulative_cost_cents'] = -baseline_history['reward'].cumsum()

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Plot 1: Battery State of Charge
    axes[0].plot(smart_history['timestamp'], smart_history['battery_soc'], label='Smart Agent SoC', color='b')
    axes[0].plot(baseline_history['timestamp'], baseline_history['battery_soc'], label='Baseline SoC',
                 color='lightblue', linestyle='--')
    axes[0].set_ylabel('Battery SoC (%)')
    axes[0].set_title('Battery State of Charge Comparison')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Agent Actions
    axes[1].plot(smart_history['timestamp'], smart_history['action'], label='Smart Agent Action', color='g', alpha=0.7)
    axes[1].set_ylabel('Action (-1: Discharge, 1: Charge)')
    axes[1].set_title('Smart Agent Actions Over Time')
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Cumulative Cost
    axes[2].plot(smart_history['timestamp'], smart_history['cumulative_cost_cents'] / 100, label='Smart Agent Cost ($)',
                 color='r')
    axes[2].plot(baseline_history['timestamp'], baseline_history['cumulative_cost_cents'] / 100,
                 label='Baseline Cost ($)', color='orange', linestyle='--')
    axes[2].set_ylabel('Cumulative Cost ($)')
    axes[2].set_title('Cumulative Cost Comparison')
    axes[2].legend()
    axes[2].grid(True)

    plt.xlabel('Time')
    fig.tight_layout()
    plt.show()

    # Print final costs
    final_smart_cost = smart_history['cumulative_cost_cents'].iloc[-1] / 100
    final_baseline_cost = baseline_history['cumulative_cost_cents'].iloc[-1] / 100
    savings = final_baseline_cost - final_smart_cost

    print("\n--- Simulation Results ---")
    print(f"Baseline Strategy Final Cost: ${final_baseline_cost:.2f}")
    print(f"Smart Agent Final Cost: ${final_smart_cost:.2f}")
    print(f"Total Savings: ${savings:.2f} ({savings / final_baseline_cost:.2%})")


if __name__ == '__main__':
    # --- 1. Set up Environment and Load Model ---
    env = SmartGridEnv(data_path='data/simulated_grid_data.csv')

    # Load the best model saved during training
    # NOTE: If your best model has a different name, change it here.
    model_path = "saved_models/best_model.zip"
    try:
        model = PPO.load(model_path, env=env)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        print("Please ensure you have run train_agent.py and a model has been saved.")
        exit()

    # --- 2. Run Simulations ---
    print("Running simulation with the trained SMART AGENT...")
    smart_agent_history = run_simulation(env, model)

    print("\nRunning simulation with the BASELINE STRATEGY...")
    baseline_strategy_history = run_simulation(env, model=None)

    # --- 3. Plot Results ---
    plot_results(smart_agent_history, baseline_strategy_history)
