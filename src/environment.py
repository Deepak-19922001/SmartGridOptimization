import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from .components import Battery


class SmartGridEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, data_path: str, battery_capacity_kwh: float = 10.0, battery_max_charge_kw: float = 5.0,
                 battery_max_discharge_kw: float = 5.0):
        super(SmartGridEnv, self).__init__()

        # --- Load and prepare data ---
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.max_steps = len(self.df) - 1

        # --- Initialize grid components ---
        self.battery = Battery(
            capacity_kwh=battery_capacity_kwh,
            max_charge_kw=battery_max_charge_kw,
            max_discharge_kw=battery_max_discharge_kw
        )

        # --- Define the Action Space ---
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # --- Define the Observation Space ---
        low_bounds = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        high_bounds = np.array([
            23,  # Hour
            self.df['solar_generation_kw'].max(),
            self.df['wind_generation_kw'].max(),
            self.df['household_demand_kw'].max(),
            self.df['grid_price_cents_kwh'].max(),
            1.0  # Battery State of Charge
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # --- Environment State ---
        self.current_step = 0

    def _get_obs(self):
        """Constructs the observation array for the current time step."""
        current_row = self.df.iloc[self.current_step]
        hour_of_day = current_row.name.hour

        obs = np.array([
            hour_of_day,
            current_row['solar_generation_kw'],
            current_row['wind_generation_kw'],
            current_row['household_demand_kw'],
            current_row['grid_price_cents_kwh'],
            self.battery.get_state_of_charge()
        ], dtype=np.float32)
        return obs

    def _get_info(self):
        """Returns auxiliary information about the current step."""
        return {"current_step": self.current_step}

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.battery.current_charge_kwh = 0.0  # Reset battery

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Executes one time step within the environment.
        This is the core logic of the simulation.
        """
        # --- 1. Execute Agent's Action ---
        action_value = action[0]

        if action_value > 0:
            power_to_charge = action_value * self.battery.max_charge_kw
            power_from_battery = -self.battery.charge(power_to_charge)
        else:
            power_to_discharge = -action_value * self.battery.max_discharge_kw
            power_from_battery = self.battery.discharge(power_to_discharge)

        # --- 2. Calculate Energy Balance ---
        current_data = self.df.iloc[self.current_step]
        solar_gen = current_data['solar_generation_kw']
        wind_gen = current_data['wind_generation_kw']
        demand = current_data['household_demand_kw']

        local_generation = solar_gen + wind_gen + power_from_battery
        net_power = demand - local_generation

        # --- 3. Interact with the Main Grid ---
        power_bought_from_grid = max(0, net_power)
        power_sold_to_grid = max(0, -net_power)

        # --- 4. Calculate the Reward (Cost) ---
        grid_price = current_data['grid_price_cents_kwh']
        grid_sell_price = grid_price * 0.8

        cost = power_bought_from_grid * grid_price
        revenue = power_sold_to_grid * grid_sell_price
        reward = revenue - cost

        # --- 5. Update State and Check for Termination ---
        self.current_step += 1
        terminated = self.current_step > self.max_steps

        # *** FIX IS HERE ***
        # If the episode is terminated, the observation for the next state is irrelevant.
        # We create a dummy observation to avoid the IndexError.
        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._get_obs()

        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self, mode='human'):
        """Renders the environment (optional, for visualization)."""
        # To avoid errors on the last step, ensure we don't render past the end
        step_to_render = min(self.current_step, self.max_steps)
        current_data = self.df.iloc[step_to_render]

        print(f"Step: {step_to_render}")
        print(f"  - Demand: {current_data['household_demand_kw']:.2f} kW")
        print(f"  - Solar: {current_data['solar_generation_kw']:.2f} kW")
        print(f"  - Wind: {current_data['wind_generation_kw']:.2f} kW")
        print(f"  - Grid Price: {current_data['grid_price_cents_kwh']:.2f} cents")
        print(f"  - Battery SoC: {self.battery.get_state_of_charge():.1%}")
        print("-" * 20)


if __name__ == '__main__':
    env = SmartGridEnv(data_path='data/simulated_grid_data.csv')

    obs, info = env.reset()
    done = False
    total_reward = 0

    for _ in range(48):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated or truncated:
            break

    print(f"\nSimulation finished after 48 steps.")
    print(f"Total reward (random actions): {total_reward:.2f}")
