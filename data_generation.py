import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Let's simulate 30 days of data at 1-hour intervals.
SIMULATION_DAYS = 30
HOURS_IN_DAY = 24
TOTAL_HOURS = SIMULATION_DAYS * HOURS_IN_DAY


# --- Data Generation Functions ---

def generate_solar_power(total_hours):
    """
    Generates simulated solar power generation data.
    - Peaks during the day, zero at night.
    - Adds some random noise to simulate weather (e.g., clouds).
    """
    # Create a daily solar pattern (peaks around noon)
    daily_solar_pattern = np.sin(np.linspace(0, np.pi, HOURS_IN_DAY)) ** 2
    # Scale it to a max power (e.g., 5 kW)
    daily_solar_pattern *= 5

    # Repeat the daily pattern for the entire simulation period
    solar_generation = np.tile(daily_solar_pattern, SIMULATION_DAYS)

    # Add some random noise to simulate cloud cover and weather variations
    noise = np.random.normal(0, 0.3, total_hours)
    solar_generation += noise

    # Ensure generation is not negative
    solar_generation[solar_generation < 0] = 0
    return solar_generation


def generate_wind_power(total_hours):
    """
    Generates simulated wind power generation data.
    - More random and less predictable than solar.
    """
    # Start with some random noise
    base_wind = np.random.normal(2, 1, total_hours)
    # Use a moving average to smooth it out, making it more realistic
    wind_generation = pd.Series(base_wind).rolling(window=6, min_periods=1, center=True).mean().to_numpy()

    # Ensure generation is not negative
    wind_generation[wind_generation < 0] = 0
    return wind_generation


def generate_household_demand(total_hours):
    """
    Generates simulated household energy demand (load).
    - Typically has morning and evening peaks.
    """
    # Create a base daily demand pattern
    base_demand = np.ones(HOURS_IN_DAY) * 1.5  # Base load
    # Morning peak (7-9 AM)
    base_demand[7:10] = np.array([2.5, 3.5, 3.0])
    # Evening peak (6-9 PM)
    base_demand[18:22] = np.array([4.0, 4.5, 4.2, 3.8])

    # Repeat the daily pattern
    household_demand = np.tile(base_demand, SIMULATION_DAYS)

    # Add some random noise for variability
    noise = np.random.normal(0, 0.4, total_hours)
    household_demand += noise

    # Ensure demand is not negative
    household_demand[household_demand < 0] = 0.5
    return household_demand


def generate_grid_prices(total_hours):
    """
    Generates simulated electricity prices from the main grid.
    - Prices are higher during peak demand hours.
    """
    # Create a base daily price pattern (in cents per kWh)
    base_price = np.ones(HOURS_IN_DAY) * 12
    # Off-peak hours (e.g., midnight to 6 AM) are cheaper
    base_price[0:7] = 8
    # Peak hours (e.g., 5 PM to 9 PM) are more expensive
    base_price[17:22] = np.array([20, 25, 25, 22, 20])

    # Repeat the daily pattern
    grid_prices = np.tile(base_price, SIMULATION_DAYS)

    # Add some random noise
    noise = np.random.normal(0, 1, total_hours)
    grid_prices += noise

    # Ensure prices are not unrealistically low
    grid_prices[grid_prices < 5] = 5
    return grid_prices


# --- Main Script Execution ---
if __name__ == "__main__":
    print("Generating simulated smart grid data...")

    # Create a datetime index for our data
    time_index = pd.date_range(start='2024-01-01', periods=TOTAL_HOURS, freq='H')

    # Generate all data components
    df = pd.DataFrame(index=time_index)
    df['solar_generation_kw'] = generate_solar_power(TOTAL_HOURS)
    df['wind_generation_kw'] = generate_wind_power(TOTAL_HOURS)
    df['household_demand_kw'] = generate_household_demand(TOTAL_HOURS)
    df['grid_price_cents_kwh'] = generate_grid_prices(TOTAL_HOURS)

    # Save the generated data to a CSV file
    output_filename = 'data/simulated_grid_data.csv'
    df.to_csv(output_filename)

    print(f"Data generation complete. Saved to '{output_filename}'")
    print("\nFirst 5 rows of the generated data:")
    print(df.head())

    # --- Visualize the first 24 hours of data ---
    print("\nPlotting the first 24 hours of data...")
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot generation and demand on the primary y-axis
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Power (kW)', color='tab:blue')
    ax1.plot(df.index[:24], df['solar_generation_kw'][:24], label='Solar Generation', color='orange', linestyle='--')
    ax1.plot(df.index[:24], df['wind_generation_kw'][:24], label='Wind Generation', color='skyblue', linestyle='--')
    ax1.plot(df.index[:24], df['household_demand_kw'][:24], label='Household Demand', color='tab:blue', linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, axis='y', linestyle=':')

    # Create a secondary y-axis for the price
    ax2 = ax1.twinx()
    ax2.set_ylabel('Price (cents/kWh)', color='tab:red')
    ax2.plot(df.index[:24], df['grid_price_cents_kwh'][:24], label='Grid Price', color='tab:red', marker='o')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Final plot touches
    fig.suptitle('Simulated Smart Grid Data (First 24 Hours)', fontsize=16)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()