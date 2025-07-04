import numpy as np


class Battery:


    def __init__(self, capacity_kwh: float, max_charge_kw: float, max_discharge_kw: float, efficiency: float = 0.95):

        self.capacity_kwh = capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.efficiency = efficiency

        # Internal state
        self.current_charge_kwh = 0.0  # Start with an empty battery

    def charge(self, power_kw: float) -> float:

        # 1. Respect the maximum charge rate
        power_to_charge = np.clip(power_kw, 0, self.max_charge_kw)

        # 2. Account for charging efficiency
        effective_power_to_store = power_to_charge * self.efficiency

        # 3. Determine available capacity
        available_capacity = self.capacity_kwh - self.current_charge_kwh

        # 4. Determine how much energy can actually be stored in this time step (1 hour)
        # Since power_kw is energy per hour, for a 1-hour step, power = energy.
        energy_to_store = np.min([effective_power_to_store, available_capacity])

        # 5. Update the battery's charge
        self.current_charge_kwh += energy_to_store

        # Return the actual power that was drawn from the source to achieve this charge
        actual_power_drawn = energy_to_store / self.efficiency
        return actual_power_drawn

    def discharge(self, power_kw: float) -> float:

        # 1. Respect the maximum discharge rate
        power_to_supply = np.clip(power_kw, 0, self.max_discharge_kw)

        # 2. Determine how much energy can actually be drawn in this time step (1 hour)
        # Note: Efficiency is lost on charging, so we assume 100% on discharge side
        # for a round-trip efficiency model.
        energy_to_supply = np.min([power_to_supply, self.current_charge_kwh])

        # 3. Update the battery's charge
        self.current_charge_kwh -= energy_to_supply

        # Return the actual power supplied
        return energy_to_supply

    def get_state_of_charge(self) -> float:

        return self.current_charge_kwh / self.capacity_kwh


# --- Example Usage (for testing purposes) ---
if __name__ == '__main__':
    # Create a battery with 10 kWh capacity, 5 kW charge/discharge rate
    my_battery = Battery(capacity_kwh=10, max_charge_kw=5, max_discharge_kw=5)

    print(f"Initial Charge: {my_battery.current_charge_kwh:.2f} kWh, SoC: {my_battery.get_state_of_charge():.0%}")

    # Try to charge with 3 kW
    power_drawn = my_battery.charge(3)
    print(f"\nCharging with 3 kW...")
    print(f"Actual power drawn from source: {power_drawn:.2f} kW")
    print(f"New Charge: {my_battery.current_charge_kwh:.2f} kWh, SoC: {my_battery.get_state_of_charge():.0%}")

    # Try to charge with 7 kW (should be capped at 5 kW)
    power_drawn = my_battery.charge(7)
    print(f"\nCharging with 7 kW (should be capped at 5kW)...")
    print(f"Actual power drawn from source: {power_drawn:.2f} kW")
    print(f"New Charge: {my_battery.current_charge_kwh:.2f} kWh, SoC: {my_battery.get_state_of_charge():.0%}")

    # Discharge 4 kW
    power_supplied = my_battery.discharge(4)
    print(f"\nDischarging 4 kW...")
    print(f"Actual power supplied: {power_supplied:.2f} kW")
    print(f"New Charge: {my_battery.current_charge_kwh:.2f} kWh, SoC: {my_battery.get_state_of_charge():.0%}")

    # Try to discharge 10 kW (should be capped by available charge)
    power_supplied = my_battery.discharge(10)
    print(f"\nDischarging 10 kW (should be capped by available charge)...")
    print(f"Actual power supplied: {power_supplied:.2f} kW")
    print(f"New Charge: {my_battery.current_charge_kwh:.2f} kWh, SoC: {my_battery.get_state_of_charge():.0%}")
