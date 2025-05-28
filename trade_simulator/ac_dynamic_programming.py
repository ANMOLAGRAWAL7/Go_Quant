import numpy as np
import math
from decimal import Decimal

def temporary_impact_dp(volume_rate, alpha, eta):
    # volume_rate is (shares / time_step_duration)
    return eta * (volume_rate ** alpha)

def permanent_impact_dp(volume_rate, beta, gamma):
    return gamma * (volume_rate ** beta)

def hamiltonian_dp(inventory, sell_amount, risk_aversion, 
                 alpha, beta, gamma, eta, 
                 volatility_per_step, time_step_duration):
    """
    Hamiltonian equation for the dynamic programming.
    All numeric inputs (except inventory, sell_amount) should be floats for numpy.
    time_step_duration: actual duration of one time step (e.g., in days)
    volatility_per_step: volatility corresponding to one time_step_duration
    """
    if time_step_duration <= 0:# Avoid division by zero
        return float('inf')
    if sell_amount < 0: 
        return float('inf')
    
    sell_amount = min(inventory, sell_amount)

    # Calculate trade rate (shares per unit of time_step_duration)
    # If sell_amount is 0, trade_rate is 0, impacts are 0.
    trade_rate = 0.0
    if time_step_duration > 0 and sell_amount > 0: # only calculate if selling and time passes
        trade_rate = float(sell_amount) / float(time_step_duration) # sell_amount is int, time_step_duration is float

    # Cost components from the provided script's Hamiltonian structure
    
    # Term 1: "temp_impact" in original script, resembles risk_aversion * execution_cost_current_trade_due_to_perm_impact
    # risk_aversion * q_k * PermanentImpact(q_k / tau)
    cost_term1 = 0.0
    if sell_amount > 0 and trade_rate > 0:
         cost_term1 = risk_aversion * float(sell_amount) * permanent_impact_dp(trade_rate, beta, gamma)

    # Term 2: "perm_impact" in original script, resembles risk_aversion * future_inventory_risk_due_to_temp_impact_of_current_trade
    # risk_aversion * (X_k - q_k) * tau * TemporaryImpact(q_k / tau)
    # This term relates to the impact of current trade's temporary effect on the risk/cost of remaining inventory.
    cost_term2 = 0.0
    if sell_amount > 0 and trade_rate > 0: # only if selling
        cost_term2 = risk_aversion * float(inventory - sell_amount) * float(time_step_duration) * \
                     temporary_impact_dp(trade_rate, alpha, eta)

    # Term 3: Execution Risk
    # 0.5 * risk_aversion^2 * sigma^2 * tau * (X_k - q_k)^2
    exec_risk = 0.5 * (risk_aversion ** 2) * (volatility_per_step ** 2) * \
                float(time_step_duration) * (float(inventory - sell_amount) ** 2)
    
    return cost_term1 + cost_term2 + exec_risk


def get_optimal_execution_trajectory_and_cost(
        num_time_steps, total_shares_to_liquidate, risk_aversion,
        alpha, beta, gamma, eta,
        volatility_per_step, time_step_duration):
    """
    Dynamic programming to find optimal execution trajectory and total cost.
    
    Parameters:
    - num_time_steps (int): Number of time intervals for liquidation.
    - total_shares_to_liquidate (int): Total shares (asset quantity), must be integer for array indexing.
    - risk_aversion (float): Risk aversion parameter.
    - alpha, beta, gamma, eta (float): Coefficients/exponents for impact functions.
    - volatility_per_step (float): Volatility per single time step.
    - time_step_duration (float): Duration of a single time step (e.g., fraction of a day).

    Returns:
    - total_cost (float): The minimized objective function value (log of DP value function).
    - optimal_trajectory (list of ints): Number of shares to trade at each step.
    - inventory_path (list of ints): Inventory remaining after each step.
    """
    if total_shares_to_liquidate <= 0:
        return 0.0, [], [0] * num_time_steps
    if num_time_steps <= 0:
        return 0.0, [], []
        
    # Ensure total_shares_to_liquidate is int for numpy array indexing
    total_shares_int = int(round(total_shares_to_liquidate))
    if total_shares_int <= 0:
         return 0.0, [], [0] * num_time_steps


    value_function = np.zeros((num_time_steps, total_shares_int + 1), dtype="float64")
    best_moves = np.zeros((num_time_steps, total_shares_int + 1), dtype="int") 
    
    # Terminal condition: V_T(X_T) = exp( X_T * h(X_T/tau) ) which is cost of liquidating remaining X_T in one go
    # Or, if goal is to have 0 shares at T, V_T(0)=0, V_T(X > 0) = large_penalty
    # This represents the cost of liquidating all remaining shares 'shares' in the last period.
    for shares_left in range(total_shares_int + 1):
        if shares_left == 0:
            value_function[num_time_steps - 1, shares_left] = 1.0 
            best_moves[num_time_steps - 1, shares_left] = 0
        else:
            terminal_trade_rate = 0.0
            if time_step_duration > 0:
                terminal_trade_rate = float(shares_left) / float(time_step_duration)
            
            # Cost of liquidating 'shares_left' in the last step.
            # Using a simplified terminal cost: shares_left * temp_impact(rate)
            terminal_cost_val = 0.0
            if terminal_trade_rate > 0:
                 terminal_cost_val = float(shares_left) * temporary_impact_dp(terminal_trade_rate, alpha, eta)
            
            value_function[num_time_steps - 1, shares_left] = np.exp(terminal_cost_val)
            best_moves[num_time_steps - 1, shares_left] = shares_left # Must sell all remaining

    # Backward induction
    for t in range(num_time_steps - 2, -1, -1):
        for current_inventory in range(total_shares_int + 1): # Shares currently held: 0 to total_shares_int
            if current_inventory == 0:
                value_function[t, current_inventory] = 1.0 
                best_moves[t, current_inventory] = 0
                continue

            # Initialize with selling all remaining shares 'current_inventory' in this step 't'
            # sell_now = current_inventory
            hamiltonian_val = hamiltonian_dp(current_inventory, current_inventory, risk_aversion,
                                             alpha, beta, gamma, eta,
                                             volatility_per_step, time_step_duration)
            # If sell_now = current_inventory, inventory_remaining = 0. V[t+1, 0] = 1.0 (exp(0))
            best_value_this_state = value_function[t + 1, 0] * np.exp(hamiltonian_val)
            best_sell_amount_this_state = current_inventory
            
            # Iterate over possible amounts to sell 'n' (from 0 to current_inventory - 1)
            for n_sell_now in range(current_inventory): # Test selling 0, 1, ..., current_inventory-1
                inventory_remaining = current_inventory - n_sell_now
                hamiltonian_val = hamiltonian_dp(current_inventory, n_sell_now, risk_aversion,
                                                 alpha, beta, gamma, eta,
                                                 volatility_per_step, time_step_duration)
                
                current_value_for_this_action = value_function[t + 1, inventory_remaining] * np.exp(hamiltonian_val)
                
                if current_value_for_this_action < best_value_this_state:
                    best_value_this_state = current_value_for_this_action
                    best_sell_amount_this_state = n_sell_now
            
            value_function[t, current_inventory] = best_value_this_state
            best_moves[t, current_inventory] = best_sell_amount_this_state

    # Optimal trajectory and inventory path
    inventory_path_list = [0] * num_time_steps
    optimal_trajectory_list = [0] * num_time_steps # Shares sold at each step t

    current_inv = total_shares_int
    inventory_path_list[0] = current_inv

    for t in range(num_time_steps):
        if current_inv <= 0 :
            optimal_trajectory_list[t] = 0
            if t + 1 < num_time_steps:
                 inventory_path_list[t+1] = 0
            continue

        shares_to_sell_this_step = best_moves[t, current_inv]
        optimal_trajectory_list[t] = shares_to_sell_this_step
        current_inv -= shares_to_sell_this_step
        if t + 1 < num_time_steps:
            inventory_path_list[t+1] = current_inv
        if current_inv < 0: current_inv = 0


    # Total cost is log of the value function at t=0 for total_shares
    # This assumes V = exp(Cost_to_go), so log(V) = Cost_to_go
    total_cost_from_dp = 0.0
    if value_function[0, total_shares_int] > 0: # Avoid log(0) or log(negative)
        total_cost_from_dp = np.log(value_function[0, total_shares_int])
    else: 
        total_cost_from_dp = float('inf') 

    return float(total_cost_from_dp), optimal_trajectory_list, inventory_path_list

if __name__ == "__main__":
    # Example Parameters
    num_ts = 10 
    total_inv = 50 
    risk_aversion_p = 0.001 
    
    # Impact parameters (these need to be appropriate for asset and time_step_duration)
    # The example values (0.05) are very high if time_step_duration is small or shares are many.
    # For typical crypto, these might be much smaller, e.g., 1e-7 to 1e-9 range if quantity is in asset units.
    alpha_p = 1.0  # exponent for temporary impact
    beta_p = 1.0   # exponent for permanent impact
    gamma_p = 0.0000005 # permanent impact coefficient (price change per unit of trade rate^beta)
    eta_p = 0.0000005   # temporary impact coefficient (price change per unit of trade rate^alpha)

    # Market and time parameters
    annual_vol_pct = 50.0 # 50% annual volatility
    total_exec_horizon_d = 1.0 # Liquidate over 1 day

    # Derived parameters for DP
    daily_vol_decimal = (annual_vol_pct / 100.0) / math.sqrt(252)
    time_step_dur_d = total_exec_horizon_d / float(num_ts) # Duration of one time step in days
    vol_per_step = daily_vol_decimal * math.sqrt(time_step_dur_d) # Volatility over one time step

    print(f"Total Inventory: {total_inv} shares")
    print(f"Time Steps: {num_ts}")
    print(f"Time Step Duration (days): {time_step_dur_d:.4f}")
    print(f"Volatility per step: {vol_per_step:.6f}")
    print(f"Risk Aversion: {risk_aversion_p}")
    print(f"Impact Params: alpha={alpha_p}, beta={beta_p}, gamma={gamma_p}, eta={eta_p}")

    cost, trajectory, inventory = get_optimal_execution_trajectory_and_cost(
        num_ts, total_inv, risk_aversion_p,
        alpha_p, beta_p, gamma_p, eta_p,
        vol_per_step, time_step_dur_d
    )

    print(f"\nTotal DP Objective Cost: {cost:.6f}") # This cost is in the units derived from gamma/eta
    print("Optimal Execution Trajectory (shares per step):")
    print(trajectory)
    print("Inventory Path (shares remaining after step):")
    inventory_at_start = [total_inv] + inventory[:-1] # inventory[0] is after step 0, etc.
    print(inventory_at_start)


    # --- Plotting (optional, if matplotlib is available) ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(inventory_at_start, marker='o', linestyle='-', color='blue', label='Inventory at Start of Period')
        plt.title(f'Optimal Liquidation Path (Total Cost: {cost:.4f})')
        plt.xlabel('Trading Period')
        plt.ylabel('Number of Shares')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.bar(range(len(trajectory)), trajectory, color='green', label='Shares Traded per Period')
        plt.xlabel('Trading Period')
        plt.ylabel('Shares Traded')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not found, skipping plots.")
    except Exception as e:
        print(f"Error during plotting: {e}")