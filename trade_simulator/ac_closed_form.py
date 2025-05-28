import numpy as np
import math
from decimal import Decimal, getcontext

class AlmgrenChrissClosedForm:
    def __init__(self,
                 total_shares_to_sell: float,
                 starting_price: float,
                 annual_volatility_pct: float,
                 bid_ask_spread_usd: float,
                 daily_trade_volume_asset: float,
                 risk_aversion: float,        
                 liquidation_time_days: float, 
                 num_trading_intervals: int,   
                 trading_days_per_year: int = 250):

        if num_trading_intervals <= 0:
            raise ValueError("Number of trading intervals must be positive.")
        if liquidation_time_days <= 0:
            raise ValueError("Liquidation time must be positive.")
        if daily_trade_volume_asset <=0:
            daily_trade_volume_asset = 1e-9


        self.total_shares = float(total_shares_to_sell)
        self.starting_price = float(starting_price)
        
        self.annual_volat = float(annual_volatility_pct) / 100.0
        self.bid_ask_sp = float(bid_ask_spread_usd)
        self.daily_trade_vol = float(daily_trade_volume_asset)
        self.trad_days = int(trading_days_per_year)
        self.daily_volat = self.annual_volat / np.sqrt(self.trad_days) if self.trad_days > 0 else self.annual_volat

        self.llambda = float(risk_aversion)
        self.liquidation_time = float(liquidation_time_days) # Total horizon T
        self.num_n = int(num_trading_intervals)              # Number of trades N

        # Epsilon: fixed cost of selling per share (half spread)
        self.epsilon = self.bid_ask_sp / 2.0

        # Single step variance (sigma^2 * tau)
        # Original script: self.singleStepVariance = (DAILY_VOLAT * STARTING_PRICE) ** 2
        # Let's use variance of daily returns * S^2 for price variance per day.
        # sigma_sq_price_daily = (self.daily_volat**2) * (self.starting_price**2)
        # The tau in A-C context is T/N. So variance for one trading interval (tau) is:
        # sigma_sq_price_per_tau = sigma_sq_price_daily * self.tau (if tau is in days)

        self.tau = self.liquidation_time / self.num_n # Trading interval duration in days
        # Eta: temporary impact coefficient
        # ETA = BID_ASK_SP / (0.01 * DAILY_TRADE_VOL)
        if self.daily_trade_vol > 1e-12 : # Avoid division by 0
             self.eta_param = self.bid_ask_sp / (0.01 * self.daily_trade_vol)
        else:
             self.eta_param = float('inf') 

        # Gamma: permanent impact coefficient
        # GAMMA = BID_ASK_SP / (0.1 * DAILY_TRADE_VOL)
        if self.daily_trade_vol > 1e-12:
            self.gamma_param = self.bid_ask_sp / (0.1 * self.daily_trade_vol)
        else:
            self.gamma_param = float('inf')


        # Derived A-C parameters (kappa, etc.)
        # Single step variance for the stock PRICE (not return) over one interval tau
        # This is (sigma * S)^2 * tau where sigma is daily return volatility
        self.single_step_price_variance = (self.daily_volat * self.starting_price)**2 * self.tau

        # eta_hat from A-C paper for temporary impact cost calculation in shortfall
        self.eta_hat = self.eta_param - (0.5 * self.gamma_param * self.tau)

        # kappa_hat and kappa for optimal trajectory calculations
        if self.eta_hat <= 1e-12: # Avoid division by zero or sqrt of negative
            self.kappa_hat = float('inf')
            self.kappa = float('inf')
        else:
            self.kappa_hat = np.sqrt(abs(self.llambda * self.single_step_price_variance) / self.eta_hat)
            acosh_arg = (((self.kappa_hat ** 2) * (self.tau ** 2)) / 2.0) + 1.0
            if acosh_arg < 1.0: acosh_arg = 1.0 # Clamp due to potential float inaccuracies

            if self.tau > 1e-12:
                self.kappa = np.arccosh(acosh_arg) / self.tau
            else: # if tau is extremely small
                self.kappa = float('inf') 


    def get_expected_shortfall_optimal_strategy(self):
        """
        Calculates the expected shortfall for the optimal Almgren-Chriss strategy.
        This is E[X_0 * S_0 - sum(q_k * S_k_exec)]
        According to equation (20) of the Almgren and Chriss paper (2000).
        """
        if self.total_shares == 0:
            return 0.0
        if math.isinf(self.kappa) or math.isinf(self.eta_hat) or \
           math.isinf(self.gamma_param) or math.isinf(self.eta_param):
            # If kappa is inf, implies immediate execution, shortfall is roughly total_shares * epsilon + permanent impact of full trade
            # This indicates extreme parameters. For a very large impact, we might return a large number.
            # A more nuanced handling could estimate single-step liquidation cost.
            if self.eta_hat <= 1e-9 and self.total_shares > 0 : # Extremely high temp impact.
                 return float('inf') # Cannot calculate meaningfully.
            # Fallback for other inf parameters, assuming it means high cost.
            # This typically suggests an issue with input parameters leading to extreme derived values.
            # A robust way is to consider the cost of immediate liquidation.
            # For now, if kappa is inf, it means very fast trading.
            # Cost would be total_shares * epsilon (spread) + 0.5 * gamma * total_shares^2 (permanent impact of full block)
            # + eta * total_shares (temporary impact of full block if eta is per share)
            # This part needs careful thought if parameters are truly extreme.
            # For this integration, let's return inf if key derived params are inf.
            if math.isinf(self.kappa):
                # Simplified estimate for infinite kappa (instant liquidation like)
                cost_spread = self.epsilon * self.total_shares
                cost_perm_impact = 0.5 * self.gamma_param * (self.total_shares**2)
                # Temp impact cost is harder here; A-C form is (eta_hat/tau) * sum(q_k^2)
                # If instant, q_0 = X_0, tau -> 0. (eta_hat/tau) * X_0^2 becomes large.
                # Let's use the simple formula for now, it might give inf if kappa_hat was inf due to eta_hat=0
                # print(f"Warning: Kappa is inf. Params: lambda={self.llambda}, sspv={self.single_step_price_variance}, eta_hat={self.eta_hat}")
                # return float('inf') # Simplification for now
                pass # Allow calculation to proceed, it might become inf naturally


        # Term 1: Permanent impact component based on initial total shares
        # 0.5 * gamma * X_0^2
        term1_perm_impact_total = 0.5 * self.gamma_param * (self.total_shares ** 2)
        
        # Term 2: Fixed cost per share (spread)
        # epsilon * X_0
        term2_spread_cost = self.epsilon * self.total_shares
        
        # Term 3: Temporary impact component
        # eta_hat * X_0^2 * [ (tanh(0.5*kappa*tau) * (tau*sinh(2*kappa*T) + 2*T*sinh(kappa*tau)) ) /
        #                       (2 * tau^2 * sinh(kappa*T)^2) ]
        # Let factor = [ ... ]
        term3_temp_impact_coeff = self.eta_hat * (self.total_shares ** 2)

        # Numerator of the factor for temporary impact
        # np.tanh(0.5 * self.kappa * self.tau) * (self.tau * np.sinh(2 * self.kappa * self.liquidation_time) \
        #                                               + 2 * self.liquidation_time * np.sinh(self.kappa * self.tau))
        try:
            # Guard against overflow if kappa or T are very large
            k_tau_half = 0.5 * self.kappa * self.tau
            k_T = self.kappa * self.liquidation_time

            if abs(k_tau_half) > 700 or abs(k_T) > 700: # exp(709) is approx max float
                # Handle large arguments by approximation or returning inf
                # For very large kappa*T, sinh(kappa*T) ~ 0.5*exp(kappa*T)
                # tanh tends to 1.
                # This part needs careful asymptotic analysis if triggered often.
                # For now, if arguments are too large, it implies extreme execution / high cost.
                # print(f"Warning: Large arguments in temporary impact factor. k*tau/2={k_tau_half}, k*T={k_T}")
                if math.isinf(self.kappa): # If kappa is infinite, this factor should lead to high cost
                    factor_temp_num = float('inf') if self.eta_hat > 0 else 0
                else: # Heuristic: if large but finite, calculate, might overflow to inf
                    factor_temp_num = np.tanh(k_tau_half) * \
                                  (self.tau * np.sinh(2 * k_T) + \
                                   2 * self.liquidation_time * np.sinh(self.kappa * self.tau))

            else:
                 factor_temp_num = np.tanh(0.5 * self.kappa * self.tau) * \
                                  (self.tau * np.sinh(2 * self.kappa * self.liquidation_time) + \
                                   2 * self.liquidation_time * np.sinh(self.kappa * self.tau))

            # Denominator of the factor
            # 2 * (self.tau ** 2) * (np.sinh(self.kappa * self.liquidation_time) ** 2)
            # sinh_k_T = np.sinh(self.kappa * self.liquidation_time)
            if abs(k_T) > 700 and self.kappa * self.liquidation_time > 0: # Avoid sinh(very large number)^2 if it's finite
                sinh_k_T_sq = float('inf')
            elif abs(k_T) > 700 and self.kappa * self.liquidation_time < 0: # sinh of large negative
                sinh_k_T_sq = float('inf') # (-inf)^2 = inf
            else:
                 sinh_k_T_sq = np.sinh(self.kappa * self.liquidation_time) ** 2

            if self.tau < 1e-12 or sinh_k_T_sq < 1e-12 : # Avoid division by zero or very small denominator
                 # If sinh_k_T_sq is zero (e.g. kappa*T = 0), means no decay, different formula might apply or limit.
                 # For now, treat as potentially high cost if numerator isn't also zero.
                 if abs(factor_temp_num) < 1e-9: # If num is also zero, factor is 0
                      factor_for_temp = 0.0
                 else: # Num is non-zero, Denom is zero -> Inf
                      factor_for_temp = float('inf') if factor_temp_num > 0 else float('-inf')
            else:
                factor_temp_denom = 2 * (self.tau ** 2) * sinh_k_T_sq
                if factor_temp_denom == 0: # Should be caught by sinh_k_T_sq check
                    factor_for_temp = float('inf') if factor_temp_num > 0 else (float('-inf') if factor_temp_num < 0 else 0.0)
                else:
                    factor_for_temp = factor_temp_num / factor_temp_denom
        
        except OverflowError:
            # print("Warning: OverflowError during temporary impact factor calculation.")
            factor_for_temp = float('inf') # If overflow, assume very high cost contribution


        term3_temp_impact_cost = term3_temp_impact_coeff * factor_for_temp
        
        expected_shortfall = term1_perm_impact_total + term2_spread_cost + term3_temp_impact_cost
        return float(expected_shortfall)


    def get_expected_variance_optimal_strategy(self):
        """
        Calculates the variance of execution cost for the optimal Almgren-Chriss strategy.
        According to equation (20) of the Almgren and Chriss paper (2000).
        Var[X_0 * S_0 - sum(q_k * S_k_exec)]
        """
        if self.total_shares == 0:
            return 0.0
        if math.isinf(self.kappa) or math.isinf(self.single_step_price_variance):
            # print(f"Warning: Kappa or sspv is inf in variance calc. kappa={self.kappa}, sspv={self.single_step_price_variance}")
            return float('inf') # Variance would be very high / infinite

        # Term 1 coefficient for variance: 0.5 * sigma_price_step^2 * X_0^2
        term1_var_coeff = 0.5 * self.single_step_price_variance * (self.total_shares ** 2)

        try:
            k_T = self.kappa * self.liquidation_time
            k_tau = self.kappa * self.tau
            
            if any(abs(x) > 700 for x in [k_T, k_tau, self.kappa * (self.liquidation_time - self.tau)]):
                # print(f"Warning: Large arguments in variance factor. k*T={k_T}, k*tau={k_tau}")
                # Asymptotic analysis needed for robust handling of extreme values.
                # Heuristic: if large, calculate, might overflow.
                 pass # Let it try to calculate

            # Numerator of the factor for variance
            # tau * sinh(kappa*T) * cosh(kappa*(T-tau)) - T * sinh(kappa*tau)
            factor_var_num = self.tau * np.sinh(k_T) * np.cosh(self.kappa * (self.liquidation_time - self.tau)) - \
                             self.liquidation_time * np.sinh(k_tau)

            sinh_k_T_sq = np.sinh(k_T)**2
            sinh_k_tau = np.sinh(k_tau)

            if abs(sinh_k_T_sq * sinh_k_tau) < 1e-12: # Avoid division by 0
                if abs(factor_var_num) < 1e-9: factor_for_var = 0.0
                else: factor_for_var = float('inf') if factor_var_num > 0 else float('-inf')
            else:
                factor_var_denom = sinh_k_T_sq * sinh_k_tau
                if factor_var_denom == 0:
                    factor_for_var = float('inf') if factor_var_num > 0 else (float('-inf') if factor_var_num < 0 else 0.0)
                else:
                    factor_for_var = factor_var_num / factor_var_denom

        except OverflowError:
            factor_for_var = float('inf')

        expected_variance = term1_var_coeff * factor_for_var
        return float(expected_variance)

    def get_ac_utility(self):
        """
        Calculates Almgren-Chriss Utility = E_shortfall + lambda * Var_shortfall
        """
        E = self.get_expected_shortfall_optimal_strategy()
        V = self.get_expected_variance_optimal_strategy()

        if math.isinf(E) or math.isinf(V):
            # If either is infinite, utility is likely infinite (assuming lambda > 0)
            # Or handle based on signs if negative infinities are possible (not typical for cost/variance)
            if E == float('inf') or (self.llambda > 0 and V == float('inf')):
                return float('inf')
            # More complex cases if V is -inf or lambda is negative
        
        utility = E + self.llambda * V
        return float(utility)


if __name__ == '__main__':
    # Example parameters
    TOTAL_SHARES_TEST = 1000000.0
    STARTING_PRICE_TEST = 50.0
    ANNUAL_VOLAT_TEST = 12.0 # In percent
    BID_ASK_SP_TEST = 1.0 / 8.0 # USD
    DAILY_TRADE_VOL_TEST = 5e6 # Asset units
    LLAMBDA_TEST = 1e-6
    LIQUIDATION_TIME_TEST = 60.0 # Days
    NUM_N_TEST = 60 # Number of trades

    print(f"--- Testing AlmgrenChrissClosedForm ---")
    print(f"Total Shares: {TOTAL_SHARES_TEST}, Start Price: {STARTING_PRICE_TEST}")
    print(f"Annual Vol: {ANNUAL_VOLAT_TEST}%, Bid-Ask Spread: {BID_ASK_SP_TEST:.4f} USD")
    print(f"Daily Volume: {DAILY_TRADE_VOL_TEST:.0f} shares, Risk Aversion (lambda): {LLAMBDA_TEST:.1E}")
    print(f"Liquidation Time: {LIQUIDATION_TIME_TEST} days, Num Intervals: {NUM_N_TEST}")

    ac_model = AlmgrenChrissClosedForm(
        total_shares_to_sell=TOTAL_SHARES_TEST,
        starting_price=STARTING_PRICE_TEST,
        annual_volatility_pct=ANNUAL_VOLAT_TEST,
        bid_ask_spread_usd=BID_ASK_SP_TEST,
        daily_trade_volume_asset=DAILY_TRADE_VOL_TEST,
        risk_aversion=LLAMBDA_TEST,
        liquidation_time_days=LIQUIDATION_TIME_TEST,
        num_trading_intervals=NUM_N_TEST
    )

    print(f"\nDerived parameters:")
    print(f"tau (interval duration, days): {ac_model.tau:.4f}")
    print(f"epsilon (cost per share): {ac_model.epsilon:.4f}")
    print(f"eta_param (temp. impact coeff): {ac_model.eta_param:.3E}")
    print(f"gamma_param (perm. impact coeff): {ac_model.gamma_param:.3E}")
    print(f"Daily Price Volatility (abs): {ac_model.daily_volat * ac_model.starting_price:.4f}")
    print(f"Single Step Price Var (sigma*S)^2*tau: {ac_model.single_step_price_variance:.4f}")
    print(f"eta_hat: {ac_model.eta_hat:.3E}")
    print(f"kappa: {ac_model.kappa:.4f}")

    expected_shortfall = ac_model.get_expected_shortfall_optimal_strategy()
    expected_variance = ac_model.get_expected_variance_optimal_strategy()
    utility = ac_model.get_ac_utility()

    print(f"\nExpected Shortfall (Optimal Strategy): {expected_shortfall:,.2f} USD")
    print(f"Expected Variance (Optimal Strategy): {expected_variance:,.2f} (USD^2)")
    print(f"Utility (E + lambda*V): {utility:,.2f} USD")

    # Test with slightly different parameters
    print("\n--- Test with higher risk aversion (lambda = 1e-5) ---")
    LLAMBDA_TEST_2 = 1e-5
    ac_model_2 = AlmgrenChrissClosedForm(
        total_shares_to_sell=TOTAL_SHARES_TEST, starting_price=STARTING_PRICE_TEST,
        annual_volatility_pct=ANNUAL_VOLAT_TEST, bid_ask_spread_usd=BID_ASK_SP_TEST,
        daily_trade_volume_asset=DAILY_TRADE_VOL_TEST, risk_aversion=LLAMBDA_TEST_2,
        liquidation_time_days=LIQUIDATION_TIME_TEST, num_trading_intervals=NUM_N_TEST
    )
    print(f"kappa (lambda={LLAMBDA_TEST_2:.1E}): {ac_model_2.kappa:.4f}")
    E2 = ac_model_2.get_expected_shortfall_optimal_strategy()
    V2 = ac_model_2.get_expected_variance_optimal_strategy()
    U2 = ac_model_2.get_ac_utility()
    print(f"Expected Shortfall: {E2:,.2f} USD, Expected Variance: {V2:,.2f} (USD^2), Utility: {U2:,.2f} USD")

    print("\n--- Test with shorter liquidation time (30 days) ---")
    LIQUIDATION_TIME_TEST_3 = 30.0
    NUM_N_TEST_3 = 30
    ac_model_3 = AlmgrenChrissClosedForm(
        total_shares_to_sell=TOTAL_SHARES_TEST, starting_price=STARTING_PRICE_TEST,
        annual_volatility_pct=ANNUAL_VOLAT_TEST, bid_ask_spread_usd=BID_ASK_SP_TEST,
        daily_trade_volume_asset=DAILY_TRADE_VOL_TEST, risk_aversion=LLAMBDA_TEST,
        liquidation_time_days=LIQUIDATION_TIME_TEST_3, num_trading_intervals=NUM_N_TEST_3
    )
    print(f"kappa (T={LIQUIDATION_TIME_TEST_3}, N={NUM_N_TEST_3}): {ac_model_3.kappa:.4f}")
    E3 = ac_model_3.get_expected_shortfall_optimal_strategy()
    V3 = ac_model_3.get_expected_variance_optimal_strategy()
    U3 = ac_model_3.get_ac_utility()
    print(f"Expected Shortfall: {E3:,.2f} USD, Expected Variance: {V3:,.2f} (USD^2), Utility: {U3:,.2f} USD")