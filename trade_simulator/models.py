import logging
from decimal import Decimal, getcontext, ROUND_HALF_UP
import math
import config
from orderbook import OrderBook 
from ac_dynamic_programming import get_optimal_execution_trajectory_and_cost
from ac_closed_form import AlmgrenChrissClosedForm 

getcontext().prec = config.DECIMAL_PRECISION

class TradeCalculators:
    def __init__(self, order_book: OrderBook):
        self.order_book = order_book

    def calculate_avg_fill_price_and_qty(self, quantity_usd_equivalent, side):
        mid_price = self.order_book.get_mid_price()
        if not mid_price or mid_price <= Decimal("0"):
            return None, Decimal("0"), Decimal("0")

        initial_ref_price = self.order_book.get_best_ask() if side == "BUY" else self.order_book.get_best_bid()
        if not initial_ref_price or initial_ref_price <= Decimal("0"): initial_ref_price = mid_price
        if not initial_ref_price or initial_ref_price <= Decimal("0"): return None, Decimal("0"), Decimal("0")

        target_quantity_asset = quantity_usd_equivalent / initial_ref_price
        if target_quantity_asset == Decimal("0"): return None, Decimal("0"), Decimal("0")

        filled_quantity_asset = Decimal("0")
        total_quote_qty = Decimal("0")
        
        book_levels = self.order_book.asks if side == "BUY" else self.order_book.bids
        if not book_levels:
            return None, Decimal("0"), Decimal("0")

        for price, available_qty_asset in book_levels:
            if filled_quantity_asset >= target_quantity_asset:
                break
            qty_to_take_from_level = min(target_quantity_asset - filled_quantity_asset, available_qty_asset)
            filled_quantity_asset += qty_to_take_from_level
            total_quote_qty += qty_to_take_from_level * price
        
        if filled_quantity_asset == Decimal("0"):
            return None, Decimal("0"), Decimal("0")
            
        avg_fill_price = total_quote_qty / filled_quantity_asset
        return avg_fill_price, filled_quantity_asset, total_quote_qty

    def calculate_slippage_regression(self, quantity_usd: Decimal, side: str, annual_volatility_pct: Decimal):
        mid_price = self.order_book.get_mid_price()
        best_price_ideal = self.order_book.get_best_ask() if side == "BUY" else self.order_book.get_best_bid()

        # Initial logging of inputs and basic market state
        logging.debug(f"[SLIPPAGE_REG] Inputs: QtyUSD={quantity_usd}, Side={side}, AnnVol%={annual_volatility_pct}")
        if mid_price:
            logging.debug(f"[SLIPPAGE_REG] MidPrice: {mid_price:.4f}, BestAsk: {self.order_book.get_best_ask()}, BestBid: {self.order_book.get_best_bid()}")
        else:
            logging.debug(f"[SLIPPAGE_REG] MidPrice: None, BestAsk: {self.order_book.get_best_ask()}, BestBid: {self.order_book.get_best_bid()}")

        if not mid_price or not best_price_ideal or mid_price <= Decimal("0") or best_price_ideal <= Decimal("0"):
            logging.warning("[SLIPPAGE_REG] Cannot calculate slippage due to missing or invalid order book price data (mid, best bid/ask).")
            return Decimal("0"), Decimal("0"), None # Cannot calculate

        # --- Feature Calculation ---

        # FEATURE 1: Inverse Depth Proxy
        # depth_proxy_asset sums asset quantity within config.DEPTH_PROXY_PERCENTAGE of mid-price on the relevant side of the book.
        depth_proxy_asset = self.order_book.get_depth_proxy(side, mid_price, config.DEPTH_PROXY_PERCENTAGE)
        
        # If depth_proxy_asset is very small, inverse_depth_proxy becomes large. Capped at 1,000,000.
        inverse_depth_proxy = Decimal("1") / depth_proxy_asset if depth_proxy_asset > Decimal("0.000001") else Decimal("1000000") 
        logging.debug(f"[SLIPPAGE_REG] Feature [Depth]: depth_proxy_asset (within {config.DEPTH_PROXY_PERCENTAGE*100}% of mid): {depth_proxy_asset:.8f}")
        logging.debug(f"[SLIPPAGE_REG] Feature [Depth]: inverse_depth_proxy (capped at 1M): {inverse_depth_proxy:.2f}")

        # FEATURE 2: Daily Volatility (as a decimal)
        # (Annual Vol % / 100) / sqrt(trading_days_in_year)
        daily_volatility_decimal = (annual_volatility_pct / Decimal("100")) / Decimal(math.sqrt(252)) # Assuming 252 trading days
        logging.debug(f"[SLIPPAGE_REG] Feature [Volatility]: daily_volatility_decimal: {daily_volatility_decimal:.8f} (from AnnVol%: {annual_volatility_pct})")

        # FEATURE 3: Quantity USD (input 'quantity_usd')
        logging.debug(f"[SLIPPAGE_REG] Feature [Size]: quantity_usd: {quantity_usd:.2f}")


        # --- Linear Regression Term Calculation ---
        term_intercept = config.SLIPPAGE_REG_INTERCEPT
        term_depth_contribution = config.SLIPPAGE_REG_COEF_DEPTH_INV * inverse_depth_proxy
        term_size_contribution = config.SLIPPAGE_REG_COEF_SIZE_USD * quantity_usd
        term_vol_contribution = config.SLIPPAGE_REG_COEF_VOL_DAILY * daily_volatility_decimal
        
        logging.debug(f"[SLIPPAGE_REG] Term [Intercept]: {term_intercept:.8f} (Coef: {config.SLIPPAGE_REG_INTERCEPT})")
        logging.debug(f"[SLIPPAGE_REG] Term [DepthImpact]: {term_depth_contribution:.8f} (Coef: {config.SLIPPAGE_REG_COEF_DEPTH_INV}, InvDepth: {inverse_depth_proxy:.2f})")
        logging.debug(f"[SLIPPAGE_REG] Term [SizeImpact]: {term_size_contribution:.8f} (Coef: {config.SLIPPAGE_REG_COEF_SIZE_USD}, QtyUSD: {quantity_usd:.2f})")
        logging.debug(f"[SLIPPAGE_REG] Term [VolImpact]: {term_vol_contribution:.8f} (Coef: {config.SLIPPAGE_REG_COEF_VOL_DAILY}, DailyVolDec: {daily_volatility_decimal:.8f})")

        # Sum of terms to get the raw decimal slippage prediction
        slippage_pct_predicted_decimal = term_intercept + term_depth_contribution + term_size_contribution + term_vol_contribution
        logging.debug(f"[SLIPPAGE_REG] Intermediate: slippage_pct_predicted_decimal (sum of terms): {slippage_pct_predicted_decimal:.8f}")
                                         
        # Convert to percentage and ensure it's not negative
        slippage_pct_predicted = max(Decimal("0"), slippage_pct_predicted_decimal * Decimal("100")) 
        logging.debug(f"[SLIPPAGE_REG] FINAL PREDICTED SLIPPAGE (%): {slippage_pct_predicted:.4f}")

        # Calculate average fill price by walking the book (actual slippage if order was fully aggressive)
        # This is independent of the regression model but useful for comparison or as another output.
        avg_fill_price_walk, filled_quantity_asset_walk, _ = self.calculate_avg_fill_price_and_qty(quantity_usd, side)
        if avg_fill_price_walk:
             logging.debug(f"[SLIPPAGE_REG] Book Walk Avg Fill Price: {avg_fill_price_walk:.4f} for {filled_quantity_asset_walk:.4f} asset")
        else:
             logging.debug(f"[SLIPPAGE_REG] Book Walk could not determine fill price.")
        
        # Slippage cost in USD based on the *predicted decimal slippage* from the regression
        slippage_cost_usd_predicted = slippage_pct_predicted_decimal * quantity_usd
        # Ensure cost is not negative if slippage_pct_predicted_decimal was negative before max(0, ...)
        if slippage_pct_predicted_decimal < Decimal("0"):
            slippage_cost_usd_predicted = Decimal("0") 

        logging.debug(f"[SLIPPAGE_REG] Predicted Slippage Cost (USD): {slippage_cost_usd_predicted:.4f}")
        
        return slippage_pct_predicted, slippage_cost_usd_predicted, avg_fill_price_walk


    def calculate_fees(self, avg_fill_price, filled_quantity_asset, fee_rate):
        if avg_fill_price is None or filled_quantity_asset is None or fee_rate is None:
            return Decimal("0")
        if avg_fill_price <= Decimal("0") or filled_quantity_asset <= Decimal("0"):
            return Decimal("0")
        fee_cost_usd = avg_fill_price * filled_quantity_asset * fee_rate
        return fee_cost_usd

    def calculate_market_impact_ac_simple(self, quantity_usd, side, annual_volatility_pct, assumed_daily_volume_asset):
        mid_price = self.order_book.get_mid_price()
        if not mid_price or mid_price <= Decimal("0") or quantity_usd <= Decimal("0"):
            return Decimal("0")

        if assumed_daily_volume_asset <= Decimal("0"):
            logging.warning("Assumed daily volume for A-C (Simple) model is zero or negative. Resulting in high/infinite impact.")
            return Decimal("Infinity") 

        quantity_asset = quantity_usd / mid_price
        daily_vol_fraction = (annual_volatility_pct / Decimal("100")) / Decimal(math.sqrt(252))
        
        if quantity_asset == Decimal("0"): return Decimal("0")

        ratio_qv = quantity_asset / assumed_daily_volume_asset
        
        power_term = Decimal("0")
        if ratio_qv < Decimal("0"): 
            logging.warning(f"A-C (Simple) model: Negative Q/V ratio ({ratio_qv}). Impact set to 0.")
        elif ratio_qv == Decimal("0"):
            if config.AC_DELTA_SIMPLE == Decimal("0"): power_term = Decimal("1") 
            elif config.AC_DELTA_SIMPLE > Decimal("0"): power_term = Decimal("0")
            else: 
                logging.warning(f"A-C (Simple) model: Q/V is 0 and delta is negative ({config.AC_DELTA_SIMPLE}). Impact set to Infinity.")
                return Decimal("Infinity")
        else: 
             try:
                power_term = ratio_qv ** config.AC_DELTA_SIMPLE 
             except Exception as e: 
                logging.error(f"Error in A-C (Simple) power calculation ({ratio_qv} ** {config.AC_DELTA_SIMPLE}): {e}. Impact set to 0.")
                return Decimal("0")

        relative_price_change = config.AC_ETA_SIMPLE * daily_vol_fraction * power_term 
        abs_price_change = relative_price_change * mid_price
        market_impact_cost_usd = abs_price_change * quantity_asset
        
        return max(Decimal("0"), market_impact_cost_usd)

    def calculate_market_impact_ac_dp(self, quantity_usd: Decimal, mid_price: Decimal, 
                                      annual_volatility_pct: Decimal, risk_aversion: Decimal,
                                      num_time_steps: int):
        if mid_price is None or mid_price <= Decimal("0") or quantity_usd <= Decimal("0"):
            logging.warning("A-C DP: Invalid mid_price or quantity_usd for calculation.")
            return Decimal("0.0") 
        if num_time_steps <= 0:
            logging.warning("A-C DP: num_time_steps must be positive.")
            return Decimal("0.0")

        total_shares_asset = quantity_usd / mid_price
        total_shares_int = int(round(float(total_shares_asset))) 
        """
        logging.info(f"[A-C DP DEBUG] quantity_usd: {quantity_usd}, mid_price: {mid_price}")
        logging.info(f"[A-C DP DEBUG] total_shares_asset: {total_shares_asset:.8f}")
        logging.info(f"[A-C DP DEBUG] total_shares_int (for DP): {total_shares_int}")
        """
        
        if total_shares_int <= 0:
            return Decimal("0.0")

        risk_aversion_float = float(risk_aversion)
        alpha_float = float(config.AC_DP_ALPHA)
        beta_float = float(config.AC_DP_BETA)
        gamma_float = float(config.AC_DP_GAMMA) 
        eta_float = float(config.AC_DP_ETA)     
        
        total_exec_horizon_days_float = float(config.DEFAULT_AC_DP_TOTAL_EXECUTION_HORIZON_DAYS)

        daily_vol_decimal = (annual_volatility_pct / Decimal("100")) / Decimal(math.sqrt(252))
        
        time_step_duration_days_float = 0.0
        if num_time_steps > 0:
            time_step_duration_days_float = total_exec_horizon_days_float / float(num_time_steps)
        
        volatility_per_step_float = 0.0
        if time_step_duration_days_float > 0:
             volatility_per_step_float = float(daily_vol_decimal) * math.sqrt(time_step_duration_days_float)
        '''
        logging.info( # Changed from debug for visibility during testing
            f"[A-C DP DEBUG PARAMS] TotalShares={total_shares_int}, Steps={num_time_steps}, "
            f"RiskAv={risk_aversion_float:.4f}, Alpha={alpha_float:.2f}, Beta={beta_float:.2f}, "
            f"Gamma={gamma_float:.3E}, Eta={eta_float:.3E}, "
            f"AnnualVol%={annual_volatility_pct}, Vol/Step={volatility_per_step_float:.6f}, "
            f"StepDurDays={time_step_duration_days_float:.4f}, TotalHorizDays={total_exec_horizon_days_float}"
        )
        '''
        try:
            total_impact_cost_usd_float, trajectory, inventory_path = get_optimal_execution_trajectory_and_cost(
                num_time_steps=num_time_steps,
                total_shares_to_liquidate=total_shares_int,
                risk_aversion=risk_aversion_float,
                alpha=alpha_float,
                beta=beta_float,
                gamma=gamma_float,
                eta=eta_float,
                volatility_per_step=volatility_per_step_float,
                time_step_duration=time_step_duration_days_float
            )
            
            if math.isinf(total_impact_cost_usd_float) or math.isnan(total_impact_cost_usd_float):
                logging.warning(f"A-C DP resulted in non-finite cost: {total_impact_cost_usd_float}")
                return Decimal("Infinity")
            
            total_impact_cost_usd_float = max(0.0, total_impact_cost_usd_float)
            return Decimal(str(total_impact_cost_usd_float))

        except Exception as e:
            logging.error(f"Error during A-C DP calculation: {e}", exc_info=True)
            return Decimal("Infinity") 

    def calculate_market_impact_ac_closed_form(
            self, quantity_usd: Decimal, mid_price: Decimal,
            annual_volatility_pct: Decimal,
            bid_ask_spread_usd: Decimal,
            daily_trade_volume_asset: Decimal,
            risk_aversion_lambda: Decimal, 
            liquidation_time_days: Decimal, 
            num_trading_intervals: int 
        ):
        if mid_price is None or mid_price <= Decimal("0") or quantity_usd <= Decimal("0"):
            logging.warning("A-C CF: Invalid mid_price or quantity_usd for calculation.")
            return Decimal("0.0")
        if daily_trade_volume_asset <= Decimal("0"): # Handled inside class too, but good to check early
            logging.warning("A-C CF: Daily trade volume must be positive.")
            return Decimal("Infinity") 
        if num_trading_intervals <= 0 or liquidation_time_days <= Decimal("0"):
            logging.warning("A-C CF: Liquidation time and number of intervals must be positive.")
            return Decimal("0.0")

        total_shares_asset_float = float(quantity_usd / mid_price)
        
        if total_shares_asset_float <= 1e-9: 
            logging.info("A-C CF: Asset quantity is effectively zero. Shortfall is zero.")
            return Decimal("0.0")
        '''
        logging.info(
            f"[A-C CF DEBUG PARAMS] TotalShares={total_shares_asset_float:.4f}, MidPrice={float(mid_price):.2f}, "
            f"AnnVol%={float(annual_volatility_pct):.2f}, SpreadUSD={float(bid_ask_spread_usd):.4f}, "
            f"DailyVolAsset={float(daily_trade_volume_asset):.2f}, RiskAv(Lambda)={float(risk_aversion_lambda):.2E}, "
            f"LiqDays={float(liquidation_time_days):.2f}, NumIntervals={num_trading_intervals}"
        )
        '''
        try:
            ac_model_cf = AlmgrenChrissClosedForm(
                total_shares_to_sell=total_shares_asset_float,
                starting_price=float(mid_price),
                annual_volatility_pct=float(annual_volatility_pct),
                bid_ask_spread_usd=float(bid_ask_spread_usd),
                daily_trade_volume_asset=float(daily_trade_volume_asset),
                risk_aversion=float(risk_aversion_lambda),
                liquidation_time_days=float(liquidation_time_days),
                num_trading_intervals=num_trading_intervals,
                trading_days_per_year=int(config.DEFAULT_AC_CF_TRADING_DAYS_PER_YEAR)
            )

            expected_shortfall_float = ac_model_cf.get_expected_shortfall_optimal_strategy()

            if math.isinf(expected_shortfall_float) or math.isnan(expected_shortfall_float):
                logging.warning(f"A-C CF resulted in non-finite expected shortfall: {expected_shortfall_float}")
                return Decimal("Infinity")
            
            expected_shortfall_float = max(0.0, expected_shortfall_float) # Shortfall shouldn't be negative
            return Decimal(str(expected_shortfall_float))

        except ValueError as ve: 
            logging.error(f"ValueError in A-C CF setup or calculation: {ve}", exc_info=True)
            return Decimal("0.0") # Or Decimal("Infinity")
        except Exception as e:
            logging.error(f"Error during A-C CF calculation: {e}", exc_info=True)
            return Decimal("Infinity")

    def calculate_maker_taker_logistic(self, quantity_usd: Decimal, side: str): # Added side parameter
        mid_price = self.order_book.get_mid_price()
        if not mid_price or mid_price <= Decimal("0"):
            return Decimal("50"), Decimal("50") # Default to 50/50 

        # 1. Spread
        spread_bps = self.order_book.get_spread_bps()
        if spread_bps is None: spread_bps = Decimal("1000") 

        # 2. Quantity USD (existing)

        # 3. Order Book Imbalance (OBI)
        # OBI_Ratio = (sum_qty_N_levels_on_OPPOSITE_side / sum_qty_N_levels_on_MY_PASSIVE_side)
        # If BUYING, my passive side is BID, opposite is ASK. OBI_Ratio = sum_ask_N / sum_bid_N
        # If SELLING, my passive side is ASK, opposite is BID. OBI_Ratio = sum_bid_N / sum_ask_N
        
        sum_qty_asks_N = self.order_book.get_sum_quantities_n_levels("BUY", config.LR_OBI_NUM_LEVELS) 
        sum_qty_bids_N = self.order_book.get_sum_quantities_n_levels("SELL", config.LR_OBI_NUM_LEVELS)

        obi_ratio = Decimal("1.0") # Default if one side is zero
        if side == "BUY":
            if sum_qty_bids_N > Decimal("0.000001"): # my passive side
                obi_ratio = sum_qty_asks_N / sum_qty_bids_N
            elif sum_qty_asks_N > Decimal("0.000001"): # only opposite side has qty
                obi_ratio = Decimal("1000000") # Very large ratio
        elif side == "SELL":
            if sum_qty_asks_N > Decimal("0.000001"): # my passive side
                obi_ratio = sum_qty_bids_N / sum_qty_asks_N
            elif sum_qty_bids_N > Decimal("0.000001"): # only opposite side has qty
                obi_ratio = Decimal("1000000") # Very large ratio
        
        # Clamp OBI ratio to avoid extreme values in logit
        obi_ratio_clamped = max(Decimal("0.01"), min(Decimal("100"), obi_ratio))


        # 4. Size Over Best (SOB)
        # SOB_Ratio = (OrderQuantityInAsset / QuantityAtBestOppositeLevel)
        quantity_asset = quantity_usd / mid_price # Approximation
        
        sob_ratio = Decimal("1.0") # Default
        if side == "BUY":
            best_ask_qty = self.order_book.get_best_ask_quantity()
            if best_ask_qty > Decimal("0.000001"):
                sob_ratio = quantity_asset / best_ask_qty
            elif quantity_asset > Decimal("0"): # Trying to buy but no ask qty
                 sob_ratio = Decimal("1000000") # Very large ratio, will be taker if any qty appears
        elif side == "SELL":
            best_bid_qty = self.order_book.get_best_bid_quantity()
            if best_bid_qty > Decimal("0.000001"):
                sob_ratio = quantity_asset / best_bid_qty
            elif quantity_asset > Decimal("0"): # Trying to sell but no bid qty
                 sob_ratio = Decimal("1000000")

        # Clamp SOB ratio
        sob_ratio_clamped = max(Decimal("0.01"), min(Decimal("100"), sob_ratio))


        # --- Logit Calculation ---
        logit_p_maker = (config.LR_INTERCEPT +
                         config.LR_COEF_SPREAD_BPS * spread_bps +
                         config.LR_COEF_SIZE_USD * quantity_usd +
                         config.LR_COEF_OBI_OPPOSITE_VS_PASSIVE * obi_ratio_clamped +  # Using new coefficient name
                         config.LR_COEF_SOB_SIZE_OVER_BEST_QTY * sob_ratio_clamped
                        )
        """
        # Log feature values and logit for debugging
        logging.debug(f"Maker/Taker Features: SpreadBPS={spread_bps:.2f}, QtyUSD={quantity_usd:.2f}, "
                      f"OBIRatio(Opp/MyPass)={obi_ratio_clamped:.2f} (raw: {obi_ratio:.2f}), "
                      f"SOBRatio(Order/BestOpp)={sob_ratio_clamped:.2f} (raw: {sob_ratio:.2f}), "
                      f"Logit={logit_p_maker:.4f}")
        """
        try:
            logit_p_maker_clamped = max(Decimal("-700"), min(Decimal("700"), logit_p_maker))
            prob_maker = Decimal("1") / (Decimal("1") + Decimal(math.exp(float(-logit_p_maker_clamped))))
        except OverflowError: 
            prob_maker = Decimal("0") if float(-logit_p_maker_clamped) > 0 else Decimal("1")
        except Exception as e:
            logging.error(f"Error in logistic calculation: {e}")
            prob_maker = Decimal("0.5") # Default to 50% if error

        prob_maker = max(Decimal("0"), min(Decimal("1"), prob_maker))
        maker_proportion_pct = prob_maker * Decimal("100")
        taker_proportion_pct = (Decimal("1") - prob_maker) * Decimal("100")
        
        return maker_proportion_pct, taker_proportion_pct