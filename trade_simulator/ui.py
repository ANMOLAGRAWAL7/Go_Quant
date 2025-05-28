import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import logging
from decimal import Decimal, getcontext, InvalidOperation, ROUND_HALF_UP
import time
import asyncio
import math

import config
from utils import fetch_okx_spot_instruments_sync
from orderbook import OrderBook
from models import TradeCalculators
from websocket_client import websocket_handler

getcontext().prec = config.DECIMAL_PRECISION

# For logging latency data
LATENCY_LOG_FILE = "latency_log.csv"
latency_logger = logging.getLogger('latency_logger')
latency_logger.setLevel(logging.INFO)

if not latency_logger.handlers:
    fh = logging.FileHandler(LATENCY_LOG_FILE)
    latency_logger.addHandler(fh)
    latency_logger.propagate = False
    try:
        with open(LATENCY_LOG_FILE, 'r') as f:
            if not f.readline():
                 latency_logger.info("app_timestamp_ms,ws_recv_timestamp_ms,data_processing_latency_ms,ui_update_latency_ms,end_to_end_latency_ms,order_book_timestamp")
    except FileNotFoundError:
        latency_logger.info("app_timestamp_ms,ws_recv_timestamp_ms,data_processing_latency_ms,ui_update_latency_ms,end_to_end_latency_ms,order_book_timestamp")


class TradeSimulatorApp:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title(f"Trade Simulator (OKX L2 via {config.WEBSOCKET_URI.split('/')[2]})")
        self.root.geometry("950x950")

        self.order_book = OrderBook()
        self.calculators = TradeCalculators(self.order_book)
        self.ui_queue = queue.Queue()

        self._is_running_lock = threading.Lock()
        self._is_running_websocket = False
        self.websocket_thread = None

        self.available_spot_assets = ["BTC-USDT"]
        self._setup_ui_elements()
        self._schedule_ui_queue_check()
        self._async_fetch_instruments()

    def _get_is_running_websocket(self):
        with self._is_running_lock:
            return self._is_running_websocket

    def _set_is_running_websocket(self, value: bool):
        with self._is_running_lock:
            self._is_running_websocket = value

    def _async_fetch_instruments(self):
        def task():
            instruments = fetch_okx_spot_instruments_sync()
            self.root.after(0, self._update_asset_combobox_options, instruments)
        threading.Thread(target=task, daemon=True).start()

    def _update_asset_combobox_options(self, instruments):
        self.available_spot_assets = instruments
        self.asset_combobox['values'] = self.available_spot_assets
        default_asset = "BTC-USDT"
        if default_asset in self.available_spot_assets:
            self.asset_var.set(default_asset)
        elif self.available_spot_assets:
            self.asset_var.set(self.available_spot_assets[0])
        self.on_asset_selected()

    def _setup_ui_elements(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.LabelFrame(main_frame, text="Input Parameters", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5, ipadx=5, ipady=5)

        row_idx = 0
        ttk.Label(left_panel, text="Exchange:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.exchange_var = tk.StringVar(value="OKX")
        ttk.Entry(left_panel, textvariable=self.exchange_var, state="readonly", width=25).grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1
        ttk.Label(left_panel, text="Spot Asset (Conceptual):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.asset_var = tk.StringVar()
        self.asset_combobox = ttk.Combobox(left_panel, textvariable=self.asset_var, values=self.available_spot_assets, state="readonly", width=23)
        self.asset_combobox.grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        self.asset_combobox.bind("<<ComboboxSelected>>", self.on_asset_selected)
        row_idx += 1
        self.actual_symbol_var = tk.StringVar(value="N/A (Stream)")
        ttk.Label(left_panel, text="Processing Symbol (WS):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        ttk.Label(left_panel, textvariable=self.actual_symbol_var, relief=tk.SUNKEN, width=25).grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1
        ttk.Label(left_panel, text="Order Type:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.order_type_var = tk.StringVar(value="Market")
        ttk.Combobox(left_panel, textvariable=self.order_type_var, values=["Market"], state="readonly", width=23).grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1
        ttk.Label(left_panel, text="Side:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.side_var = tk.StringVar(value="BUY")
        ttk.Combobox(left_panel, textvariable=self.side_var, values=["BUY", "SELL"], width=23).grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1
        ttk.Label(left_panel, text="Amount (USD Eq.):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.quantity_var = tk.StringVar(value=str(config.DEFAULT_USD_EQUIVALENT_QUANTITY))
        ttk.Entry(left_panel, textvariable=self.quantity_var, width=25).grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1
        ttk.Label(left_panel, text="Volatility (Ann. %):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.volatility_var = tk.StringVar(value=str(config.DEFAULT_ANNUAL_VOLATILITY_PCT))
        ttk.Entry(left_panel, textvariable=self.volatility_var, width=25).grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1
        ttk.Label(left_panel, text="Assumed Daily Vol (Asset):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.daily_volume_var = tk.StringVar(value=str(config.DEFAULT_ASSUMED_DAILY_VOLUME_ASSET))
        ttk.Entry(left_panel, textvariable=self.daily_volume_var, width=25).grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1
        ttk.Label(left_panel, text="Fee Tier (OKX Spot Taker):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.fee_tier_var = tk.StringVar(value=config.DEFAULT_FEE_TIER_DISPLAY_NAME)
        fee_tier_display_names = list(config.FEE_TIERS_OKX_SPOT.keys())
        self.fee_tier_combobox = ttk.Combobox(left_panel, textvariable=self.fee_tier_var, values=fee_tier_display_names, state="readonly", width=40)
        self.fee_tier_combobox.grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1

        # A-C DP Inputs
        ttk.Label(left_panel, text="A-C DP Settings:", font=('Helvetica', 10, 'bold')).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(8,0))
        row_idx +=1
        ttk.Label(left_panel, text="Risk Aversion (Ψ):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.ac_dp_risk_aversion_var = tk.StringVar(value=str(config.DEFAULT_AC_DP_RISK_AVERSION))
        ttk.Entry(left_panel, textvariable=self.ac_dp_risk_aversion_var, width=25).grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1
        ttk.Label(left_panel, text="Exec. Time Steps (N):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.ac_dp_time_steps_var = tk.StringVar(value=str(config.DEFAULT_AC_DP_TIME_STEPS))
        ttk.Entry(left_panel, textvariable=self.ac_dp_time_steps_var, width=25).grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1

        # A-C CF Inputs
        ttk.Label(left_panel, text="A-C Closed-Form Settings:", font=('Helvetica', 10, 'bold')).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(8,0))
        row_idx +=1
        ttk.Label(left_panel, text="Bid-Ask Spread (USD):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.ac_cf_bid_ask_spread_var = tk.StringVar(value=str(config.DEFAULT_AC_CF_BID_ASK_SPREAD_USD))
        ttk.Entry(left_panel, textvariable=self.ac_cf_bid_ask_spread_var, width=25).grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1
        ttk.Label(left_panel, text="Daily Vol (Asset) (A-C CF):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.ac_cf_daily_vol_asset_var = tk.StringVar(value=str(config.DEFAULT_AC_CF_DAILY_TRADE_VOL_ASSET))
        ttk.Entry(left_panel, textvariable=self.ac_cf_daily_vol_asset_var, width=25).grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1
        ttk.Label(left_panel, text="Risk Aversion (λ) (A-C CF):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.ac_cf_risk_aversion_var = tk.StringVar(value=str(config.DEFAULT_AC_CF_RISK_AVERSION))
        ttk.Entry(left_panel, textvariable=self.ac_cf_risk_aversion_var, width=25).grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW, pady=2)
        row_idx += 1

        # Connect Button
        self.connect_button = ttk.Button(left_panel, text="Start WebSocket", command=self.toggle_websocket_connection)
        self.connect_button.grid(row=row_idx, column=0, columnspan=3, pady=12, sticky=tk.EW)
        left_panel.columnconfigure(1, weight=1)

        # Right Panel: Outputs
        right_panel = ttk.LabelFrame(main_frame, text="Processed Output Values", padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5, ipadx=5, ipady=5)

        self.output_vars = {}
        outputs_config = [
            ("Timestamp", ".1f", "N/A"),
            ("Best Bid", ".4f", "N/A"),
            ("Best Ask", ".4f", "N/A"),
            ("Mid Price", ".4f", "N/A"),
            ("Spread (BPS)", ".2f", "N/A"),
            ("Est. Avg Fill Price (Book Walk)", ".4f", "N/A"),
            ("Slippage Predicted (%) (LinReg)", ".4f", "N/A"),
            ("Slippage Cost Predicted (USD) (LinReg)", ".4f", "N/A"),
            ("Fee Cost (USD)", ".4f", "N/A"),
            ("Market Impact (USD) (A-C Simple)", ".4f", "N/A"),
            ("Market Impact (USD) (A-C DP)", ".4f", "N/A"),
            ("Exp. Shortfall (USD) (A-C CF)", ".4f", "N/A"),
            ("Net Cost (USD) (Excl. DP & CF)", ".4f", "N/A"),
            ("Maker Proportion (%) (LogReg)", ".2f", "N/A"),
            ("Taker Proportion (%) (LogReg)", ".2f", "N/A"),
            ("Data Processing Latency (ms)", ".2f", "N/A"),
            ("UI Update Latency (ms)", ".2f", "N/A"),       
            ("End-to-End Latency (ms)", ".2f", "N/A")    
        ]
        for i, (label_text, _, default_val) in enumerate(outputs_config):
            ttk.Label(right_panel, text=f"{label_text}:").grid(row=i, column=0, sticky=tk.W, pady=1) # reduced pady
            self.output_vars[label_text] = tk.StringVar(value=default_val)
            ttk.Label(right_panel, textvariable=self.output_vars[label_text], width=30, anchor=tk.W, relief=tk.SUNKEN, borderwidth=1).grid(row=i, column=1, sticky=tk.EW, pady=1, padx=2) # reduced pady
        right_panel.columnconfigure(1, weight=1)

    def on_asset_selected(self, event=None):
        selected_asset = self.asset_var.get()
        logging.info(f"UI conceptual Spot Asset selected: {selected_asset}. Note: WebSocket stream is fixed by config.")

    def _validate_inputs(self):
        try:
            qty_str = self.quantity_var.get()
            vol_str = self.volatility_var.get()
            daily_vol_asset_str = self.daily_volume_var.get()
            ac_dp_risk_str = self.ac_dp_risk_aversion_var.get()
            ac_dp_steps_str = self.ac_dp_time_steps_var.get()
            ac_cf_spread_str = self.ac_cf_bid_ask_spread_var.get()
            ac_cf_daily_vol_str = self.ac_cf_daily_vol_asset_var.get()
            ac_cf_risk_str = self.ac_cf_risk_aversion_var.get()

            if not all([qty_str, vol_str, daily_vol_asset_str, ac_dp_risk_str, ac_dp_steps_str,
                        ac_cf_spread_str, ac_cf_daily_vol_str, ac_cf_risk_str]):
                 messagebox.showerror("Input Error", "All numeric fields must be filled.")
                 return False

            qty = Decimal(qty_str)
            vol = Decimal(vol_str)
            daily_vol_asset = Decimal(daily_vol_asset_str)
            ac_dp_risk = Decimal(ac_dp_risk_str)
            ac_dp_steps = int(ac_dp_steps_str)
            ac_cf_spread = Decimal(ac_cf_spread_str)
            ac_cf_daily_vol = Decimal(ac_cf_daily_vol_str)
            ac_cf_risk = Decimal(ac_cf_risk_str)

            if qty <= Decimal("0"): messagebox.showerror("Input Error", "Quantity USD must be positive."); return False
            if vol < Decimal("0"): messagebox.showerror("Input Error", "Volatility cannot be negative."); return False
            if daily_vol_asset <= Decimal("0"): messagebox.showerror("Input Error", "Assumed Daily Volume (Simple A-C) must be positive."); return False
            if ac_dp_risk < Decimal("0"): messagebox.showerror("Input Error", "A-C DP Risk Aversion cannot be negative."); return False
            if ac_dp_steps <= 0: messagebox.showerror("Input Error", "A-C DP/CF Time Steps must be positive integer."); return False
            if ac_dp_steps > 100:
                if not messagebox.askyesno("Warning", f"A-C DP/CF Time Steps ({ac_dp_steps}) is large and may be slow. Continue?"):
                    return False
            if ac_cf_spread < Decimal("0"): messagebox.showerror("Input Error", "A-C CF Bid-Ask Spread cannot be negative."); return False
            if ac_cf_daily_vol <= Decimal("0"): messagebox.showerror("Input Error", "A-C CF Daily Volume must be positive."); return False
            if ac_cf_risk < Decimal("0"): messagebox.showerror("Input Error", "A-C CF Risk Aversion (λ) cannot be negative."); return False
            if self.fee_tier_var.get() not in config.FEE_TIERS_OKX_SPOT:
                messagebox.showerror("Input Error", "Invalid Fee Tier selection."); return False
            return True
        except (InvalidOperation, ValueError) as e:
            messagebox.showerror("Input Error", f"Invalid number format or type: {e}")
            return False
        except Exception as e:
            messagebox.showerror("Input Error", f"Unexpected input validation error: {e}")
            return False

    def toggle_websocket_connection(self):
        if self._get_is_running_websocket():
            logging.info("User requested WebSocket stop.")
            self._set_is_running_websocket(False)
            self.connect_button.config(text="Start WebSocket", state=tk.NORMAL)
            self._reset_output_fields()
            self.actual_symbol_var.set("N/A (Stream)")
        else:
            if not self._validate_inputs():
                return
            logging.info("User requested WebSocket start.")
            self._set_is_running_websocket(True)
            self.connect_button.config(text="Stop WebSocket", state=tk.NORMAL)
            self.actual_symbol_var.set("Connecting...")
            self.websocket_thread = threading.Thread(target=self._run_websocket_event_loop, daemon=True)
            self.websocket_thread.start()

    def _run_websocket_event_loop(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(websocket_handler(
                self.order_book, self.ui_queue, self._get_is_running_websocket))
        except Exception as e:
            logging.error(f"Asyncio event loop in WebSocket thread crashed: {e}", exc_info=True)
            self.ui_queue.put({"type": "error", "data": f"WS Thread Crash: {str(e)[:100]}"})
            self.root.after(0, lambda: self.connect_button.config(text="Start WebSocket", state=tk.NORMAL))
            self.root.after(0, lambda: self.actual_symbol_var.set("WS Error!"))

    def _process_book_update_notification(self, timestamp_ws_recv):
        app_timestamp_ms = int(time.time() * 1000) 

        if not self._get_is_running_websocket():
            return

        try:
            quantity_usd = Decimal(self.quantity_var.get())
            side = self.side_var.get()
            annual_volatility_pct = Decimal(self.volatility_var.get())
            assumed_daily_volume_asset = Decimal(self.daily_volume_var.get())
            selected_fee_tier_display_name = self.fee_tier_var.get()
            fee_rate = config.FEE_TIERS_OKX_SPOT[selected_fee_tier_display_name]["rate"]
            ac_dp_risk_aversion = Decimal(self.ac_dp_risk_aversion_var.get())
            ac_num_time_steps = int(self.ac_dp_time_steps_var.get())
            ac_cf_bid_ask_spread = Decimal(self.ac_cf_bid_ask_spread_var.get())
            ac_cf_daily_vol_asset = Decimal(self.ac_cf_daily_vol_asset_var.get())
            ac_cf_risk_aversion_lambda = Decimal(self.ac_cf_risk_aversion_var.get())
        except (InvalidOperation, KeyError, ValueError) as e:
            logging.error(f"Error reading/parsing UI parameters for processing: {e}")
            self.output_vars["End-to-End Latency (ms)"].set(f"Input Error!") 
            return
        except Exception as e:
            logging.exception("Unexpected error gathering inputs for processing.")
            self.output_vars["End-to-End Latency (ms)"].set(f"Input Error!")
            return

        # --- Perform Calculations ---
        best_bid = self.order_book.get_best_bid()
        best_ask = self.order_book.get_best_ask()
        mid_price = self.order_book.get_mid_price()
        spread_bps = self.order_book.get_spread_bps()

        slippage_pct_regr, slippage_cost_regr, avg_fill_price_walk = \
            self.calculators.calculate_slippage_regression(quantity_usd, side, annual_volatility_pct)
        _, filled_quantity_asset_walk, _ = self.calculators.calculate_avg_fill_price_and_qty(quantity_usd, side)
        fee_cost = self.calculators.calculate_fees(avg_fill_price_walk, filled_quantity_asset_walk, fee_rate)
        market_impact_cost_simple_ac = self.calculators.calculate_market_impact_ac_simple(
            quantity_usd, side, annual_volatility_pct, assumed_daily_volume_asset)
        market_impact_cost_dp_ac = self.calculators.calculate_market_impact_ac_dp(
            quantity_usd=quantity_usd, mid_price=mid_price,
            annual_volatility_pct=annual_volatility_pct, risk_aversion=ac_dp_risk_aversion,
            num_time_steps=ac_num_time_steps)
        exec_horizon_days_for_cf = config.DEFAULT_AC_DP_TOTAL_EXECUTION_HORIZON_DAYS
        exp_shortfall_ac_cf = self.calculators.calculate_market_impact_ac_closed_form(
            quantity_usd=quantity_usd, mid_price=mid_price,
            annual_volatility_pct=annual_volatility_pct, bid_ask_spread_usd=ac_cf_bid_ask_spread,
            daily_trade_volume_asset=ac_cf_daily_vol_asset, risk_aversion_lambda=ac_cf_risk_aversion_lambda,
            liquidation_time_days=exec_horizon_days_for_cf, num_trading_intervals=ac_num_time_steps)
        net_cost = slippage_cost_regr + fee_cost + market_impact_cost_simple_ac
        maker_prop_pct, taker_prop_pct = self.calculators.calculate_maker_taker_logistic(quantity_usd,side)
        
        calculations_done_time = time.perf_counter()
        data_processing_latency_ms = (calculations_done_time - timestamp_ws_recv) * 1000

        # --- Prepare data for UI update ---
        outputs_config_dict = {
            "Timestamp": (self.order_book.timestamp, None),
            "Best Bid": (best_bid, ".4f"), "Best Ask": (best_ask, ".4f"),
            "Mid Price": (mid_price, ".4f"), "Spread (BPS)": (spread_bps, ".2f"),
            "Est. Avg Fill Price (Book Walk)": (avg_fill_price_walk, ".4f"),
            "Slippage Predicted (%) (LinReg)": (slippage_pct_regr, ".4f"),
            "Slippage Cost Predicted (USD) (LinReg)": (slippage_cost_regr, ".4f"),
            "Fee Cost (USD)": (fee_cost, ".4f"),
            "Market Impact (USD) (A-C Simple)": (market_impact_cost_simple_ac, ".4f"),
            "Market Impact (USD) (A-C DP)": (market_impact_cost_dp_ac, ".4f"),
            "Exp. Shortfall (USD) (A-C CF)": (exp_shortfall_ac_cf, ".4f"),
            "Net Cost (USD) (Excl. DP & CF)": (net_cost, ".4f"),
            "Maker Proportion (%) (LogReg)": (maker_prop_pct, ".2f"),
            "Taker Proportion (%) (LogReg)": (taker_prop_pct, ".2f"),
            "Data Processing Latency (ms)": (Decimal(data_processing_latency_ms), ".2f")
            # UI Update and End-to-End will be added after UI update itself
        }

        ui_update_start_time = time.perf_counter()
        for key, (value, fmt_spec) in outputs_config_dict.items():
            if key in self.output_vars:
                if value is None: self.output_vars[key].set("N/A")
                elif isinstance(value, float) and math.isinf(value): self.output_vars[key].set("Infinity")
                elif value == Decimal("Infinity"): self.output_vars[key].set("Infinity")
                elif isinstance(value, (Decimal, float)) and fmt_spec: self.output_vars[key].set(f"{value:{fmt_spec}}")
                else: self.output_vars[key].set(str(value))
        if self.order_book.symbol: self.actual_symbol_var.set(self.order_book.symbol)
        ui_update_done_time = time.perf_counter()
        
        ui_update_latency_ms = (ui_update_done_time - ui_update_start_time) * 1000
        end_to_end_latency_ms = (ui_update_done_time - timestamp_ws_recv) * 1000

        # Update the specific latency fields
        if "UI Update Latency (ms)" in self.output_vars:
            self.output_vars["UI Update Latency (ms)"].set(f"{ui_update_latency_ms:.2f}")
        if "End-to-End Latency (ms)" in self.output_vars:
            self.output_vars["End-to-End Latency (ms)"].set(f"{end_to_end_latency_ms:.2f}")

        # Log latency data
        ws_recv_timestamp_ms_for_log = int(timestamp_ws_recv * 1000)
        ob_ts = self.order_book.timestamp if self.order_book.timestamp else 'N/A'
        latency_logger.info(
            f"{app_timestamp_ms},{ws_recv_timestamp_ms_for_log},"
            f"{data_processing_latency_ms:.4f},{ui_update_latency_ms:.4f},{end_to_end_latency_ms:.4f},"
            f"{ob_ts}"
        )


    def _schedule_ui_queue_check(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                msg_type = msg.get("type")
                msg_data = msg.get("data")

                if msg_type == "book_update_notification":
                    self._process_book_update_notification(msg["timestamp_ws_recv"])
                elif msg_type == "status":
                    logging.info(f"UI Status Update: {msg_data}")
                    if "stream_symbol" in msg: self.actual_symbol_var.set(msg["stream_symbol"])
                    if msg_data == "Disconnected":
                        if self.connect_button['text'] == "Stop WebSocket":
                             self.connect_button.config(text="Start WebSocket", state=tk.NORMAL)
                        self.actual_symbol_var.set("N/A (Stream)")
                        self._reset_output_fields()
                    elif msg_data == "Reconnecting...":
                         self.actual_symbol_var.set("Reconnecting...")
                elif msg_type == "error":
                    logging.error(f"Error message from worker thread: {msg_data}")
                    messagebox.showerror("Background Process Error", str(msg_data))
                    if self._get_is_running_websocket():
                        self._set_is_running_websocket(False)
                        self.connect_button.config(text="Start WebSocket", state=tk.NORMAL)
                        self.actual_symbol_var.set("Error! Disconnected.")
                        self._reset_output_fields()
                else:
                    logging.warning(f"Unknown message type in UI queue: {msg_type}")
        except queue.Empty:
            pass
        except Exception as e:
            logging.exception("Error in UI queue processing loop.")
        self.root.after(10, self._schedule_ui_queue_check)

    def _reset_output_fields(self):
        defaults = {
            "Timestamp": "N/A", "Best Bid": "N/A", "Best Ask": "N/A",
            "Mid Price": "N/A", "Spread (BPS)": "N/A",
            "Est. Avg Fill Price (Book Walk)": "N/A",
            "Slippage Predicted (%) (LinReg)": "N/A",
            "Slippage Cost Predicted (USD) (LinReg)": "N/A",
            "Fee Cost (USD)": "N/A",
            "Market Impact (USD) (A-C Simple)": "N/A",
            "Market Impact (USD) (A-C DP)": "N/A",
            "Exp. Shortfall (USD) (A-C CF)": "N/A",
            "Net Cost (USD) (Excl. DP & CF)": "N/A",
            "Maker Proportion (%) (LogReg)": "N/A",
            "Taker Proportion (%) (LogReg)": "N/A",
            "Data Processing Latency (ms)": "N/A",
            "UI Update Latency (ms)": "N/A",
            "End-to-End Latency (ms)": "N/A"
        }
        for key, default_val in defaults.items():
            if key in self.output_vars:
                self.output_vars[key].set(default_val)
        logging.info("UI output fields reset.")

    def on_closing_window(self):
        logging.info("Application window closing...")
        if self._get_is_running_websocket():
            self._set_is_running_websocket(False)
        if self.websocket_thread and self.websocket_thread.is_alive():
            logging.info("Waiting for WebSocket thread to clean up...")
            self.websocket_thread.join(timeout=2.0)
            if self.websocket_thread.is_alive():
                logging.warning("WebSocket thread did not terminate cleanly after 2s.")
        self.root.destroy()
        logging.info("Application shutdown complete.")