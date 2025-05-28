## Trade Simulator System Documentation

**Version:** 1.0
**Date:** May 17, 2025
**Author:** Anmol Agrawal

### Table of Contents

1.  **Introduction and Objectives**
2.  **System Architecture**
    2.1. Core Modules
    2.2. Data Flow
3.  **Core Components**
    3.1. User Interface (UI)
    3.2. WebSocket Client
    3.3. Order Book Management
4.  **Modeling and Calculation Engine**
    4.1. Model Selection Philosophy
    4.2. Expected Slippage (Linear Regression)
    4.3. Expected Fees (Rule-Based Model)
    4.4. Expected Market Impact (Almgren-Chriss Models)
        4.4.1. A-C Simple Model
        4.4.2. A-C Dynamic Programming (DP) Model
        4.4.3. A-C Closed-Form (CF) Model
    4.5. Net Cost Calculation
    4.6. Maker/Taker Proportion (Logistic Regression)
5.  **Performance and Optimization**
    5.1. Latency Measurement and Benchmarking
    5.2. Optimization Techniques Implemented
        5.2.1. Concurrency and Asynchronous Operations
        5.2.2. Data Types and Structures
        5.2.3. Algorithmic Considerations
        5.2.4. UI Update Strategy
        5.2.5. Efficient Logging
6.  **Configuration**
7.  **Setup and Execution**
8.  **Limitations and Future Enhancements**
9.  **Video Demonstration Outline**
    9.1. System Functionality Walkthrough
    9.2. Code Review Highlights
    9.3. Implementation Explanation

---

### 1. Introduction and Objectives

This document details the design, implementation, and functionality of the High-Performance Trade Simulator. The system's primary objective is to provide real-time estimations of transaction costs and market impact for cryptocurrency trades on the OKX SPOT exchange. It achieves this by:

*   Connecting to a WebSocket endpoint (`wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP`) to receive live L2 order book data.
*   Processing this data in real-time.
*   Applying various financial models to estimate key trading metrics.
*   Presenting these estimations and input parameters through a user-friendly Tkinter-based interface.

The system is designed with a focus on performance, modularity, and extensibility, implemented in Python. It adheres to the technical requirements of processing data faster than the stream is received, robust error handling, and clean code architecture.

### 2. System Architecture

The simulator employs a modular architecture to separate concerns, enhancing maintainability and testability.

**2.1. Core Modules:**

*   **`main.py`**: Entry point of the application. Initializes logging, global settings, and launches the UI.
*   **`ui.py` (`TradeSimulatorApp`)**: Manages the Tkinter GUI, user inputs, output display, and coordinates between the WebSocket client and the calculation engine.
*   **`websocket_client.py` (`websocket_handler`)**: Handles asynchronous connection and real-time data reception from the WebSocket endpoint.
*   **`orderbook.py` (`OrderBook`)**: Represents and maintains the L2 order book state (bids and asks).
*   **`models.py` (`TradeCalculators`)**: Contains the business logic and financial models for calculating slippage, fees, market impact, and maker/taker probabilities.
*   **`ac_dynamic_programming.py`**: Implements the Almgren-Chriss model using a dynamic programming approach for optimal execution.
*   **`ac_closed_form.py`**: Implements the Almgren-Chriss model using its analytical closed-form solution.
*   **`config.py`**: Centralized configuration for API endpoints, default UI values, model coefficients, and fee structures.
*   **`utils.py`**: Utility functions, including logging setup and synchronous API calls (e.g., fetching available instruments).
*   **`latency_visualizer.py`**: A standalone script to parse and visualize latency data logged by the application.

**2.2. Data Flow:**

1.  **User Input:** The user configures parameters (quantity, volatility, etc.) via the UI (`ui.py`).
2.  **WebSocket Connection:** Upon user command, `ui.py` initiates a WebSocket connection managed by `websocket_client.py` in a separate thread.
3.  **Data Ingestion:** `websocket_client.py` receives L2 order book snapshots asynchronously.
4.  **Order Book Update:** Each snapshot is used to update the shared `OrderBook` instance (`orderbook.py`).
5.  **Notification:** `websocket_client.py` places a notification (including a high-precision timestamp of message receipt) onto a thread-safe `queue.Queue` monitored by `ui.py`.
6.  **Processing Trigger:** `ui.py` retrieves the notification from the queue and triggers the calculation process.
7.  **Calculations:** `ui.py` fetches current input parameters, and instructs `TradeCalculators` (`models.py`) to perform all estimations using the latest order book data.
8.  **Output Display:** The results from `TradeCalculators` are formatted and displayed on the UI. Latency metrics are calculated and logged.
9.  **Loop:** Steps 3-8 repeat for each incoming WebSocket message while the connection is active.

```
User Inputs --> [ ui.py ] --(start/stop)--> [ websocket_client.py (async thread) ]
                                                        |
                                                        | (L2 Data)
                                                        V
                                                [ orderbook.py ] (shared instance)
                                                        |
                                                        | (update notification via Queue)
                                                        V
                                                [ ui.py ] --(data for calc)--> [ models.py ]
                                                        |                           | (AC DP/CF)
                                                        |                           V
                                                        |                [ac_*.py modules]
                                                        |
                                                        --(results)--> Display on UI
                                                        |
                                                        +--(latency data)--> [ latency_log.csv ]
```

### 3. Core Components

**3.1. User Interface (`ui.py` - `TradeSimulatorApp`)**

*   **Framework:** Tkinter, providing a native cross-platform GUI.
*   **Layout:**
    *   **Left Panel:** Contains input fields for Exchange (fixed to OKX), Spot Asset (dynamically fetched), Order Type (fixed to Market), Side, Quantity (USD Equivalent), Volatility, Assumed Daily Volume, Fee Tier, and parameters for A-C DP and A-C CF models.
    *   **Right Panel:** Displays processed output values including Timestamp, Best Bid/Ask, Mid Price, Spread, Estimated Average Fill Price, Predicted Slippage (%), Slippage Cost (USD), Fee Cost (USD), various Market Impact estimations, Net Cost, Maker/Taker Proportions, and key latency metrics.
*   **Responsiveness:** WebSocket communication and primary data processing are offloaded to separate threads/async tasks to prevent UI freezing. A `queue.Queue` is used for inter-thread communication from the WebSocket client to the UI thread, ensuring UI updates are performed safely.
*   **Dynamic Asset Loading:** On startup, available SPOT instruments are fetched from the OKX API (`utils.fetch_okx_spot_instruments_sync`) and populated in a Combobox.
*   **Input Validation:** Basic validation is performed on numeric inputs before starting the WebSocket connection to prevent errors during calculations.
*   **Latency Logging:** The UI orchestrates the capture and logging of `app_timestamp_ms`, `ws_recv_timestamp_ms`, `data_processing_latency_ms`, `ui_update_latency_ms`, and `end_to_end_latency_ms` into `latency_log.csv`.

**3.2. WebSocket Client (`websocket_client.py` - `websocket_handler`)**

*   **Technology:** `websockets` library with `asyncio` for high-performance, non-blocking I/O.
*   **Functionality:**
    *   Connects to the specified URI: `wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP`.
    *   Receives JSON messages containing L2 order book snapshots.
    *   Parses JSON and validates basic message structure.
    *   Handles OKX-specific error messages within the stream.
    *   Updates the shared `OrderBook` instance directly.
    *   Sends a lightweight notification (including `timestamp_ws_recv` which is `time.perf_counter()` upon message arrival) to the UI thread via the `ui_queue_instance`.
    *   Implements reconnection logic with a delay in case of connection drops.
*   **Error Handling:** Catches connection errors, JSON parsing errors, and other exceptions, logging them and attempting to recover or notify the UI.

**3.3. Order Book Management (`orderbook.py` - `OrderBook`)**

*   **Data Structure:** Stores bids and asks as lists of `[Decimal(price), Decimal(quantity)]` tuples. Bids are sorted descending, asks ascending, as per OKX data format.
*   **Precision:** Uses `Decimal` type for all price and quantity data to maintain high precision, configured globally via `getcontext().prec = config.DECIMAL_PRECISION`.
*   **Core Methods:**
    *   `update(data)`: Parses incoming WebSocket data and replaces current bid/ask lists.
    *   `get_best_bid()`, `get_best_ask()`: Return the top-of-book prices.
    *   `get_best_bid_quantity()`, `get_best_ask_quantity()`: Return quantities at top-of-book.
    *   `get_mid_price()`: Calculates (best_bid + best_ask) / 2.
    *   `get_spread_bps()`: Calculates bid-ask spread in basis points.
    *   `get_depth_proxy(side, mid_price, percentage_from_mid)`: Calculates sum of asset quantity within a specified percentage from the mid-price on one side of the book. Used as a feature in slippage regression.
    *   `get_sum_quantities_n_levels(side, num_levels)`: Sums quantities for the top N levels on a given side. Used for Order Book Imbalance feature in maker/taker model.

### 4. Modeling and Calculation Engine (`models.py` - `TradeCalculators`)

This module centralizes all financial calculations. It receives necessary market data (via the `OrderBook` instance) and user-defined parameters to produce the output estimations.

**4.1. Model Selection Philosophy:**

The models were chosen to balance implementational feasibility, computational performance for real-time updates, and relevance to the core objectives:
*   **Regression Models (Linear & Logistic):** Offer a good trade-off between predictive power and speed. They are data-driven but, in this implementation, use pre-defined coefficients (from `config.py`) simulating a trained model.
*   **Rule-Based Models (Fees):** Simple, accurate, and directly reflect exchange fee structures.
*   **Almgren-Chriss Variants (Market Impact):** Standard academic and industry models for market impact and optimal execution. Different variants (Simple, DP, CF) provide varying levels of complexity and insight.

**4.2. Expected Slippage (Linear Regression)**

*   **Methodology:** A linear regression model is used to predict slippage percentage.
    *   `slippage_pct_predicted = intercept + c1*feat1 + c2*feat2 + ...`
*   **Features Used:**
    1.  **Inverse Depth Proxy (`inverse_depth_proxy`):** Calculated as `1 / depth_proxy_asset`. `depth_proxy_asset` is the sum of asset quantities within `config.DEPTH_PROXY_PERCENTAGE` (e.g., 0.5%) of the mid-price on the side of the book the order would interact with. A smaller depth (larger inverse) suggests higher slippage. Capped to prevent division by zero or extreme values.
    2.  **Daily Volatility (`daily_volatility_decimal`):** Calculated as `(annual_volatility_pct / 100) / sqrt(252)`. Higher volatility generally implies higher slippage risk.
    3.  **Order Size (`quantity_usd`):** The USD equivalent of the order. Larger orders typically incur more slippage.
*   **Coefficients (from `config.py`):**
    *   `config.SLIPPAGE_REG_INTERCEPT`: Base slippage.
    *   `config.SLIPPAGE_REG_COEF_DEPTH_INV`: Weight for inverse depth proxy.
    *   `config.SLIPPAGE_REG_COEF_SIZE_USD`: Weight for order size in USD.
    *   `config.SLIPPAGE_REG_COEF_VOL_DAILY`: Weight for daily volatility.
*   **Output:**
    *   `Slippage Predicted (%) (LinReg)`: The predicted slippage as a percentage, floored at 0.
    *   `Slippage Cost Predicted (USD) (LinReg)`: `slippage_pct_predicted_decimal * quantity_usd`.
    *   `Est. Avg Fill Price (Book Walk)`: For reference, this is also calculated by "walking the book" (simulating a fully aggressive market order consuming available liquidity). This is *actual* slippage against best bid/ask if the order were executed immediately and aggressively.
*   **Rationale:** Linear regression is computationally inexpensive, easy to interpret, and can capture primary drivers of slippage. The chosen features (liquidity, volatility, size) are standard factors influencing slippage.

**4.3. Expected Fees (Rule-Based Model)**

*   **Methodology:** A simple rule-based calculation.
    *   `fee_cost_usd = avg_fill_price_walk * filled_quantity_asset_walk * fee_rate`
*   **Parameters:**
    *   `avg_fill_price_walk`: The average fill price obtained from walking the book.
    *   `filled_quantity_asset_walk`: The total asset quantity filled from walking the book.
    *   `fee_rate`: Selected by the user from `config.FEE_TIERS_OKX_SPOT`. This model assumes the entire order is executed as a "taker" order.
*   **Output:** `Fee Cost (USD)`.
*   **Rationale:** Directly reflects exchange fee schedules for taker orders.

**4.4. Expected Market Impact (Almgren-Chriss Models)**

Market impact refers to the adverse price movement caused by the act of trading. Three variations of the Almgren-Chriss model are implemented.

**4.4.1. A-C Simple Model (`calculate_market_impact_ac_simple`)**

*   **Methodology:** A simplified, instantaneous market impact model inspired by Almgren-Chriss concepts.
    *   `relative_price_change = ETA_SIMPLE * daily_vol_fraction * (quantity_asset / assumed_daily_volume_asset)^DELTA_SIMPLE`
    *   `market_impact_cost_usd = relative_price_change * mid_price * quantity_asset`
*   **Parameters (from `config.py` and UI):**
    *   `config.AC_ETA_SIMPLE`: Sensitivity coefficient.
    *   `config.AC_DELTA_SIMPLE`: Exponent for the participation rate.
    *   `daily_vol_fraction`: Daily volatility.
    *   `quantity_asset`: Order quantity in base asset.
    *   `assumed_daily_volume_asset`: User input for typical daily volume of the asset.
    *   `mid_price`: Current mid-price from the order book.
*   **Output:** `Market Impact (USD) (A-C Simple)`.
*   **Rationale:** Provides a quick, first-order approximation of market impact based on volatility and participation rate. Computationally very cheap.

**4.4.2. A-C Dynamic Programming (DP) Model (`calculate_market_impact_ac_dp` using `ac_dynamic_programming.py`)**

*   **Methodology:** Numerically solves the Almgren-Chriss optimal execution problem using dynamic programming. It discretizes time and inventory to find a trading trajectory that minimizes a cost function incorporating both execution costs (temporary and permanent impact) and risk (variance of execution price). The "cost" returned by `get_optimal_execution_trajectory_and_cost` is the minimized objective function value, interpreted here as the total market impact cost.
*   **Hamiltonian:** The core of the DP relies on a Hamiltonian function (see `ac_dynamic_programming.hamiltonian_dp`) that balances immediate execution costs against the risk and future costs of holding remaining inventory. The structure of the Hamiltonian is:
    `H = risk_aversion * q_k * PermanentImpact(q_k/tau) + risk_aversion * (X_k - q_k) * tau * TemporaryImpact(q_k/tau) + 0.5 * risk_aversion^2 * sigma^2 * tau * (X_k - q_k)^2`
    where `q_k` is shares sold in step `k`, `X_k` is inventory at start of step `k`, `tau` is step duration.
*   **Impact Functions:**
    *   Temporary Impact: `eta * (volume_rate^alpha)`
    *   Permanent Impact: `gamma * (volume_rate^beta)`
*   **Parameters (from `config.py` and UI):**
    *   `quantity_usd`, `mid_price`: To get `total_shares_asset`.
    *   `annual_volatility_pct`: To derive `volatility_per_step`.
    *   `ac_dp_risk_aversion_var` (Ψ): User's risk aversion.
    *   `ac_dp_time_steps_var` (N): Number of discrete time steps for execution.
    *   `config.DEFAULT_AC_DP_TOTAL_EXECUTION_HORIZON_DAYS` (T): Total time horizon for liquidation.
    *   `config.AC_DP_ALPHA`, `config.AC_DP_BETA`: Exponents for impact functions.
    *   `config.AC_DP_ETA`, `config.AC_DP_GAMMA`: Coefficients for impact functions.
*   **Output:** `Market Impact (USD) (A-C DP)`.
*   **Rationale:** Provides a more sophisticated, path-dependent market impact estimate by optimizing the execution schedule. More computationally intensive than the simple or closed-form models, especially with many time steps or large inventory.

**4.4.3. A-C Closed-Form (CF) Model (`calculate_market_impact_ac_closed_form` using `ac_closed_form.py`)**

*   **Methodology:** Utilizes the analytical (closed-form) solution to the Almgren-Chriss model for expected execution shortfall. This solution assumes continuous trading and specific forms for impact functions.
    *   Calculates `epsilon` (half-spread), `eta_param` (temporary impact coefficient from daily volume), `gamma_param` (permanent impact coefficient from daily volume), `single_step_price_variance`, `eta_hat`, and `kappa` (decay rate of trading).
    *   The expected shortfall formula (Eq. 20 in Almgren & Chriss 2000) combines terms for permanent impact, spread cost, and temporary impact over the optimal trajectory.
*   **Parameters (from `config.py` and UI):**
    *   `quantity_usd`, `mid_price`: To get `total_shares_to_sell`.
    *   `annual_volatility_pct`.
    *   `ac_cf_bid_ask_spread_var`: User-defined bid-ask spread in USD.
    *   `ac_cf_daily_vol_asset_var`: User-defined daily trade volume in asset units.
    *   `ac_cf_risk_aversion_var` (λ): User's risk aversion parameter (lambda).
    *   `config.DEFAULT_AC_DP_TOTAL_EXECUTION_HORIZON_DAYS` (T): Total liquidation time (shared with DP for UI consistency).
    *   `ac_dp_time_steps_var` (N): Number of trading intervals (shared with DP).
    *   `config.DEFAULT_AC_CF_TRADING_DAYS_PER_YEAR`.
*   **Output:** `Exp. Shortfall (USD) (A-C CF)`.
*   **Rationale:** Offers an analytical solution for optimal execution cost under specific A-C assumptions. Computationally cheaper than DP but relies on more restrictive assumptions.

**4.5. Net Cost Calculation**

*   **Methodology:** Summation of key cost components.
    *   `Net Cost = Slippage Cost Predicted (USD) (LinReg) + Fee Cost (USD) + Market Impact (USD) (A-C Simple)`
*   **Rationale:** Provides an aggregate estimated transaction cost. The A-C Simple model is used here for a quick, combined view, as DP and CF represent costs under specific optimal execution strategies that might differ from a single market order. The UI displays DP and CF costs separately.

**4.6. Maker/Taker Proportion (Logistic Regression)**

*   **Methodology:** A logistic regression model predicts the probability of an order (if placed passively) acting as a "maker" order.
    *   `logit(P_maker) = intercept + c1*feat1 + c2*feat2 + ...`
    *   `P_maker = 1 / (1 + exp(-logit(P_maker)))`
    *   `P_taker = 1 - P_maker`
*   **Features Used:**
    1.  **Spread (`spread_bps`):** Current bid-ask spread in basis points. Wider spreads might offer more opportunity to be a maker.
    2.  **Order Size (`quantity_usd`):** USD equivalent of the order. Larger orders might be harder to fill passively.
    3.  **Order Book Imbalance (`obi_ratio_clamped`):**
        *   Defined as: `sum_qty_N_levels_on_OPPOSITE_side / sum_qty_N_levels_on_MY_PASSIVE_side`.
        *   If BUYING, "my passive side" is BID, "opposite side" is ASK. OBI = `sum_ask_N / sum_bid_N`. A higher ratio (more liquidity on ask side relative to bid) might make it more favorable to place a passive buy (maker).
        *   If SELLING, "my passive side" is ASK, "opposite side" is BID. OBI = `sum_bid_N / sum_ask_N`.
        *   `config.LR_OBI_NUM_LEVELS` determines N. Clamped to avoid extreme values.
    4.  **Size Over Best (`sob_ratio_clamped`):**
        *   Defined as: `(OrderQuantityInAsset / QuantityAtBestOppositeLevel)`.
        *   If BUYING, this is `OrderQty / BestAskQty`. A high ratio means the order is large compared to immediately available liquidity it would *take*, suggesting it's more likely to be a taker or only partially a maker.
        *   If SELLING, this is `OrderQty / BestBidQty`.
        *   Clamped to avoid extreme values.
*   **Coefficients (from `config.py`):**
    *   `config.LR_INTERCEPT`.
    *   `config.LR_COEF_SPREAD_BPS`.
    *   `config.LR_COEF_SIZE_USD`.
    *   `config.LR_COEF_OBI_OPPOSITE_VS_PASSIVE` (applies to the OBI ratio as defined above).
    *   `config.LR_COEF_SOB_SIZE_OVER_BEST_QTY`.
*   **Output:**
    *   `Maker Proportion (%) (LogReg)`
    *   `Taker Proportion (%) (LogReg)`
*   **Rationale:** Logistic regression is suitable for binary classification (or probability estimation for a binary outcome). The features capture aspects of market microstructure that influence the likelihood of a passive order getting filled. The sign of `LR_COEF_OBI_OPPOSITE_VS_PASSIVE` is negative, suggesting that if the opposite side has much more liquidity than your passive placement side, you are less likely to be a maker (more likely to be a taker if you want a fill). This interpretation depends on how "favorable" is defined; perhaps the coefficient implies that high imbalance *against* your passive placement makes it harder to be a maker. The SOB coefficient being negative implies larger orders relative to best available are less likely to be makers.

### 5. Performance and Optimization

Meeting the requirement to process data faster than the stream is received, and providing detailed latency insights, were key design goals.

**5.1. Latency Measurement and Benchmarking (`ui.py`, `latency_visualizer.py`)**

The system measures and logs the following latency components for each processed tick into `latency_log.csv`:

1.  **`app_timestamp_ms`**: Timestamp (Unix epoch ms) taken in `ui.py` just before initiating processing for a new order book update.
2.  **`ws_recv_timestamp_ms`**: Timestamp (from `time.perf_counter()` converted to ms relative to an arbitrary start) recorded by `websocket_client.py` immediately upon receiving the raw message from the WebSocket. This is the earliest point the application sees the data.
3.  **`data_processing_latency_ms`**: Time taken from `ws_recv_timestamp_ms` until all financial calculations in `models.py` are complete.
    *   `calculations_done_time - timestamp_ws_recv`
4.  **`ui_update_latency_ms`**: Time taken purely to update the Tkinter UI elements with new data.
    *   `ui_update_done_time - ui_update_start_time`
5.  **`end_to_end_latency_ms`**: Total time from WebSocket message receipt until UI elements are updated.
    *   `ui_update_done_time - timestamp_ws_recv`
    *   This is effectively `data_processing_latency_ms + ui_update_latency_ms` plus any minor overhead in `_process_book_update_notification`.
6.  **`order_book_timestamp`**: The timestamp provided within the WebSocket message itself (e.g., "2025-05-04T10:39:13Z").

The `latency_visualizer.py` script can be run independently to parse `latency_log.csv`, generate various plots (histograms, box plots, time series) of these latencies, and print summary statistics (mean, median, percentiles). This allows for detailed performance analysis and identification of bottlenecks.

**5.2. Optimization Techniques Implemented**

**5.2.1. Concurrency and Asynchronous Operations:**

*   **Threading (`threading` module):**
    *   The WebSocket client (`websocket_handler`) runs in a separate daemon thread. This prevents network I/O from blocking the UI, ensuring GUI responsiveness.
    *   Fetching available SPOT instruments on startup is also done in a separate thread.
*   **Asynchronous WebSocket (`asyncio`, `websockets`):**
    *   The WebSocket client uses `asyncio` for non-blocking network operations, efficiently handling message streams without tying up the thread.
*   **Queue for Inter-Thread Communication (`queue.Queue`):**
    *   A thread-safe queue (`ui_queue`) is used to pass notifications from the WebSocket thread to the main UI thread. This is crucial for safe UI updates, as Tkinter operations should only occur in the main thread. The UI thread polls this queue periodically.

**5.2.2. Data Types and Structures:**

*   **`Decimal` Type:** Used for all financial calculations (prices, quantities, costs) to ensure high precision and avoid floating-point inaccuracies common with `float`. Precision is globally set in `config.py` and `main.py`.
*   **`OrderBook` Data Structures:** Bids and asks are stored as lists of two-element `Decimal` tuples `(price, quantity)`. While Python lists are dynamic, for the expected L2 depth (e.g., 50-100 levels), list operations are generally efficient. Data is received pre-sorted.

**5.2.3. Algorithmic Considerations:**

*   **Regression Models:** Coefficients are pre-defined in `config.py`. This means no model training occurs at runtime, making predictions extremely fast (simple arithmetic operations).
*   **A-C Dynamic Programming:**
    *   The DP solution has a time complexity related to `num_time_steps * total_shares_to_liquidate * max_sell_per_step`. For large `total_shares_to_liquidate` (which is discretized), this can be slow. The UI warns if `num_time_steps` is large.
    *   `numpy` is used for array operations within `ac_dynamic_programming.py`, which is significantly faster than pure Python loops for numerical tasks.
*   **A-C Closed-Form:** Involves complex mathematical functions (`sinh`, `cosh`, `tanh`) but is an analytical solution, generally fast. `numpy` functions are used for these.
*   **Order Book Operations:** Methods like `get_mid_price`, `get_best_bid/ask` are O(1) as data is sorted. `get_depth_proxy` and `get_sum_quantities_n_levels` iterate a portion of the book, efficient for typical depths.

**5.2.4. UI Update Strategy:**

*   **Selective Updates:** Only relevant output labels are updated.
*   **Queued Processing:** UI updates and calculations are triggered by messages from the `ui_queue`, processed in batches if multiple messages arrive between polling intervals (currently polls every 10ms via `root.after`). This prevents the UI from being overwhelmed by extremely high-frequency data.
*   **Minimal Processing in UI Thread:** Heavy computations (`TradeCalculators`) are done within the UI thread's processing loop but are optimized to be as fast as possible. The primary time-consuming part, WebSocket I/O, is in a separate thread.

**5.2.5. Efficient Logging:**

*   **Standard `logging` module:** Configured for appropriate levels and formats.
*   **Latency CSV Logging:** Raw data is written to `latency_log.csv` without expensive formatting per line, optimized for write speed. Analysis is deferred to `latency_visualizer.py`.

### 6. Configuration (`config.py`)

The `config.py` file serves as a centralized location for all critical parameters and settings:

*   **Endpoints:** `WEBSOCKET_URI`, `OKX_INSTRUMENTS_API`.
*   **Default UI Values:** Initial values for quantity, volatility, etc.
*   **Fee Model:** `FEE_TIERS_OKX_SPOT` dictionary defining various fee tiers and their rates.
*   **Model Coefficients:** All coefficients for Linear Regression (slippage), Logistic Regression (maker/taker), A-C Simple, A-C DP, and A-C CF models. This allows easy tuning and experimentation without code changes.
*   **Global Settings:** `DECIMAL_PRECISION`.

This centralization simplifies maintenance, tuning, and adaptation of the system.

### 7. Setup and Execution

1.  **Prerequisites:**
    *   Python 3.7+
    *   Required libraries (install via `pip install -r requirements.txt`):
        *   `requests`
        *   `websockets`
        *   `numpy`
        *   `pandas` (for `latency_visualizer.py`)
        *   `matplotlib` (for `latency_visualizer.py`)
        *   `seaborn` (for `latency_visualizer.py`)
    *   **VPN:** A VPN connection might be required to access the OKX API and WebSocket endpoint depending on your geographical location.

2.  **Running the Application:**
    *   cd over to the main.py directory in your terminal
    *   Execute the main script: `python main.py`

3.  **Using the Latency Visualizer:**
    *   After running the main application and generating some data in `latency_log.csv`:
        `python latency_visualizer.py`

### 8. Limitations and Future Enhancements

*   **Fixed WebSocket Symbol:** The WebSocket URI in `config.py` is hardcoded for `BTC-USDT-SWAP`. While the UI allows selecting other SPOT assets conceptually, the actual stream processing is fixed. Future work could make the WebSocket stream symbol configurable from the UI.
*   **Model Coefficients:** Regression coefficients are currently placeholders in `config.py`. For production use, these would need to be derived from historical data analysis and model training.
*   **A-C DP Granularity:** The DP model's performance and accuracy are sensitive to the discretization of inventory (`total_shares_int`). For very large orders, `total_shares_int` can become prohibitively large for the `numpy` arrays if each unit is 1 share. A scaling factor or adaptive granularity could be implemented.
*   **Order Type:** Only "Market" order type is simulated for cost estimation. Limit order strategies are not explicitly modeled beyond the Maker/Taker probability.
*   **Error Handling in Models:** While basic checks exist (e.g., for zero mid-price), models could be made more robust to unusual market conditions or parameter inputs (e.g., by returning specific error codes or more descriptive "N/A" states).
*   **No Live Trading:** This is purely a simulator and does not place actual trades.
*   **Single Exchange:** Currently hardcoded for OKX structure and endpoints.

**Potential Future Enhancements:**

*   UI-configurable WebSocket stream symbol.
*   Integration of a historical data pipeline for training regression models.
*   More sophisticated order book imbalance features.
*   Simulation of limit order placement strategies and fill probabilities.
*   Support for multiple exchanges.
*   Backtesting capabilities against historical L2 data.
