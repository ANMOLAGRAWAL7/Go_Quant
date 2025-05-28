import logging
from decimal import Decimal, getcontext
import config # For DECIMAL_PRECISION
getcontext().prec = config.DECIMAL_PRECISION
class OrderBook:
    def __init__(self):
        self.bids = []  # List of [Decimal(price), Decimal(quantity)]
        self.asks = []  # List of [Decimal(price), Decimal(quantity)]
        self.timestamp = None
        self.symbol = None

    def update(self, data):
        try:
            self.timestamp = data.get("timestamp")
            self.symbol = data.get("symbol", "UNKNOWN")
            # OKX L2 data comes sorted: bids descending, asks ascending
            self.bids = [(Decimal(price_str), Decimal(qty_str)) for price_str, qty_str in data.get("bids", [])]
            self.asks = [(Decimal(price_str), Decimal(qty_str)) for price_str, qty_str in data.get("asks", [])]
            return True
        except Exception as e:
            logging.error(f"Error updating order book: {e} with data (first 100 chars): {str(data)[:100]}")
            return False

    def get_best_bid(self):
        return self.bids[0][0] if self.bids else None

    def get_best_ask(self):
        return self.asks[0][0] if self.asks else None
    
    def get_best_bid_quantity(self): # New method
        return self.bids[0][1] if self.bids else Decimal("0")

    def get_best_ask_quantity(self): # New method
        return self.asks[0][1] if self.asks else Decimal("0")
    
    def get_mid_price(self):
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / Decimal("2")
        return None

    def get_spread_bps(self):
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        mid = self.get_mid_price()
        if best_bid and best_ask and mid and mid > Decimal("0"):
            return ((best_ask - best_bid) / mid) * Decimal("10000") 
        return Decimal("0") 

    def get_depth_proxy(self, side, mid_price, percentage_from_mid):
        """Calculates sum of asset quantity within a percentage of mid_price."""
        if not mid_price or mid_price <= Decimal("0"): return Decimal("1") # Avoid div by zero

        depth_sum_asset = Decimal("0")
        book_side_levels = self.asks if side == "BUY" else self.bids
        
        for price, qty in book_side_levels:
            if side == "BUY": #asks
                if price <= mid_price * (Decimal("1") + percentage_from_mid):
                    depth_sum_asset += qty
                else: 
                    break
            else: #bids
                if price >= mid_price * (Decimal("1") - percentage_from_mid):
                    depth_sum_asset += qty
                else: 
                    break
        return depth_sum_asset if depth_sum_asset > Decimal("0") else Decimal("0.000001") # Avoid zero depth
    
    def get_sum_quantities_n_levels(self, side: str, num_levels: int) -> Decimal:
        """Calculates sum of asset quantity for the top N levels."""
        book_side = self.bids if side == "SELL" else self.asks # If selling, look at bids; if buying, look at asks
        
        sum_qty = Decimal("0")
        levels_to_consider = min(num_levels, len(book_side))
        
        for i in range(levels_to_consider):
            sum_qty += book_side[i][1] 
        return sum_qty if sum_qty > Decimal("0") else Decimal("0.000001") # Avoid zero for ratios