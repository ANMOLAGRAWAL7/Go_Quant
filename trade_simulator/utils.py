import logging
import requests
import config 

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

OKX_API_BASE_URL = "https://www.okx.com" # Base URL

def fetch_okx_ticker_data_sync(instrument_id: str):
    """
    Synchronously fetches ticker data (including 24h volume) for a given instrument ID from OKX.
    Returns the 24h volume in base currency (asset) or None on error.
    """
    if not instrument_id:
        return None
        
    ticker_url = f"{OKX_API_BASE_URL}/api/v5/market/ticker"
    params = {"instId": instrument_id}
    
    try:
        response = requests.get(ticker_url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        if data.get("code") == "0" and data.get("data") and isinstance(data["data"], list) and len(data["data"]) > 0:
            ticker_info = data["data"][0] 
            # vol24h: 24h trading volume, in base currency.
            vol_24h_asset_str = ticker_info.get("vol24h") 
            if vol_24h_asset_str:
                logging.info(f"Fetched 24h volume for {instrument_id}: {vol_24h_asset_str} (base asset)")
                return vol_24h_asset_str 
            else:
                logging.warning(f"vol24h not found in ticker response for {instrument_id}: {ticker_info}")
        else:
            err_msg = data.get('msg', f"Unknown API error or no data for {instrument_id}")
            logging.error(f"Error fetching ticker data from OKX API for {instrument_id}: {err_msg}")

    except requests.exceptions.RequestException as e:
        logging.error(f"RequestException while fetching ticker for {instrument_id}: {e}")
    except Exception as e:
        logging.error(f"General error fetching ticker for {instrument_id}: {e}")
    
    return None
def fetch_okx_spot_instruments_sync():
    """
    Synchronously fetches spot instruments from OKX.
    Returns a list of instrument IDs or a default list on error.
    """
    try:
        response = requests.get(config.OKX_INSTRUMENTS_API, timeout=15)
        response.raise_for_status() 
        data = response.json()
        if data.get("code") == "0" and data.get("data"):
            instruments = sorted([item["instId"] for item in data["data"] if "-" in item["instId"]])
            if instruments:
                logging.info(f"Fetched {len(instruments)} SPOT instruments from OKX.")
                return instruments
            else:
                logging.warning("No SPOT instruments found in OKX API response.")
        else:
            logging.error(f"Error fetching instruments from OKX API: {data.get('msg', 'Unknown API error')}")
    except requests.exceptions.RequestException as e:
        logging.error(f"RequestException while fetching instruments: {e}")
    except Exception as e: 
        logging.error(f"General error fetching instruments: {e}")
    
    logging.warning("Failed to fetch instruments, using default list.")
    return ["BTC-USDT", "ETH-USDT", "OKB-USDT"] # Fallback list