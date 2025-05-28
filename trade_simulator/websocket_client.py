import asyncio
import websockets
import json
import logging
import time
import config 

async def websocket_handler(order_book_instance, ui_queue_instance, get_is_running_flag):
    """
    Manages WebSocket connection.
    - Receives messages.
    - Updates the shared order_book_instance.
    - Puts a notification on ui_queue_instance for the UI thread to process calculations.
    """
    uri = config.WEBSOCKET_URI
    last_tick_processing_time = None 
    while get_is_running_flag():
        try:
            async with websockets.connect(uri) as websocket:
                logging.info(f"WebSocket connected to {uri}")
                ui_queue_instance.put({"type": "status", "data": "Connected", "stream_symbol": uri.split('/')[-1]})
                
                async for message_str in websocket:
                    if not get_is_running_flag(): break
                    
                    ws_client_tick_start_time = time.perf_counter() 
                    
                    try:
                        data = json.loads(message_str)
                        if isinstance(data, dict) and data.get("event") == "error":
                            err_msg = data.get('msg', 'Unknown WebSocket error event')
                            code = data.get('code', '')
                            logging.error(f"WebSocket Error Event: {err_msg} (Code: {code})")
                            ui_queue_instance.put({"type": "error", "data": f"WS Error: {err_msg} (Code: {code})"})
                            continue 

                        if "asks" not in data or "bids" not in data: 
                            logging.warning(f"Received non-orderbook or malformed message (first 100 chars): {str(data)[:100]}")
                            continue

                        if order_book_instance.update(data): 
                            # Put a message on the queue for the UI thread to trigger calculations
                            ui_queue_instance.put({
                                "type": "book_update_notification",
                                "timestamp_ws_recv": ws_client_tick_start_time 
                            })
                        else:
                            logging.warning("Failed to update order book from tick in WebSocket client.")

                    except json.JSONDecodeError:
                        logging.error(f"Failed to decode JSON from WebSocket: {message_str}")
                    except Exception as e:
                        logging.exception(f"Error processing message in WebSocket client loop")
                    
                    if last_tick_processing_time:
                        time_since_last = time.perf_counter() - last_tick_processing_time
                        if time_since_last < 0.01: 
                             await asyncio.sleep(0.01 - time_since_last) 
                    last_tick_processing_time = time.perf_counter()


        except websockets.exceptions.ConnectionClosed as e:
            logging.warning(f"WebSocket connection closed: {e}. Reconnecting if still running...")
            if get_is_running_flag():
                ui_queue_instance.put({"type": "status", "data": "Reconnecting..."})
                await asyncio.sleep(5) # Wait before retrying
            else: break 
        except ConnectionRefusedError:
            logging.error(f"WebSocket connection refused for {uri}. Check VPN/endpoint accessibility.")
            ui_queue_instance.put({"type": "error", "data": "Connection refused. Check VPN/endpoint."})
            break 
        except Exception as e:
            logging.error(f"Unhandled WebSocket error for {uri}: {e}")
            if get_is_running_flag():
                ui_queue_instance.put({"type": "status", "data": f"WS Error: {str(e)[:50]}... Retrying..."})
                await asyncio.sleep(5)
            else: break
            
    logging.info("WebSocket client task has finished.")
    ui_queue_instance.put({"type": "status", "data": "Disconnected"})