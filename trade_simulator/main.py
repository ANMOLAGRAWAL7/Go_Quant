"""
 █████╗ ███╗   ██╗███╗   ███╗ ██████╗ ██╗             ██╗ █████╗ 
██╔══██╗████╗  ██║████╗ ████║██╔═══██╗██║            ███║██╔══██╗
███████║██╔██╗ ██║██╔████╔██║██║   ██║██║            ╚██║╚█████╔╝
██╔══██║██║╚██╗██║██║╚██╔╝██║██║   ██║██║             ██║██╔══██╗
██║  ██║██║ ╚████║██║ ╚═╝ ██║╚██████╔╝███████╗███████╗██║╚█████╔╝
╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝ ╚════╝ 
-----------------------------------------------------------------------------------------------------------------
AUTHOR: Anmol_Agrawal(CSE undergrad at NIT Rourkela)
DURATION OF PROJECT : 14-05-2025 to 17-05-2025
Version: 1.0
Date: 17-05-2025
Description:-->
This script serves as the entry point for a high performance trading simulator application built using Tkinter(GUI). 
-----------------------------------------------------------------------------------------------------------------                                                                
"""
import tkinter as tk
from decimal import getcontext
import config 
from utils import setup_logging
from ui import TradeSimulatorApp

if __name__ == "__main__":
    setup_logging() 
    getcontext().prec = config.DECIMAL_PRECISION # Set global decimal precision

    root = tk.Tk()
    app = TradeSimulatorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing_window) # Handle window close button
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, shutting down.")
        app.on_closing_window() 
    except Exception as e:
        logging.critical(f"Unhandled exception in main Tkinter loop: {e}", exc_info=True)
        try:
            app.on_closing_window()
        except Exception as cleanup_e:
            logging.error(f"Error during critical shutdown: {cleanup_e}")