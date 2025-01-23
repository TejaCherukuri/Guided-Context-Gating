import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True)

# Create the full log file path
log_file_path = os.path.join(logs_path, LOG_FILE)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] - %(filename)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # Save logs to a file
        logging.StreamHandler()  # Display logs in the console
    ]
)