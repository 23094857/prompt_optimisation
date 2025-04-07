import logging
import datetime
import os

log_directory = None

def get_or_create_log_directory():
    global log_directory
    if log_directory is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_directory = f"./logs/{timestamp}"
        os.makedirs(log_directory, exist_ok=True)
    return log_directory

def setup_logger(name, directory, log_file, level=logging.INFO):
    """Set up as many loggers as you want with unique file handlers."""
    full_log_path = os.path.join(directory, log_file)
    formatter = logging.Formatter('%(asctime)s %(message[tid8)s', datefmt='%m/%d/%Y %H:%M:%S')
    handler = logging.FileHandler(full_log_path)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger