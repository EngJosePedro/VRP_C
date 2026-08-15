import logging
import os
from datetime import datetime


def build_logger(log_dir="results/logs", name="train"):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # terminal
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # arquivo
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger