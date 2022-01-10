import logging
import sys


def setup_logger():
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%m/%d %H:%M:%S",
        handlers=[ch],
    )
