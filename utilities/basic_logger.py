import logging
import os


class BasicLogger(object):

    @staticmethod
    def setup_console_logging(config):
        """Sets up the logger"""
        logging_sv_path = os.path.join(config.out_logger, "training.log")
        logging.basicConfig(level=config.logging_level,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=logging_sv_path,
                            filemode='w')

        #################################################################################################
        handler = logging.StreamHandler()
        handler.setLevel(config.logging_level)
        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        logging.getLogger('').addHandler(handler)
        return logging


def setup_logger(filename="train.log", use_console_log=True, use_file_log=False):
    """Sets up the logger"""
    # remove the old log file
    if os.path.isfile(filename):
        os.remove(filename)

    log_level = logging.INFO
    formatter = logging.Formatter("%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s")

    logger = logging.getLogger()
    logger.setLevel(log_level)

    if use_console_log:
        # add a console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if use_file_log:
        # add a file handler
        handler = logging.FileHandler(filename)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger