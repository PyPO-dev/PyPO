import sys
import logging

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    #format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)" 
    #format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s "#(%(filename)s:%(lineno)d)" 
    format = "%(asctime)s - %(levelname)s - %(message)s "#(%(filename)s:%(lineno)d)" 
    

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

class CustomLogger(object):
    def __init__(self, owner=None):
        self.owner = "Logger" if owner is None else owner

    def __del__(self):
        del self

    def getCustomLogger(self, stdout=None):
        stdout = sys.stdout if stdout is None else stdout

        logger = logging.getLogger(self.owner)
        
        if logger.hasHandlers():
            logger.handlers = []
        
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(stdout)
        ch.setLevel(logging.DEBUG)

        ch.setFormatter(CustomFormatter())
        #logger.basicConfig(datefmt='%Y-%m-%d %H:%M:%S')

        logger.addHandler(ch)

        return logger

    def getNewStream(self):
        pass
