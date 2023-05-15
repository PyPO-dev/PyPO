import sys
import logging

##
# @file
# This file contains class definitions of the custom logger objects used in PyPO.
# A distinction is made for logging in the terminal and in the GUI.
def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

##
# Class for formatting of the logging from the terminal.
# Has custom colors for different logging levels.
class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;1m"
    blue = "\x1b[34;1m"
    purple = "\x1b[35;1m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s "#(%(filename)s:%(lineno)d)" 
    
    addLoggingLevel('WORK', logging.INFO-1)
    addLoggingLevel('RESULT', logging.INFO-2)
    
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WORK: blue + format + reset,
        logging.RESULT: purple + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: bold_red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)
    
##
# Class for formatting of the logging to the GUI console.
class CustomGUIFormatter(logging.Formatter):
    format = "%(asctime)s - %(levelname)s - %(message)s "#(%(filename)s:%(lineno)d)" 
    
    #addLoggingLevel('WORK', logging.INFO)

    FORMATS = {
        logging.DEBUG: format,
        logging.INFO: format,
        logging.WORK: format,
        logging.RESULT: format,
        logging.WARNING: format,
        logging.ERROR: format,
        logging.CRITICAL: format
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

##
# Class for instantiating a logger object for the terminal.
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
        logger.addHandler(ch)

        return logger

##
# Utility class for creating GUI loggers.
# Has emit method which writes the logging output to the console.
class GUILogger(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)

    def emit(self, record):
        self.edit.append(self.format(record))
        self.edit.ensureCursorVisible()

##
# Class for instantiating a GUI logger.
class CustomGUILogger(object):
    def __init__(self, owner=None):
        self.owner = "Logger" if owner is None else owner

    def __del__(self):
        del self

    def getCustomGUILogger(self, TextEditWidget):
        ch = GUILogger()
        
        ch.edit = TextEditWidget
        ch.setFormatter(CustomGUIFormatter())
        
        logger = logging.getLogger(self.owner)
        
        if logger.hasHandlers():
            logger.handlers = []
        
        logger.setLevel(logging.DEBUG)

        logger.addHandler(ch)
        return logger
