import configparser
from datetime import datetime
from enum import Enum

config = configparser.ConfigParser()
config.read('resources/config/config.ini')


class LogLevel(Enum):
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    MSG = 99

    def __str__(self):
        if self.value == 1:
            return 'DEBUG'
        elif self.value == 2:
            return 'INFO'
        elif self.value == 3:
            return 'WARNING'
        elif self.value == 4:
            return 'ERROR'
        elif self.value == 99:
            return 'MSG'
        else:
            return ''

    @staticmethod
    def parse_str(s):
        if s == 'DEBUG':
            return LogLevel.DEBUG
        elif s == 'INFO':
            return LogLevel.INFO
        elif s == 'WARNING':
            return LogLevel.WARNING
        elif s == 'ERROR':
            return LogLevel.ERROR
        elif s == 'MSG':
            return LogLevel.MSG
        else:
            return None


# INIT LOGGING LEVEL BASED ON CONFIG FILE
if 'log.level' in config['GENERAL SETTINGS']:
    MIN_LOG_LEVEL: LogLevel = LogLevel.parse_str(config['GENERAL SETTINGS']['log.level'])
else:
    MIN_LOG_LEVEL: LogLevel = LogLevel.WARNING


#

class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Logger:
    MSG_STR = "[{}] {} [{}]: {}"

    def __init__(self, speaker: str):
        self.speaker = speaker

    def log(self, msg: str):
        ts = datetime.now()
        print(self.MSG_STR.format(LogLevel.__str__(MIN_LOG_LEVEL), ts, self.speaker, msg))

    def debug(self, msg: str):
        if LogLevel.DEBUG.value >= MIN_LOG_LEVEL.value:
            ts = datetime.now()
            print(self.MSG_STR.format(LogLevel.__str__(LogLevel.DEBUG), ts, self.speaker, msg))

    def info(self, msg: str):
        if LogLevel.INFO.value >= MIN_LOG_LEVEL.value:
            ts = datetime.now()
            print(bcolor.OKBLUE + self.MSG_STR.format(LogLevel.__str__(LogLevel.INFO), ts, self.speaker,
                                                      msg) + bcolor.ENDC)

    def warn(self, msg: str):
        if LogLevel.WARNING.value >= MIN_LOG_LEVEL.value:
            ts = datetime.now()
            print(bcolor.WARNING + self.MSG_STR.format(LogLevel.__str__(LogLevel.WARNING), ts, self.speaker,
                                                       msg) + bcolor.ENDC)

    def error(self, msg: str):
        if LogLevel.ERROR.value >= MIN_LOG_LEVEL.value:
            ts = datetime.now()
            print(bcolor.FAIL + self.MSG_STR.format(LogLevel.__str__(LogLevel.ERROR), ts, self.speaker,
                                                    msg) + bcolor.ENDC)
