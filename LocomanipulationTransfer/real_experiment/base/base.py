import logging
from colorlog import ColoredFormatter
from pynput import keyboard

class Base(object):
    def __init__(self) -> None:
        self.logger = self.setup_logger()
        self.key_pressed = False
        self.pressed_key = None
        self.keyboard_listener = None
        
    def setup_logger(self):
        """Return a logger with a default ColoredFormatter."""
        formatter = ColoredFormatter(
            "%(log_color)s[%(levelname)s] %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )
        
        logger = logging.getLogger(str(self.__class__))
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        return logger

    def logInfo(self, message):
        self.logger.info(message)

    def logWarning(self, message):
        self.logger.warning(message)

    def logError(self, message):
        self.logger.error(message)

    def logCritical(self, message):
        self.logger.critical(message)

    def keyboardCallback(self, key):
        try:
            self.pressed_key = key.char  # single-char keys
        except:
            self.pressed_key = key.name # Do not change the default pressed key
        self.key_pressed = True

    def startKeyboardListener(self):
        self.keyboard_listener = keyboard.Listener(on_press=self.keyboardCallback)
        self.keyboard_listener.start()
