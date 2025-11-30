"""Queue with optional periodic logging of remaining tasks for worker pipelines."""
import logging
import threading
from queue import Queue
from typing import TypeVar

T = TypeVar('T')
class TaskQueue(Queue[T]):
    """Queue that periodically logs remaining tasks when logger level allows."""
    def __init__(self, logger: logging.Logger | logging.LoggerAdapter, log_period: int = 60):
        super().__init__()
        # Support LoggerAdapter/BoundLogger that may not expose .level
        level = getattr(logger, "level", None)
        if level is None and hasattr(logger, "logger"):
            level = getattr(getattr(logger, "logger"), "level", None)
        if level is not None and level <= logging.INFO:
            self.logger = logger
            self.log_period = log_period
            self.finish_event = threading.Event()
            self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self.log_thread.start()

    def _log_worker(self):
        while not self.finish_event.is_set():
            self.finish_event.wait(self.log_period)
            if self.finish_event.is_set():
                break
            self.logger.info("Remaining tasks: %d", self.unfinished_tasks)

    def stop_logging(self):
        """Stop periodic logging thread."""
        if hasattr(self, 'logger'):
            self.finish_event.set()
            self.log_thread.join(timeout=5)
