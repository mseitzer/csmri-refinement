import logging

from .checkpoint_paths import get_logfile_path


_LOG_FMT = '[{levelname}] {asctime}: {message}'
_TIME_FMT = '%Y-%m-%d %H:%M:%S'


def setup_logging(log_dir, mode, verbose, dry):
  handlers = [logging.StreamHandler()]
  if not dry:
    handlers.append(logging.FileHandler(get_logfile_path(log_dir, mode)))

  level = logging.DEBUG if verbose else logging.INFO
  logging.basicConfig(format=_LOG_FMT,
                      datefmt=_TIME_FMT,
                      style='{',  # Use .format style string formatting
                      handlers=handlers,
                      level=level)
