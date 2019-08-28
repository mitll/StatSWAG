import logging
import warnings

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.ERROR)

__version__ = '1.0.0'

__all__ = ['estimators','datasets']
