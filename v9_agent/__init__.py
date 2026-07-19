from .config import V8Config, config_from_mapping, default_config_dict
from .session import GameSession, LevelRunLimitReached

V9Config = V8Config

__version__ = "9.0.0"
__all__ = ["V8Config", "V9Config", "config_from_mapping", "default_config_dict", "GameSession", "LevelRunLimitReached", "__version__"]
