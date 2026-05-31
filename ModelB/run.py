import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .src.config import get_default_config, parse_args
    from .src.train import train_model
except ImportError:  # pragma: no cover - script-style fallback
    from src.config import get_default_config, parse_args
    from src.train import train_model

if __name__ == "__main__":
    config = parse_args()
    train_model(config)
