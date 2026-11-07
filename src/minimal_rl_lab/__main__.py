"""Support python -m minimal_rl_lab without running on import."""
from .cli import main
if __name__ == '__main__':
    raise SystemExit(main())
