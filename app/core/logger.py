"""Central logging. Uses loguru if available, else stdlib logging."""
from __future__ import annotations

import sys

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    try:
        from loguru import logger
        logger.remove()
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | "
                   "<cyan>{extra[name]}</cyan> - {message}",
        )
        try:
            from app.core.config import DATA_DIR
            logger.add(
                DATA_DIR / "app.log",
                level="DEBUG", rotation="1 MB", retention=3, enqueue=True,
            )
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001 - loguru not installed
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    _configured = True


def get_logger(name: str = "app"):
    _configure()
    try:
        from loguru import logger
        return logger.bind(name=name)
    except Exception:  # noqa: BLE001
        import logging
        return logging.getLogger(name)