"""Entry point — run with:  uvicorn main:app --reload  or  python main.py"""

import uvicorn
from src.config import get_settings
from src.api.app import app  # noqa: F401  (imported for `uvicorn main:app`)

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=(settings.app_env == "development"),
        log_level=settings.log_level.lower(),
    )
