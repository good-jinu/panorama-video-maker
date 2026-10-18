"""Database session and connection management for SQLite."""
import os
from pathlib import Path
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = ROOT_DIR / "data" / "studio.db"

def get_database_url() -> str:
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB_PATH}"

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        url = get_database_url()
        # check_same_thread=False is required for SQLite with multithreaded FastAPI
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, echo=False, connect_args=connect_args)
    return _engine

def init_db() -> None:
    """Create all tables if they do not exist."""
    engine = get_engine()
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database sessions."""
    engine = get_engine()
    with Session(engine) as session:
        yield session
