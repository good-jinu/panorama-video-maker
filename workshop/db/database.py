"""Database session and connection management for SQLite."""
from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine

from workshop.config import settings

DEFAULT_DB_PATH = settings.data_dir / "studio.db"

def get_database_url() -> str:
    return settings.resolved_database_url


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

def get_session() -> Generator[Session]:
    """FastAPI dependency for yielding database sessions."""
    engine = get_engine()
    with Session(engine) as session:
        yield session
