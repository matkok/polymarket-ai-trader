from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DEFAULT_DATABASE_URL = "postgresql+asyncpg://agent_trader_structured:devpassword@localhost:5433/agent_trader_structured"


def get_engine(url: str | None = None) -> AsyncEngine:
    """Create an async SQLAlchemy engine.

    Args:
        url: Database connection URL. Falls back to the DATABASE_URL
            environment variable if not provided.

    Returns:
        Configured ``AsyncEngine`` instance.
    """
    database_url = url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    return create_async_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
    )


def get_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to *engine*.

    Args:
        engine: The ``AsyncEngine`` to bind sessions to.

    Returns:
        An ``async_sessionmaker`` producing ``AsyncSession`` instances.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
