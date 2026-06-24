"""
Alembic environment configuration for kalygo3-agent-api.

Reads the database URL from the POSTGRES_URL environment variable
and uses the project's SQLAlchemy models for autogenerate support.
"""
import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context

# Load .env so POSTGRES_URL is available
load_dotenv()

# Alembic Config object
config = context.config

# Set the database URL from the environment (same var the app uses)
db_url = os.getenv("POSTGRES_URL", "")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
config.set_main_option("sqlalchemy.url", db_url)

# Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import models so Alembic sees the metadata for autogenerate
from src.db.database import Base  # noqa: E402
from src.db import models  # noqa: E402, F401  – triggers model registration

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a live connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (live database connection)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
