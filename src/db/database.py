from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
import os

load_dotenv()

SQL_ALCHEMY_DATABASE_URL = os.getenv("POSTGRES_URL")

if not SQL_ALCHEMY_DATABASE_URL:
    raise ValueError("POSTGRES_URL environment variable is required")

if SQL_ALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQL_ALCHEMY_DATABASE_URL = SQL_ALCHEMY_DATABASE_URL.replace(
        "postgres://", "postgresql://", 1
    )

# QueuePool (default) keeps a small set of persistent connections and recycles
# them with pre-ping, avoiding SSL-handshake storms under concurrent load.
# Set DB_USE_NULL_POOL=true only when an external pooler (e.g. PgBouncer in
# transaction mode) explicitly requires single-use connections.
DB_USE_NULL_POOL = os.getenv("DB_USE_NULL_POOL", "false").lower() in {"1", "true", "yes"}
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "120"))

engine_kwargs = {
    "use_native_hstore": False,
    "connect_args": {
        "sslmode": "require",
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 20,
        "keepalives_interval": 5,
        "keepalives_count": 3,
        "application_name": "kalygo3-completion",
    },
}

if DB_USE_NULL_POOL:
    engine_kwargs["poolclass"] = NullPool
else:
    engine_kwargs.update(
        {
            "pool_size": DB_POOL_SIZE,
            "max_overflow": DB_MAX_OVERFLOW,
            "pool_timeout": DB_POOL_TIMEOUT,
            "pool_recycle": DB_POOL_RECYCLE,
            "pool_pre_ping": True,
            "pool_use_lifo": True,
            "pool_reset_on_return": "rollback",
        }
    )

engine = create_engine(SQL_ALCHEMY_DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
