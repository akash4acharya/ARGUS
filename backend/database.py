import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/cctv_speed_db",
)

SQLITE_FALLBACK_URL = os.getenv(
    "SQLITE_FALLBACK_URL",
    "sqlite+aiosqlite:///cctv_speed_local.db",
)

try:
    engine = create_async_engine(DATABASE_URL, pool_size=5, max_overflow=10, echo=False)
except Exception:
    # Use SQLite fallback directly if connection string is invalid
    engine = create_async_engine(SQLITE_FALLBACK_URL, echo=False)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    """
    Initializes database tables. Gracefully falls back to SQLite if PostgreSQL server is offline.
    """
    global engine, AsyncSessionLocal

    try:
        async with engine.begin() as conn:
            # Check if PostgreSQL or SQLite dialect
            is_sqlite = "sqlite" in str(engine.url)
            id_col = "INTEGER PRIMARY KEY AUTOINCREMENT" if is_sqlite else "BIGSERIAL PRIMARY KEY"
            time_default = "CURRENT_TIMESTAMP"

            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS speed_violations (
                        id {id_col},
                        camera_id INT NOT NULL DEFAULT 1,
                        track_id INT NOT NULL,
                        vehicle_class VARCHAR(32) NOT NULL,
                        measured_speed_kmh FLOAT NOT NULL,
                        speed_limit_kmh FLOAT NOT NULL,
                        snapshot_path TEXT NOT NULL,
                        captured_at TIMESTAMP NOT NULL DEFAULT {time_default}
                    );
                    """
                )
            )
        print("[INFO] Database connected and schema verified.")
    except Exception as err:
        print(f"[WARN] PostgreSQL connection failed ({err}). Switching to SQLite local fallback: {SQLITE_FALLBACK_URL}")
        engine = create_async_engine(SQLITE_FALLBACK_URL, echo=False)
        AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS speed_violations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id INT NOT NULL DEFAULT 1,
                        track_id INT NOT NULL,
                        vehicle_class VARCHAR(32) NOT NULL,
                        measured_speed_kmh FLOAT NOT NULL,
                        speed_limit_kmh FLOAT NOT NULL,
                        snapshot_path TEXT NOT NULL,
                        captured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
            )
        print("[INFO] Local SQLite database initialized successfully.")
