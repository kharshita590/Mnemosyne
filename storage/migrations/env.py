import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config

from config.settings import settings
from storage.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Always defer to the app's env-driven settings rather than the static URL
# in alembic.ini, so `alembic upgrade head` targets the right DB in every
# environment (local Docker Postgres vs. the deployed Supabase instance).
# Set the value directly in the section dict rather than via
# config.set_main_option(), which routes through configparser's string
# interpolation and chokes on the "%" in a percent-encoded password.
_db_url = settings.database_url


def run_migrations_offline() -> None:
    context.configure(url=_db_url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _db_url
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
    )

    async def run_async_migrations():
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
        await connectable.dispose()

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
