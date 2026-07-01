from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncConnection


async def set_user_context(conn: AsyncConnection, user_id: str) -> None:
    """
    Sets the app.user_id session variable so PostgreSQL RLS policies
    can enforce row-level isolation. Call before any query in a user-scoped session.

    Example RLS policy (add to migration):
        ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
        CREATE POLICY memories_user_isolation ON memories
            USING (user_id = current_setting('app.user_id')::uuid);
    """
    await conn.execute(
        f"SET LOCAL app.user_id = '{user_id}'"  # noqa: S608
    )
