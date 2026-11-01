import sqlite3

import sqlite_vec


def open_db(path: str, dim: int) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row  # to read as row["text"] instead of row[2]
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    # run all create at once
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS message (
            id INTEGER PRIMARY KEY,
            channel_id INTEGER,
            author TEXT,
            timestamp TEXT,
            url TEXT,
            content TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_message USING vec0(
            embedding float[{dim}]
        );
        CREATE TABLE IF NOT EXISTS posting (
            url TEXT PRIMARY KEY,
            thread_id INTEGER,
            hash TEXT,
            deadline TEXT,
            archive INTEGER DEFAULT 0
        );
    """)
    return conn


def add_messages(
    db: sqlite3.Connection, messages: list[dict], embeddings: list[list[float]]
):
    with db:
        for m, e in zip(messages, embeddings, strict=True):
            db.execute(
                """
                INSERT OR REPLACE INTO message
                VALUES (:id, :channel_id, :author, :timestamp, :url, :content)
                """,
                m,
            )
            # vec0 dont support insert or replace, so delete first
            db.execute(
                """
                DELETE FROM vec_message
                WHERE rowid = ?
                """,
                (m["id"],),
            )
            db.execute(
                """
                INSERT INTO vec_message(rowid, embedding)
                VALUES (?, ?)
                """,
                (m["id"], sqlite_vec.serialize_float32(e)),
            )


def search(db: sqlite3.Connection, vector: list[float], k: int) -> list[dict]:
    cursor = db.execute(
        """
        SELECT id, content, url
        FROM vec_message
        JOIN message
        ON message.id = vec_message.rowid
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
        LIMIT 20
        """,
        (sqlite_vec.serialize_float32(vector), k),
    )
    return cursor.fetchall()


def delete_message(db, id):
    with db:
        db.execute(
            """
            DELETE FROM message
            WHERE id = ?
            """,
            (id,),
        )
        db.execute(
            """
            DELETE FROM vec_message
            WHERE rowid = ?
            """,
            (id,),
        )


def last_id(db, channel_id):
    row = db.execute(
        """
        SELECT MAX(id)
        FROM message
        WHERE channel_id = ?
        """,
        (channel_id,),
    ).fetchone()
    return row[0]


def save_posting(db, url, thread_id, hash, deadline):
    with db:
        db.execute(
            # use ON CONFLICT ... to avoid resetting archive to 0 on an
            # already-archived posting
            """
            INSERT INTO posting(url, thread_id, hash, deadline)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                thread_id = excluded.thread_id,
                hash = excluded.hash,
                deadline = excluded.deadline
            """,
            (url, thread_id, hash, deadline.isoformat() if deadline else None),
        )


def get_posting(db, url) -> list[dict]:
    row = db.execute(
        """
        SELECT *
        FROM posting
        WHERE url = ?
        """,
        (url,),
    ).fetchone()
    return row
