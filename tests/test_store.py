from datetime import date

import pytest

import store


@pytest.fixture
def db(tmp_path):
    return store.open_db(tmp_path / "t.db", dim=3)


def msg(id, content, channel_id=1):
    return {"id": id, "channel_id": channel_id, "author": "sam", "ts": "2026-07-01T10:00:00",
            "url": f"https://discord.com/channels/0/{channel_id}/{id}", "content": content}


def test_search_returns_nearest_first(db):
    store.add_messages(db, [msg(1, "clip"), msg(2, "cern"), msg(3, "food")],
                       [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    hits = store.search(db, [0.1, 0.9, 0], k=2)
    assert [h["id"] for h in hits] == [2, 1]
    assert hits[0]["content"] == "cern" and hits[0]["url"].endswith("/2")


def test_add_is_upsert(db):
    store.add_messages(db, [msg(1, "old")], [[1, 0, 0]])
    store.add_messages(db, [msg(1, "edited")], [[0, 1, 0]])
    hits = store.search(db, [0, 1, 0], k=5)
    assert [(h["id"], h["content"]) for h in hits] == [(1, "edited")]


def test_delete_removes_from_search(db):
    store.add_messages(db, [msg(1, "a"), msg(2, "b")], [[1, 0, 0], [0, 1, 0]])
    store.delete_message(db, 1)
    assert [h["id"] for h in store.search(db, [1, 0, 0], k=5)] == [2]


def test_last_id_per_channel(db):
    assert store.last_id(db, 1) is None
    store.add_messages(db, [msg(5, "a"), msg(9, "b"), msg(7, "c", channel_id=2)],
                       [[1, 0, 0]] * 3)
    assert store.last_id(db, 1) == 9
    assert store.last_id(db, 2) == 7


def test_data_survives_reopen(tmp_path):
    db = store.open_db(tmp_path / "t.db", dim=3)
    store.add_messages(db, [msg(1, "a")], [[1, 0, 0]])
    store.save_posting(db, "https://x/1", 42, "h", None)
    db.close()
    db = store.open_db(tmp_path / "t.db", dim=3)
    assert [h["id"] for h in store.search(db, [1, 0, 0], k=1)] == [1]
    assert store.get_posting(db, "https://x/1")["thread_id"] == 42


def test_posting_save_and_update(db):
    assert store.get_posting(db, "https://x/1") is None
    store.save_posting(db, "https://x/1", 42, "h1", date(2027, 1, 15))
    store.save_posting(db, "https://x/1", 42, "h2", date(2027, 1, 20))
    p = store.get_posting(db, "https://x/1")
    assert (p["thread_id"], p["hash"]) == (42, "h2")


def test_expired_postings(db):
    store.save_posting(db, "https://x/past", 1, "h", date(2026, 10, 1))
    store.save_posting(db, "https://x/future", 2, "h", date(2027, 1, 1))
    store.save_posting(db, "https://x/none", 3, "h", None)
    store.save_posting(db, "https://x/done", 4, "h", date(2026, 9, 1))
    store.mark_archived(db, "https://x/done")
    assert [p["url"] for p in store.expired_postings(db, date(2026, 11, 1))] == ["https://x/past"]
