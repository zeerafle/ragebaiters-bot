from datetime import date

from ai import Posting, keep_valid, posting_hash

TODAY = date(2026, 11, 1)


def p(url="https://careers.cern/job/1", deadline=date(2027, 1, 15), **kw):
    return Posting(**{"title": "Summer Student", "organization": "CERN", "url": url,
                      "deadline": deadline, "location": "Geneva", "summary": "Physics", **kw})


def test_keep_valid_drops_expired_and_unseen_urls():
    run = b'[{"tool_result": "see https://careers.cern/job/1 and https://careers.cern/job/2"}]'
    kept = keep_valid([
        p(),                                                  # seen, open
        p(url="https://careers.cern/job/2", deadline=None),   # seen, no deadline: keep
        p(url="https://careers.cern/job/3"),                  # never seen: made up
        p(url="https://careers.cern/job/2", deadline=date(2026, 10, 31)),  # expired
    ], run, TODAY)
    assert [(k.url, k.deadline) for k in kept] == [
        ("https://careers.cern/job/1", date(2027, 1, 15)),
        ("https://careers.cern/job/2", None),
    ]


def test_deadline_today_is_still_open():
    assert keep_valid([p(deadline=TODAY)], b"https://careers.cern/job/1", TODAY)


def test_hash_ignores_summary_but_not_deadline():
    assert posting_hash(p()) == posting_hash(p(summary="Reworded by the model"))
    assert posting_hash(p()) != posting_hash(p(deadline=date(2027, 2, 1)))
    assert posting_hash(p()) != posting_hash(p(location="Remote"))


def test_hash_is_stable_sha256_hex():
    h = posting_hash(p())
    assert len(h) == 64 and int(h, 16) >= 0
    assert h == posting_hash(p())
