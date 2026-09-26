import datetime as dt
import zipfile

import pytest

from dispersion import ridb


@pytest.fixture
def fake_repo_zip(tmp_path):
    """A tiny archive laid out like GitHub's zip of the database repository."""
    z = tmp_path / "repo.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("refractiveindex.info-database-main/database/catalog-nk.yml", "[]\n")
        zf.writestr("refractiveindex.info-database-main/README.md", "not extracted\n")
    return z.as_uri()


@pytest.fixture
def root(tmp_path):
    return tmp_path / "data" / "database"


def test_first_download_then_throttled(root, fake_repo_zip):
    assert ridb.next_download_allowed(root) is None
    ridb.download_database(root, url=fake_repo_zip)
    assert ridb.database_available(root)
    assert not (root / "README.md").exists()

    allowed = ridb.next_download_allowed(root)
    assert allowed is not None
    expected = dt.datetime.now(dt.timezone.utc) + ridb.UPDATE_INTERVAL
    assert abs((allowed - expected).total_seconds()) < 60
    with pytest.raises(ridb.DownloadThrottled, match="once per 24 hours"):
        ridb.download_database(root, url=fake_repo_zip)


def test_allowed_again_after_a_day(root, fake_repo_zip):
    ridb.download_database(root, url=fake_repo_zip)
    old = ridb._format_time(dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=25))
    (root / ridb._STAMP_FILE).write_text(old, encoding="utf-8")
    (root.parent / ridb._ATTEMPT_FILE).write_text(old, encoding="utf-8")
    assert ridb.next_download_allowed(root) is None
    ridb.download_database(root, url=fake_repo_zip)


def test_failed_attempts_are_spaced(root, tmp_path):
    bad_url = (tmp_path / "missing.zip").as_uri()
    with pytest.raises(OSError):
        ridb.download_database(root, url=bad_url)
    assert not ridb.database_available(root)
    allowed = ridb.next_download_allowed(root)
    assert allowed is not None
    assert allowed - dt.datetime.now(dt.timezone.utc) <= ridb.RETRY_INTERVAL
    with pytest.raises(ridb.DownloadThrottled, match="every 10 minutes"):
        ridb.download_database(root, url=bad_url)


def test_legacy_stamp_format_is_understood(root, fake_repo_zip):
    ridb.download_database(root, url=fake_repo_zip)
    (root / ridb._STAMP_FILE).write_text("2026-09-23 18:38 UTC", encoding="utf-8")
    (root.parent / ridb._ATTEMPT_FILE).unlink()
    assert ridb.next_download_allowed(root) is None  # more than a day ago


def test_concurrent_download_refused(root, fake_repo_zip):
    assert ridb._download_lock.acquire(blocking=False)
    try:
        with pytest.raises(ridb.DownloadThrottled, match="already in progress"):
            ridb.download_database(root, url=fake_repo_zip)
    finally:
        ridb._download_lock.release()
