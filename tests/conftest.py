from pathlib import Path

import pytest

from modulemate.db import Database, Store
from modulemate.ingest import parse_upload


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.sqlite3")


@pytest.fixture
def store(db):
    return Store(db, db.register("student", "a strong password"))


@pytest.fixture
def course(store):
    return store.create_course("Learning Lab")


@pytest.fixture
def populated(store, course):
    parsed = parse_upload("Learning Lab.md", Path("examples/learning-lab.md").read_bytes())
    document, _ = store.add_document(course, parsed)
    return course, document
