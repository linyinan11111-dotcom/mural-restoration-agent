from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from contextlib import contextmanager, closing
import json
import sqlite3
import uuid


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def new_id():
    return uuid.uuid4().hex


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.db = root / 'workbench.sqlite3'
        with self.connect() as con:
            con.execute('CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT PRIMARY KEY, document TEXT NOT NULL)')
            con.execute('CREATE INDEX IF NOT EXISTS records_kind ON records(kind)')
        for job in self.list('job'):
            if job['status'] in {'queued', 'planning', 'running', 'evaluating'}:
                self.update('job', job['id'], status='interrupted', error='应用已重启，可从检查点续跑。')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db, timeout=15)
        try:
            # WAL keeps the 900ms UI polling loop from blocking the worker thread's
            # writes; synchronous=NORMAL keeps commit latency low for local use.
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA synchronous=NORMAL')
            with connection:
                yield connection
        finally:
            connection.close()

    def put(self, kind, document):
        with self.lock, self.connect() as con:
            con.execute('INSERT OR REPLACE INTO records VALUES (?, ?, ?)',
                        (kind, document['id'], json.dumps(document, ensure_ascii=False)))
        return document

    def backup(self, destination):
        """Consistent SQLite snapshot; images are archived by the backup service."""
        with self.lock, self.connect() as source, closing(sqlite3.connect(destination)) as target:
            source.backup(target)

    def get(self, kind, identifier):
        with self.connect() as con:
            row = con.execute('SELECT document FROM records WHERE kind=? AND id=?', (kind, identifier)).fetchone()
        return json.loads(row[0]) if row else None

    def update(self, kind, identifier, **fields):
        with self.lock:
            document = self.get(kind, identifier)
            if document is None:
                raise KeyError(identifier)
            document.update(fields)
            document['updated_at'] = timestamp()
            return self.put(kind, document)

    def list(self, kind):
        with self.connect() as con:
            rows = con.execute('SELECT document FROM records WHERE kind=? ORDER BY rowid DESC', (kind,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def folder(self, identifier):
        # Only identifiers issued by this service can be used as a path component.
        if len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('无效的任务编号')
        path = self.root / identifier
        path.mkdir(exist_ok=True)
        return path

    def event(self, identifier, message):
        with self.lock:
            job = self.get('job', identifier)
            # Keep the newest 200 events; old ones only slow down every snapshot.
            events = (job.get('events', []) + [{'time': timestamp(), 'message': message}])[-200:]
            self.update('job', identifier, events=events)

