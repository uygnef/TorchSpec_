import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FeatureHandle:
    sample_key: str
    mooncake_key: str
    tensor_shapes: dict[str, tuple[int, ...]]
    tensor_dtypes: dict[str, str]
    feature_schema_version: str
    created_at: float
    expires_at: float | None = None


@dataclass(frozen=True)
class FeatureIndexEntry:
    sample_key: str
    mooncake_key: str
    tensor_shapes_json: str
    tensor_dtypes_json: str
    feature_schema_version: str
    created_at: float
    last_access_at: float
    expires_at: float | None = None
    status: str = "ready"

    def to_handle(self) -> FeatureHandle:
        return FeatureHandle(
            sample_key=self.sample_key,
            mooncake_key=self.mooncake_key,
            tensor_shapes=_decode_shapes(self.tensor_shapes_json),
            tensor_dtypes=json.loads(self.tensor_dtypes_json),
            feature_schema_version=self.feature_schema_version,
            created_at=self.created_at,
            expires_at=self.expires_at,
        )


def _encode_shapes(shapes: dict[str, tuple[int, ...]]) -> str:
    return json.dumps({name: list(shape) for name, shape in shapes.items()}, sort_keys=True)


def _decode_shapes(payload: str) -> dict[str, tuple[int, ...]]:
    raw = json.loads(payload)
    return {name: tuple(shape) for name, shape in raw.items()}


class CacheManifest:
    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_manifest (
                    sample_key TEXT PRIMARY KEY,
                    mooncake_key TEXT NOT NULL,
                    tensor_shapes_json TEXT NOT NULL,
                    tensor_dtypes_json TEXT NOT NULL,
                    feature_schema_version TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_access_at REAL NOT NULL,
                    expires_at REAL,
                    status TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feature_manifest_last_access
                ON feature_manifest(last_access_at)
                """
            )

    def get(self, sample_key: str, *, touch: bool = True) -> Optional[FeatureHandle]:
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT sample_key, mooncake_key, tensor_shapes_json, tensor_dtypes_json,
                       feature_schema_version, created_at, last_access_at, expires_at, status
                FROM feature_manifest
                WHERE sample_key = ?
                """,
                (sample_key,),
            ).fetchone()
            if row is None:
                return None
            if touch:
                conn.execute(
                    "UPDATE feature_manifest SET last_access_at = ? WHERE sample_key = ?",
                    (now, sample_key),
                )
            return _row_to_entry(row).to_handle()

    def upsert(self, handle: FeatureHandle, status: str = "ready") -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feature_manifest (
                    sample_key, mooncake_key, tensor_shapes_json, tensor_dtypes_json,
                    feature_schema_version, created_at, last_access_at, expires_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sample_key) DO UPDATE SET
                    mooncake_key = excluded.mooncake_key,
                    tensor_shapes_json = excluded.tensor_shapes_json,
                    tensor_dtypes_json = excluded.tensor_dtypes_json,
                    feature_schema_version = excluded.feature_schema_version,
                    created_at = excluded.created_at,
                    last_access_at = excluded.last_access_at,
                    expires_at = excluded.expires_at,
                    status = excluded.status
                """,
                (
                    handle.sample_key,
                    handle.mooncake_key,
                    _encode_shapes(handle.tensor_shapes),
                    json.dumps(handle.tensor_dtypes, sort_keys=True),
                    handle.feature_schema_version,
                    handle.created_at,
                    now,
                    handle.expires_at,
                    status,
                ),
            )

    def touch(self, sample_key: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE feature_manifest SET last_access_at = ? WHERE sample_key = ?",
                (time.time(), sample_key),
            )

    def delete(self, sample_key: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM feature_manifest WHERE sample_key = ?", (sample_key,))


def _row_to_entry(row: sqlite3.Row) -> FeatureIndexEntry:
    return FeatureIndexEntry(
        sample_key=row["sample_key"],
        mooncake_key=row["mooncake_key"],
        tensor_shapes_json=row["tensor_shapes_json"],
        tensor_dtypes_json=row["tensor_dtypes_json"],
        feature_schema_version=row["feature_schema_version"],
        created_at=row["created_at"],
        last_access_at=row["last_access_at"],
        expires_at=row["expires_at"],
        status=row["status"],
    )
