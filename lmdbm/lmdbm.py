import logging
from collections.abc import Mapping, MutableMapping
import gzip
from pathlib import Path
from typing import Any, Generic, Iterator, List, Optional, Tuple, TypeVar, Union, Callable
from threading import Lock
import json

import lmdb
from typing_extensions import Self

GenericT = TypeVar("GenericT")
KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")

_DEFAULT = object()

text_encoding = "utf8"


class Error(Exception):
    pass


class SizeError(Exception):
    pass


class GrowError(SizeError):
    pass


class MissingOk:
    # for python < 3.8 compatibility

    def __init__(self, ok: bool) -> None:
        self.ok = ok

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if isinstance(exc_value, FileNotFoundError) and self.ok:
            return True


def remove_lmdbm(path: str, missing_ok: bool = True) -> None:
    base = Path(path)
    with MissingOk(missing_ok):
        (base / "data.mdb").unlink()
    with MissingOk(missing_ok):
        (base / "lock.mdb").unlink()
    with MissingOk(missing_ok):
        base.rmdir()


def to_bytes(value):
    if isinstance(value, bytes):
        return value
    elif isinstance(value, str):
        return value.encode(text_encoding)

    raise TypeError(value)


def split_kwargs(keys, kwargs):
    found = {}
    other = {}

    for i in kwargs:
        if i in keys:
            found[i] = kwargs[i]
        else:
            other[i] = kwargs[i]

    return found, other


class Lmdb(MutableMapping, Generic[KeyT, ValueT]):
    autogrow_error = "Failed to grow LMDB ({}). Is there enough disk space available?"
    autogrow_msg = "Grew database (%s) map size to %s"

    def __init__(self, env: lmdb.Environment, autogrow: bool = True, logger: Callable = None, **kwargs) -> None:
        self.env = env
        self.autogrow = autogrow
        self.logger = logger

    @classmethod
    def open(
        cls,
        path: str,
        flag: str = "r",
        mode: int = 0o755,
        map_size: int = 2**20,
        **kwargs,
    ) -> "Lmdb":
        """
        Opens the database `path`.
        `flag`: r (read only, existing),
                w (read and write, existing),
                a (read, write, create if not exists)
                c (read, write, create if not exists),
                n (read, write, overwrite existing)

        `autogrow`: Automatically grow the database size when `map_size` is exceeded.
                WARNING: Set this to `False` for multi-process write access.
        `path`: Location of directory (if subdir=True) or file prefix to store the database.
        `mode`: File creation mode.
        `logger`: Function used for logging.
        `map_size`: Maximum size database may grow to; used to size the memory mapping. Defaults to 2**20 (1MB). If database grows larger than map_size, an exception will be raised and the user must close and reopen Environment. On 64-bit there is no penalty for making this huge (say 1TB). Must be <2GB on 32-bit.
            Note
            The default map size is set low to encourage a crash, so users can figure out a good value before learning about this option too late.
        `subdir`: If True, path refers to a subdirectory to store the data and lock files in, otherwise it refers to a filename prefix.
        `metasync`: If False, flush system buffers to disk only once per transaction, omit the metadata flush. Defer that until the system flushes files to disk, or next commit or sync().
            This optimization maintains database integrity, but a system crash may undo the last committed transaction. I.e. it preserves the ACI (atomicity, consistency, isolation) but not D (durability) database property.
        `sync`: If False, don’t flush system buffers to disk when committing a transaction. This optimization means a system crash can corrupt the database or lose the last transactions if buffers are not yet flushed to disk.
            The risk is governed by how often the system flushes dirty buffers to disk and how often sync() is called. However, if the filesystem preserves write order and writemap=False, transactions exhibit ACI (atomicity, consistency, isolation) properties and only lose D (durability). I.e. database integrity is maintained, but a system crash may undo the final transactions.
            Note that sync=False, writemap=True leaves the system with no hint for when to write transactions to disk, unless sync() is called. map_async=True, writemap=True may be preferable.
        `readahead`: If False, LMDB will disable the OS filesystem readahead mechanism, which may improve random read performance when a database is larger than RAM.
        `writemap`: If True, use a writeable memory map unless readonly=True. This is faster and uses fewer mallocs, but loses protection from application bugs like wild pointer writes and other bad updates into the database. Incompatible with nested transactions.
            Processes with and without writemap on the same environment do not cooperate well.
        `meminit`: If False LMDB will not zero-initialize buffers prior to writing them to disk. This improves performance but may cause old heap data to be written saved in the unused portion of the buffer. Do not use this option if your application manipulates confidential data (e.g. plaintext passwords) in memory. This option is only meaningful when writemap=False; new pages are always zero-initialized when writemap=True.
        `map_async`: When writemap=True, use asynchronous flushes to disk. As with sync=False, a system crash can then corrupt the database or lose the last transactions. Calling sync() ensures on-disk database integrity until next commit.
        `max_readers`: Maximum number of simultaneous read transactions. Can only be set by the first process to open an environment, as it affects the size of the lock file and shared memory area. Attempts to simultaneously start more than this many read transactions will fail.
        `max_spare_txns`: Read-only transactions to cache after becoming unused. Caching transactions avoids two allocations, one lock and linear scan of the shared environment per invocation of begin(), Transaction, get(), gets(), or cursor(). Should match the process’s maximum expected concurrent transactions (e.g. thread count).
        """

        settings, args = split_kwargs(
            {
                "lock",
                "sync",
                "readahead",
                "writemap",
                "meminit",
                "mode",
                "subdir",
                "metasync",
                "map_async",
                "max_readers",
                "max_spare_txns",
            },
            kwargs,
        )

        if flag == "r":  # Open existing database for reading only (default)
            env = lmdb.open(path, map_size=map_size, max_dbs=1, readonly=True, create=False, mode=mode, **settings)
        elif flag == "w":  # Open existing database for reading and writing
            env = lmdb.open(path, map_size=map_size, max_dbs=1, readonly=False, create=False, mode=mode, **settings)
        # what a retarded convention to name it 'c'
        elif flag == "a" or flag == "c":  # Open database for reading and writing, creating it if it doesn't exist
            env = lmdb.open(path, map_size=map_size, max_dbs=1, readonly=False, create=True, mode=mode, **settings)
        elif flag == "n":  # Always create a new, empty database, open for reading and writing
            remove_lmdbm(path)
            env = lmdb.open(path, map_size=map_size, max_dbs=1, readonly=False, create=True, mode=mode, **settings)
        else:
            raise ValueError("Invalid flag")

        return cls(env, **args)

    @property
    def map_size(self) -> int:
        return self.env.info()["map_size"]

    @map_size.setter
    def map_size(self, value: int) -> None:
        self.env.set_mapsize(value)

    def _pre_key(self, key: KeyT) -> bytes:
        return to_bytes(key)

    def _post_key(self, key: bytes) -> KeyT:
        return key

    def _pre_value(self, value: ValueT) -> bytes:
        return to_bytes(value)

    def _post_value(self, value: bytes) -> ValueT:
        return value

    def __getitem__(self, key: KeyT) -> ValueT:
        with self.env.begin() as txn:
            value = txn.get(self._pre_key(key))
        if value is None:
            raise KeyError(key)
        return self._post_value(value)

    def __setitem__(self, key: KeyT, value: ValueT) -> None:
        k = self._pre_key(key)
        v = self._pre_value(value)
        for _i in range(12):
            try:
                with self.env.begin(write=True) as txn:
                    txn.put(k, v)
                    return
            except lmdb.MapFullError:
                if not self.autogrow:
                    raise SizeError("Map size exceeded")
                new_map_size = self.map_size * 2
                self.map_size = new_map_size
                if self.logger is not None:
                    self.logger("{} {} {}".format(self.autogrow_msg, self.env.path(), new_map_size))

        raise GrowError(self.autogrow_error.format(self.env.path()))

    def __delitem__(self, key: KeyT) -> None:
        with self.env.begin(write=True) as txn:
            txn.delete(self._pre_key(key))

    def keys(self) -> Iterator[KeyT]:
        with self.env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                yield self._post_key(key)

    def items(self) -> Iterator[Tuple[KeyT, ValueT]]:
        with self.env.begin() as txn:
            for key, value in txn.cursor().iternext(keys=True, values=True):
                yield (self._post_key(key), self._post_value(value))

    def values(self) -> Iterator[ValueT]:
        with self.env.begin() as txn:
            for value in txn.cursor().iternext(keys=False, values=True):
                yield self._post_value(value)

    def __contains__(self, key: KeyT) -> bool:
        with self.env.begin() as txn:
            value = txn.get(self._pre_key(key))
        return value is not None

    def __iter__(self) -> Iterator[KeyT]:
        return self.keys()

    def __len__(self) -> int:
        with self.env.begin() as txn:
            return txn.stat()["entries"]

    def pop(self, key: KeyT, default: Union[ValueT, GenericT] = _DEFAULT) -> Union[ValueT, GenericT]:
        with self.env.begin(write=True) as txn:
            value = txn.pop(self._pre_key(key))
        if value is None:
            return default
        return self._post_value(value)

    def update(self, __other: Any = (), **kwds: ValueT) -> None:  # python3.8 only: update(self, other=(), /, **kwds)
        # fixme: `kwds`

        # note: benchmarking showed that there is no real difference between using lists or iterables
        # as input to `putmulti`.
        # lists: Finished 14412594 in 253496 seconds.
        # iter:  Finished 14412594 in 256315 seconds.

        # save generated lists in case the insert fails and needs to be retried
        # for performance reasons, but mostly because `__other` could be an iterable
        # which would already be exhausted on the second try
        pairs_other: Optional[List[Tuple[bytes, bytes]]] = None
        pairs_kwds: Optional[List[Tuple[bytes, bytes]]] = None

        for _i in range(12):
            try:
                with self.env.begin(write=True) as txn:
                    with txn.cursor() as curs:
                        if isinstance(__other, Mapping):
                            pairs_other = pairs_other or [
                                (self._pre_key(key), self._pre_value(__other[key])) for key in __other
                            ]
                            curs.putmulti(pairs_other)
                        elif hasattr(__other, "keys"):
                            pairs_other = pairs_other or [
                                (self._pre_key(key), self._pre_value(__other[key])) for key in __other.keys()
                            ]
                            curs.putmulti(pairs_other)
                        else:
                            pairs_other = pairs_other or [
                                (self._pre_key(key), self._pre_value(value)) for key, value in __other
                            ]
                            curs.putmulti(pairs_other)

                        pairs_kwds = pairs_kwds or [
                            (self._pre_key(key), self._pre_value(value)) for key, value in kwds.items()
                        ]
                        curs.putmulti(pairs_kwds)

                        return
            except lmdb.MapFullError:
                if not self.autogrow:
                    raise SizeError("Map size exceeded")
                new_map_size = self.map_size * 2
                self.map_size = new_map_size
                if self.logger is not None:
                    self.logger("{} {} {}".format(self.autogrow_msg, self.env.path(), new_map_size))

        raise GrowError(self.autogrow_error.format(self.env.path()))

    def sync(self) -> None:
        self.env.sync()

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args):
        self.close()


def LmdbJson(classtype=Lmdb):
    class entity(classtype):
        def _pre_value(self, value):
            value = json.dumps(value, separators=(",", ":")).encode(text_encoding)
            return super()._pre_value(value)

        def _post_value(self, value):
            value = super()._post_value(value)
            return json.loads(value)

    return entity


def LmdbCompress(classtype=Lmdb):
    class entity(classtype):
        def __init__(
            self, env, compressfunc=gzip.compress, decompressfunc=gzip.decompress, compresslevel: int = 9, **kwargs
        ):
            super().__init__(env, **kwargs)
            self.compresslevel = compresslevel
            self.compressfunc = compressfunc
            self.decompressfunc = decompressfunc

        def _pre_value(self, value: ValueT) -> bytes:
            value = self.compressfunc(to_bytes(value), self.compresslevel)
            return super()._pre_value(value)

        def _post_value(self, value: bytes) -> ValueT:
            value = self.decompressfunc(value)
            return super()._post_value(value)

    return entity


def chain(args):
    assert len(args) > 0
    ret = Lmdb
    for i in reversed(args):
        ret = i(ret)
    return ret


def open(path, flag="r", mode=0o755, classtype=Lmdb, **kwargs):
    if isinstance(classtype, list):
        classtype = chain(classtype)
    return classtype.open(path, flag, mode, **kwargs)
