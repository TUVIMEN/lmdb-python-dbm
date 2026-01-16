"""Python DBM style wrapper around LMDB (Lightning Memory-Mapped Database)"""

from .lmdbm import Lmdb, SizeError, GrowError, Error, open, LmdbJson, LmdbCompress

__version__ = "0.0.6"

__all__ = ["Lmdb", "Error", "SizeError", "GrowError", "open", "LmdbJson", "LmdbCompress", "__version__"]
