from pathlib import Path
import typing as t
import joblib
from logging import Logger

F = t.TypeVar("F", bound=t.Callable[..., t.Any])


class SkipOrExecute:
    def __init__(
        self, cache_dir: Path, key: str, f: F, logger: t.Optional[Logger] = None
    ) -> None:
        self._cache_dir = cache_dir
        self._key = key
        self._f = f
        self.logger = logger

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        path = self._cache_dir.joinpath(self._key)
        if path.exists():
            if self.logger is not None:
                self.logger.info(f"skip key={self._key}")
            return joblib.load(path)
        else:
            res: t.Any = self._f(*args, **kwargs)
            joblib.dump(res, path)
            return res


class Cache:
    def __init__(self, cache_dir: str, logger: t.Optional[Logger] = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logger

    def __call__(self, key: str, f: F) -> F:
        caller = SkipOrExecute(self.cache_dir, key, f, self.logger)
        return t.cast(F, caller)
