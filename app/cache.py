from pathlib import Path
import typing as t
import joblib

F = t.TypeVar("F", bound=t.Callable[..., t.Any])


class SkipOrExecute:
    def __init__(self, path: Path, f: F) -> None:
        self.path = path
        self.f = f

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if self.path.exists():
            return joblib.load(self.path)
        else:
            res: t.Any = self.f(*args, **kwargs)
            joblib.dump(res, self.path)
            return res


class Cache:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def __call__(self, key: str, f: F) -> F:
        path = self.cache_dir.joinpath(key)
        caller = SkipOrExecute(path, f)
        return t.cast(F, caller)
