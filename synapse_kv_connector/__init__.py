"""Deprecated alias: this package was renamed to ``semblend_kv_connector``.

Any import under the old name (including submodule paths) resolves to
the canonical module objects — never a duplicate instance — via a
meta-path alias. The alias will be removed in a future release.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_OLD = "synapse_kv_connector"
_NEW = "semblend_kv_connector"

warnings.warn(
    f"{_OLD} has been renamed to {_NEW}; update imports, this alias "
    "will be removed in a future release",
    DeprecationWarning,
    stacklevel=2,
)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, real_name):
        self._real_name = real_name

    def create_module(self, spec):
        # return the canonical module object so both names share one
        # instance (module-level state, isinstance checks)
        return importlib.import_module(self._real_name)

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _OLD and not fullname.startswith(_OLD + "."):
            return None
        real = _NEW + fullname[len(_OLD):]
        return importlib.util.spec_from_loader(fullname, _AliasLoader(real))


sys.meta_path.insert(0, _AliasFinder())
sys.modules[_OLD] = importlib.import_module(_NEW)
