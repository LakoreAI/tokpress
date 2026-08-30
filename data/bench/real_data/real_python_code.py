"""runpy.py - locating and running Python code using the module namespace

Provides support for locating and running Python scripts using the Python
module namespace instead of the native filesystem.

This allows Python code to play nicely with non-filesystem based PEP 302
importers when locating support scripts as well as when importing modules.
"""
# Written by Nick Coghlan <ncoghlan at gmail.com>
#    to implement PEP 338 (Executing Modules as Scripts)


import sys
import importlib.machinery # importlib first so we can test #15386 via -m
import importlib.util
import io
import os

__all__ = [
    "run_module", "run_path",
]

# avoid 'import types' just for ModuleType
ModuleType = type(sys)

class _TempModule(object):
    """Temporarily replace a module in sys.modules with an empty namespace"""
    def __init__(self, mod_name):
        self.mod_name = mod_name
        self.module = ModuleType(mod_name)
        self._saved_module = []

    def __enter__(self):
        mod_name = self.mod_name
        try:
            self._saved_module.append(sys.modules[mod_name])
        except KeyError:
            pass
        sys.modules[mod_name] = self.module
        return self

    def __exit__(self, *args):
        if self._saved_module:
            sys.modules[self.mod_name] = self._saved_module[0]
        else:
            del sys.modules[self.mod_name]
        self._saved_module = []

class _ModifiedArgv0(object):
    def __init__(self, value):
        self.value = value
        self._saved_value = self._sentinel = object()

    def __enter__(self):
        if self._saved_value is not self._sentinel:
            raise RuntimeError("Already preserving saved value")
        self._saved_value = sys.argv[0]
        sys.argv[0] = self.value

    def __exit__(self, *args):
        self.value = self._sentinel
        sys.argv[0] = self._saved_value

# TODO: Replace these helpers with importlib._bootstrap_external functions.
def _run_code(code, run_globals, init_globals=None,
              mod_name=None, mod_spec=None,
              pkg_name=None, script_name=None):
    """Helper to run code in nominated namespace"""
    if init_globals is not None:
        run_globals.update(init_globals)
    if mod_spec is None:
        loader = None
        fname = script_name
        cached = None
    else:
        loader = mod_spec.loader
        fname = mod_spec.origin
        cached = mod_spec.cached
        if pkg_name is None:
            pkg_name = mod_spec.parent
    run_globals.update(__name__ = mod_name,
                       __file__ = fname,
                       __cached__ = cached,
                       __doc__ = None,
                       __loader__ = loader,
                       __package__ = pkg_name,
                       __spec__ = mod_spec)
    exec(code, run_globals)
    return run_globals

def _run_module_code(code, init_globals=None,
                    mod_name=None, mod_spec=None,
                    pkg_name=None, script_name=None):
    """Helper to run code in new namespace with sys modified"""
    fname = script_name if mod_spec is None else mod_spec.origin
    with _TempModule(mod_name) as temp_module, _ModifiedArgv0(fname):
        mod_globals = temp_module.module.__dict__
        _run_code(code, mod_globals, init_globals,
                  mod_name, mod_spec, pkg_name, script_name)
    # Copy the globals of the temporary module, as they
    # may be cleared when the temporary module goes away
    return mod_globals.copy()

# Helper to get the full name, spec and code for a module
def _get_module_details(mod_name, error=ImportError):
    # name= is only accepted by ImportError and its subclasses.
    kwargs = {"name": mod_name} if issubclass(error, ImportError) else {}
    if mod_name.startswith("."):
        raise error("Relative module names not supported", **kwargs)
    pkg_name, _, _ = mod_name.rpartition(".")
    if pkg_name:
        # Try importing the parent to avoid catching initialization errors
        try:
            __import__(pkg_name)
        except ImportError as e:
            # If the parent or higher ancestor package is missing, let the
            # error be raised by find_spec() below and then be caught. But do
            # not allow other errors to be caught.
            if e.name is None or (e.name != pkg_name and
                    not pkg_name.startswith(e.name + ".")):
                raise
        # Warn if the module has already been imported under its normal name
        existing = sys.modules.get(mod_name)
        if existing is not None and not hasattr(existing, "__path__"):
            from warnings import warn
            msg = "{mod_name!r} found in sys.modules after import of " \
                "package {pkg_name!r}, but prior to execution of " \
                "{mod_name!r}; this may result in unpredictable " \
                "behaviour".format(mod_name=mod_name, pkg_name=pkg_name)
            warn(RuntimeWarning(msg))

    try:
        spec = importlib.util.find_spec(mod_name)
    except (ImportError, AttributeError, TypeError, ValueError) as ex:
        # This hack fixes an impedance mismatch between pkgutil and
        # importlib, where the latter raises other errors for cases where
        # pkgutil previously raised ImportError
        msg = "Error while finding module specification for {!r} ({}: {})"
        if mod_name.endswith(".py"):
            msg += (f". Try using '{mod_name[:-3]}' instead of "
                    f"'{mod_name}' as the module name.")
        raise error(msg.format(mod_name, type(ex).__name__, ex),
                    **kwargs) from ex
    if spec is None:
        raise error("No module named %s" % mod_name, **kwargs)
    if spec.submodule_search_locations is not None:
        if mod_name == "__main__" or mod_name.endswith(".__main__"):
            raise error("Cannot use package as __main__ module", **kwargs)
        try:
            pkg_main_name = mod_name + ".__main__"
            return _get_module_details(pkg_main_name, error)
        except error as e:
            if mod_name not in sys.modules:
                raise  # No module loaded; being a package is irrelevant
            raise error(("%s; %r is a package and cannot " +
                               "be directly executed") %(e, mod_name),
                        **kwargs)
    loader = spec.loader
    if loader is None:
        raise error("%r is a namespace package and cannot be executed"
                                                                 % mod_name,
                    **kwargs)
    try:
        code = loader.get_code(mod_name)
    except ImportError as e:
        raise error(format(e), **kwargs) from e
    if code is None:
        raise error("No code object available for %s" % mod_name, **kwargs)
    return mod_name, spec, code

class _Error(Exception):
    """Error that _run_module_as_main() should report without a traceback"""

# XXX ncoghlan: Should this be documented and made public?
# (Current thoughts: don't repeat the mistake that lead to its
# creation when run_module() no longer met the needs of
# mainmodule.c, but couldn't be changed because it was public)
def _run_module_as_main(mod_name, alter_argv=True):
    """Runs the designated module in the __main__ namespace

       Note that the executed module will have full access to the
       __main__ namespace. If this is not desirable, the run_module()
       function should be used to run the module code in a fresh namespace.

       At the very least, these variables in __main__ will be overwritten:
           __name__
           __file__
           __cached__
           __loader__
           __package__
    """
    try:
        if alter_argv or mod_name != "__main__": # i.e. -m switch
            mod_name, mod_spec, code = _get_module_details(mod_name, _Error)
        else:          # i.e. directory or zipfile execution
            mod_name, mod_spec, code = _get_main_module_details(_Error)
    except _Error as exc:
        msg = "%s: %s" % (sys.executable, exc)
        sys.exit(msg)
    main_globals = sys.modules["__main__"].__dict__
    if alter_argv:
        sys.argv[0] = mod_spec.origin
    return _run_code(code, main_globals, None,
                     "__main__", mod_spec)

def run_module(mod_name, init_globals=None,
               run_name=None, alter_sys=False):
    """Execute a module's code without importing it.

       mod_name -- an absolute module name or package name.

       Optional arguments:
       init_globals -- dictionary used to pre-populate the module’s
       globals dictionary before the code is executed.

       run_name -- if not None, this will be used for setting __name__;
       otherwise, __name__ will be set to mod_name + '__main__' if the
       named module is a package and to just mod_name otherwise.

       alter_sys -- if True, sys.argv[0] is updated with the value of
       __file__ and sys.modules[__name__] is updated with a temporary
       module object for the module being executed. Both are
       restored to their original values before the function returns.

       Returns the resulting module globals dictionary.
    """
    mod_name, mod_spec, code = _get_module_details(mod_name)
    if run_name is None:
        run_name = mod_name
    if alter_sys:
        return _run_module_code(code, init_globals, run_name, mod_spec)
    else:
        # Leave the sys module alone
        return _run_code(code, {}, init_globals, run_name, mod_spec)

def _get_main_module_details(error=ImportError):
    # Helper that gives a nicer error message when attempting to
    # execute a zipfile or directory by invoking __main__.py
    # Also moves the standard __main__ out of the way so that the
    # preexisting __loader__ entry doesn't cause issues
    main_name = "__main__"
    kwargs = {"name": main_name} if issubclass(error, ImportError) else {}
    saved_main = sys.modules[main_name]
    del sys.modules[main_name]
    try:
        return _get_module_details(main_name)
    except ImportError as exc:
        if main_name in str(exc):
            raise error("can't find %r module in %r" %
                              (main_name, sys.path[0]),
                        **kwargs) from exc
        raise
    finally:
        sys.modules[main_name] = saved_main


def _get_code_from_file(fname):
    # Check for a compiled file first
    from pkgutil import read_code
    code_path = os.path.abspath(fname)
    with io.open_code(code_path) as f:
        code = read_code(f)
    if code is None:
        # That didn't work, so try it as normal source code
        with io.open_code(code_path) as f:
            code = compile(f.read(), fname, 'exec')
    return code

def run_path(path_name, init_globals=None, run_name=None):
    """Execute code located at the specified filesystem location.

       path_name -- filesystem location of a Python script, zipfile,
       or directory containing a top level __main__.py script.

       Optional arguments:
       init_globals -- dictionary used to pre-populate the module’s
       globals dictionary before the code is executed.

       run_name -- if not None, this will be used to set __name__;
       otherwise, '<run_path>' will be used for __name__.

       Returns the resulting module globals dictionary.
    """
    if run_name is None:
        run_name = "<run_path>"
    pkg_name = run_name.rpartition(".")[0]
    from pkgutil import get_importer
    importer = get_importer(path_name)
    path_name = os.fsdecode(path_name)
    if isinstance(importer, type(None)):
        # Not a valid sys.path entry, so run the code directly
        # execfile() doesn't help as we want to allow compiled files
        code = _get_code_from_file(path_name)
        return _run_module_code(code, init_globals, run_name,
                                pkg_name=pkg_name, script_name=path_name)
    else:
        # Finder is defined for path, so add it to
        # the start of sys.path
        sys.path.insert(0, path_name)
        try:
            # Here's where things are a little different from the run_module
            # case. There, we only had to replace the module in sys while the
            # code was running and doing so was somewhat optional. Here, we
            # have no choice and we have to remove it even while we read the
            # code. If we don't do this, a __loader__ attribute in the
            # existing __main__ module may prevent location of the new module.
            mod_name, mod_spec, code = _get_main_module_details()
            with _TempModule(run_name) as temp_module, \
                 _ModifiedArgv0(path_name):
                mod_globals = temp_module.module.__dict__
                return _run_code(code, mod_globals, init_globals,
                                    run_name, mod_spec, pkg_name).copy()
        finally:
            try:
                sys.path.remove(path_name)
            except ValueError:
                pass


if __name__ == "__main__":
    # Run the module specified as the next command line argument
    if len(sys.argv) < 2:
        print("No module specified for execution", file=sys.stderr)
    else:
        del sys.argv[0] # Make the requested module sys.argv[0]
        _run_module_as_main(sys.argv[0])

"""Record of phased-in incompatible language changes.

Each line is of the form:

    FeatureName = "_Feature(" OptionalRelease "," MandatoryRelease ","
                              CompilerFlag ")"

where, normally, OptionalRelease < MandatoryRelease, and both are 5-tuples
of the same form as sys.version_info:

    (PY_MAJOR_VERSION, # the 2 in 2.1.0a3; an int
     PY_MINOR_VERSION, # the 1; an int
     PY_MICRO_VERSION, # the 0; an int
     PY_RELEASE_LEVEL, # "alpha", "beta", "candidate" or "final"; string
     PY_RELEASE_SERIAL # the 3; an int
    )

OptionalRelease records the first release in which

    from __future__ import FeatureName

was accepted.

In the case of MandatoryReleases that have not yet occurred,
MandatoryRelease predicts the release in which the feature will become part
of the language.

Else MandatoryRelease records when the feature became part of the language;
in releases at or after that, modules no longer need

    from __future__ import FeatureName

to use the feature in question, but may continue to use such imports.

MandatoryRelease may also be None, meaning that a planned feature got
dropped or that the release version is undetermined.

Instances of class _Feature have two corresponding methods,
.getOptionalRelease() and .getMandatoryRelease().

CompilerFlag is the (bitfield) flag that should be passed in the fourth
argument to the builtin function compile() to enable the feature in
dynamically compiled code.  This flag is stored in the .compiler_flag
attribute on _Future instances.  These values must match the appropriate
#defines of CO_xxx flags in Include/cpython/compile.h.

No feature line is ever to be deleted from this file.
"""

all_feature_names = [
    "nested_scopes",
    "generators",
    "division",
    "absolute_import",
    "with_statement",
    "print_function",
    "unicode_literals",
    "barry_as_FLUFL",
    "generator_stop",
    "annotations",
]

__all__ = ["all_feature_names"] + all_feature_names

# The CO_xxx symbols are defined here under the same names defined in
# code.h and used by compile.h, so that an editor search will find them here.
# However, they're not exported in __all__, because they don't really belong to
# this module.
CO_NESTED = 0x0010                      # nested_scopes
CO_GENERATOR_ALLOWED = 0                # generators (obsolete, was 0x1000)
CO_FUTURE_DIVISION = 0x20000            # division
CO_FUTURE_ABSOLUTE_IMPORT = 0x40000     # perform absolute imports by default
CO_FUTURE_WITH_STATEMENT = 0x80000      # with statement
CO_FUTURE_PRINT_FUNCTION = 0x100000     # print function
CO_FUTURE_UNICODE_LITERALS = 0x200000   # unicode string literals
CO_FUTURE_BARRY_AS_BDFL = 0x400000
CO_FUTURE_GENERATOR_STOP = 0x800000     # StopIteration becomes RuntimeError in generators
CO_FUTURE_ANNOTATIONS = 0x1000000       # annotations become strings at runtime


class _Feature:

    def __init__(self, optionalRelease, mandatoryRelease, compiler_flag):
        self.optional = optionalRelease
        self.mandatory = mandatoryRelease
        self.compiler_flag = compiler_flag

    def getOptionalRelease(self):
        """Return first release in which this feature was recognized.

        This is a 5-tuple, of the same form as sys.version_info.
        """
        return self.optional

    def getMandatoryRelease(self):
        """Return release in which this feature will become mandatory.

        This is a 5-tuple, of the same form as sys.version_info, or, if
        the feature was dropped, or the release date is undetermined, is None.
        """
        return self.mandatory

    def __repr__(self):
        return "_Feature" + repr((self.optional,
                                  self.mandatory,
                                  self.compiler_flag))


nested_scopes = _Feature((2, 1, 0, "beta",  1),
                         (2, 2, 0, "alpha", 0),
                         CO_NESTED)

generators = _Feature((2, 2, 0, "alpha", 1),
                      (2, 3, 0, "final", 0),
                      CO_GENERATOR_ALLOWED)

division = _Feature((2, 2, 0, "alpha", 2),
                    (3, 0, 0, "alpha", 0),
                    CO_FUTURE_DIVISION)

absolute_import = _Feature((2, 5, 0, "alpha", 1),
                           (3, 0, 0, "alpha", 0),
                           CO_FUTURE_ABSOLUTE_IMPORT)

with_statement = _Feature((2, 5, 0, "alpha", 1),
                          (2, 6, 0, "alpha", 0),
                          CO_FUTURE_WITH_STATEMENT)

print_function = _Feature((2, 6, 0, "alpha", 2),
                          (3, 0, 0, "alpha", 0),
                          CO_FUTURE_PRINT_FUNCTION)

unicode_literals = _Feature((2, 6, 0, "alpha", 2),
                            (3, 0, 0, "alpha", 0),
                            CO_FUTURE_UNICODE_LITERALS)

barry_as_FLUFL = _Feature((3, 1, 0, "alpha", 2),
                          (4, 0, 0, "alpha", 0),
                          CO_FUTURE_BARRY_AS_BDFL)

generator_stop = _Feature((3, 5, 0, "beta", 1),
                          (3, 7, 0, "alpha", 0),
                          CO_FUTURE_GENERATOR_STOP)

annotations = _Feature((3, 7, 0, "beta", 1),
                       None,
                       CO_FUTURE_ANNOTATIONS)

r"""OS routines for NT or Posix depending on what system we're on.

This exports:
  - all functions from posix or nt, e.g. unlink, stat, etc.
  - os.path is either posixpath or ntpath
  - os.name is either 'posix' or 'nt'
  - os.curdir is a string representing the current directory (always '.')
  - os.pardir is a string representing the parent directory (always '..')
  - os.sep is the (or a most common) pathname separator ('/' or '\\')
  - os.extsep is the extension separator (always '.')
  - os.altsep is the alternate pathname separator (None or '/')
  - os.pathsep is the component separator used in $PATH etc
  - os.linesep is the line separator in text files ('\n' or '\r\n')
  - os.defpath is the default search path for executables
  - os.devnull is the file path of the null device ('/dev/null', etc.)

Programs that import and use 'os' stand a better chance of being
portable between different platforms.  Of course, they must then
only use functions that are defined by all platforms (e.g., unlink
and opendir), and leave all pathname manipulation to os.path
(e.g., split and join).
"""

#'
import abc
import sys
import stat as st

from _collections_abc import _check_methods

GenericAlias = type(list[int])

_names = sys.builtin_module_names

# Note:  more names are added to __all__ later.
__all__ = ["altsep", "curdir", "pardir", "sep", "pathsep", "linesep",
           "defpath", "name", "path", "devnull", "SEEK_SET", "SEEK_CUR",
           "SEEK_END", "fsencode", "fsdecode", "get_exec_path", "fdopen",
           "extsep"]

def _exists(name):
    return name in globals()

def _get_exports_list(module):
    try:
        return list(module.__all__)
    except AttributeError:
        return [n for n in dir(module) if n[0] != '_']

# Any new dependencies of the os module and/or changes in path separator
# requires updating importlib as well.
if 'posix' in _names:
    name = 'posix'
    linesep = '\n'
    from posix import *
    try:
        from posix import _exit
        __all__.append('_exit')
    except ImportError:
        pass
    import posixpath as path

    try:
        from posix import _have_functions
    except ImportError:
        pass
    try:
        from posix import _create_environ
    except ImportError:
        pass

    import posix
    __all__.extend(_get_exports_list(posix))
    del posix

elif 'nt' in _names:
    name = 'nt'
    linesep = '\r\n'
    from nt import *
    try:
        from nt import _exit
        __all__.append('_exit')
    except ImportError:
        pass
    import ntpath as path

    import nt
    __all__.extend(_get_exports_list(nt))
    del nt

    try:
        from nt import _have_functions
    except ImportError:
        pass
    try:
        from nt import _create_environ
    except ImportError:
        pass

else:
    raise ImportError('no os specific module found')

sys.modules['os.path'] = path
from os.path import (curdir, pardir, sep, pathsep, defpath, extsep, altsep,
    devnull)

del _names


if _exists("_have_functions"):
    _globals = globals()
    def _add(str, fn):
        if (fn in _globals) and (str in _have_functions):
            _set.add(_globals[fn])

    _set = set()
    _add("HAVE_FACCESSAT",  "access")
    _add("HAVE_FCHMODAT",   "chmod")
    _add("HAVE_FCHOWNAT",   "chown")
    _add("HAVE_FSTATAT",    "stat")
    _add("HAVE_LSTAT",      "lstat")
    _add("HAVE_FUTIMESAT",  "utime")
    _add("HAVE_LINKAT",     "link")
    _add("HAVE_MKDIRAT",    "mkdir")
    _add("HAVE_MKFIFOAT",   "mkfifo")
    _add("HAVE_MKNODAT",    "mknod")
    _add("HAVE_OPENAT",     "open")
    _add("HAVE_READLINKAT", "readlink")
    _add("HAVE_RENAMEAT",   "rename")
    _add("HAVE_SYMLINKAT",  "symlink")
    _add("HAVE_UNLINKAT",   "unlink")
    _add("HAVE_UNLINKAT",   "rmdir")
    _add("HAVE_UTIMENSAT",  "utime")
    supports_dir_fd = _set

    _set = set()
    _add("HAVE_FACCESSAT",  "access")
    supports_effective_ids = _set

    _set = set()
    _add("HAVE_FCHDIR",     "chdir")
    _add("HAVE_FCHMOD",     "chmod")
    _add("MS_WINDOWS",      "chmod")
    _add("HAVE_FCHOWN",     "chown")
    _add("HAVE_FDOPENDIR",  "listdir")
    _add("HAVE_FDOPENDIR",  "scandir")
    _add("HAVE_FEXECVE",    "execve")
    _set.add(stat) # fstat always works
    _add("HAVE_FTRUNCATE",  "truncate")
    _add("HAVE_FUTIMENS",   "utime")
    _add("HAVE_FUTIMES",    "utime")
    _add("HAVE_FPATHCONF",  "pathconf")
    if _exists("statvfs") and _exists("fstatvfs"): # mac os x10.3
        _add("HAVE_FSTATVFS", "statvfs")
    supports_fd = _set

    _set = set()
    _add("HAVE_FACCESSAT",  "access")
    # Some platforms don't support lchmod().  Often the function exists
    # anyway, as a stub that always returns ENOSUP or perhaps EOPNOTSUPP.
    # (No, I don't know why that's a good design.)  ./configure will detect
    # this and reject it--so HAVE_LCHMOD still won't be defined on such
    # platforms.  This is Very Helpful.
    #
    # However, sometimes platforms without a working lchmod() *do* have
    # fchmodat().  (Examples: Linux kernel 3.2 with glibc 2.15,
    # OpenIndiana 3.x.)  And fchmodat() has a flag that theoretically makes
    # it behave like lchmod().  So in theory it would be a suitable
    # replacement for lchmod().  But when lchmod() doesn't work, fchmodat()'s
    # flag doesn't work *either*.  Sadly ./configure isn't sophisticated
    # enough to detect this condition--it only determines whether or not
    # fchmodat() minimally works.
    #
    # Therefore we simply ignore fchmodat() when deciding whether or not
    # os.chmod supports follow_symlinks.  Just checking lchmod() is
    # sufficient.  After all--if you have a working fchmodat(), your
    # lchmod() almost certainly works too.
    #
    # _add("HAVE_FCHMODAT",   "chmod")
    _add("HAVE_FCHOWNAT",   "chown")
    _add("HAVE_FSTATAT",    "stat")
    _add("HAVE_LCHFLAGS",   "chflags")
    _add("HAVE_LCHMOD",     "chmod")
    _add("MS_WINDOWS",      "chmod")
    if _exists("lchown"): # mac os x10.3
        _add("HAVE_LCHOWN", "chown")
    _add("HAVE_LINKAT",     "link")
    _add("HAVE_LUTIMES",    "utime")
    _add("HAVE_LSTAT",      "stat")
    _add("HAVE_FSTATAT",    "stat")
    _add("HAVE_UTIMENSAT",  "utime")
    _add("MS_WINDOWS",      "stat")
    supports_follow_symlinks = _set

    del _set
    del _have_functions
    del _globals
    del _add


# Python uses fixed values for the SEEK_ constants; they are mapped
# to native constants if necessary in posixmodule.c
# Other possible SEEK values are directly imported from posixmodule.c
SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2

# Super directory utilities.
# (Inspired by Eric Raymond; the doc strings are mostly his)

def makedirs(name, mode=0o777, exist_ok=False):
    """makedirs(name [, mode=0o777][, exist_ok=False])

    Super-mkdir; create a leaf directory and all intermediate ones.  Works
    like mkdir, except that any intermediate path segment (not just the
    rightmost) will be created if it does not exist.  If the target
    directory already exists, raise an OSError if exist_ok is False.
    Otherwise no exception is raised.  This is recursive.

    """
    head, tail = path.split(name)
    if not tail:
        head, tail = path.split(head)
    if head and tail and not path.exists(head):
        try:
            makedirs(head, exist_ok=exist_ok)
        except FileExistsError:
            # Defeats race condition when another thread created the path
            pass
        cdir = curdir
        if isinstance(tail, bytes):
            cdir = bytes(curdir, 'ASCII')
        if tail == cdir:           # xxx/newdir/. exists if xxx/newdir exists
            return
    try:
        mkdir(name, mode)
    except OSError:
        # Cannot rely on checking for EEXIST, since the operating system
        # could give priority to other errors like EACCES or EROFS
        if not exist_ok or not path.isdir(name):
            raise

def removedirs(name):
    """removedirs(name)

    Super-rmdir; remove a leaf directory and all empty intermediate
    ones.  Works like rmdir except that, if the leaf directory is
    successfully removed, directories corresponding to rightmost path
    segments will be pruned away until either the whole path is
    consumed or an error occurs.  Errors during this latter phase are
    ignored -- they generally mean that a directory was not empty.

    """
    rmdir(name)
    head, tail = path.split(name)
    if not tail:
        head, tail = path.split(head)
    while head and tail:
        try:
            rmdir(head)
        except OSError:
            break
        head, tail = path.split(head)

def renames(old, new):
    """renames(old, new)

    Super-rename; create directories as necessary and delete any left
    empty.  Works like rename, except creation of any intermediate
    directories needed to make the new pathname good is attempted
    first.  After the rename, directories corresponding to rightmost
    path segments of the old name will be pruned until either the
    whole path is consumed or a nonempty directory is found.

    Note: this function can fail with the new directory structure made
    if you lack permissions needed to unlink the leaf directory or
    file.

    """
    head, tail = path.split(new)
    if head and tail and not path.exists(head):
        makedirs(head)
    rename(old, new)
    head, tail = path.split(old)
    if head and tail:
        try:
            removedirs(head)
        except OSError:
            pass

__all__.extend(["makedirs", "removedirs", "renames"])

# Private sentinel that makes walk() classify all symlinks and junctions as
# regular files.
_walk_symlinks_as_files = object()

def walk(top, topdown=True, onerror=None, followlinks=False):
    """Directory tree generator.

    For each directory in the directory tree rooted at top (including top
    itself, but excluding '.' and '..'), yields a 3-tuple

        dirpath, dirnames, filenames

    dirpath is a string, the path to the directory.  dirnames is a list of
    the names of the subdirectories in dirpath (including symlinks to
    directories, and excluding '.' and '..').
    filenames is a list of the names of the non-directory files in dirpath.
    Note that the names in the lists are just names, with no path
    components.  To get a full path (which begins with top) to a file or
    directory in dirpath, do os.path.join(dirpath, name).

    If optional arg 'topdown' is true or not specified, the triple for a
    directory is generated before the triples for any of its subdirectories
    (directories are generated top down).  If topdown is false, the triple
    for a directory is generated after the triples for all of its
    subdirectories (directories are generated bottom up).

    When topdown is true, the caller can modify the dirnames list in-place
    (e.g., via del or slice assignment), and walk will only recurse into the
    subdirectories whose names remain in dirnames; this can be used to prune
    the search, or to impose a specific order of visiting.  Modifying
    dirnames when topdown is false has no effect on the behavior of
    os.walk(), since the directories in dirnames have already been generated
    by the time dirnames itself is generated. No matter the value of
    topdown, the list of subdirectories is retrieved before the tuples for
    the directory and its subdirectories are generated.

    By default errors from the os.scandir() call are ignored.  If
    optional arg 'onerror' is specified, it should be a function; it
    will be called with one argument, an OSError instance.  It can
    report the error to continue with the walk, or raise the exception
    to abort the walk.  Note that the filename is available as the
    filename attribute of the exception object.

    By default, os.walk does not follow symbolic links to subdirectories on
    systems that support them.  In order to get this functionality, set the
    optional argument 'followlinks' to true.

    Caution:  if you pass a relative pathname for top, don't change the
    current working directory between resumptions of walk.  walk never
    changes the current directory, and assumes that the client doesn't
    either.

    Example:

    import os
    from os.path import join, getsize
    for root, dirs, files in os.walk('python/Lib/xml'):
        print(root, "consumes ")
        print(sum(getsize(join(root, name)) for name in files), end=" ")
        print("bytes in", len(files), "non-directory files")
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')  # don't visit __pycache__ directories

    """
    sys.audit("os.walk", top, topdown, onerror, followlinks)

    stack = [fspath(top)]
    islink, join = path.islink, path.join
    while stack:
        top = stack.pop()
        if isinstance(top, tuple):
            yield top
            continue

        dirs = []
        nondirs = []
        walk_dirs = []

        # We may not have read permission for top, in which case we can't
        # get a list of the files the directory contains.
        # We suppress the exception here, rather than blow up for a
        # minor reason when (say) a thousand readable directories are still
        # left to visit.
        try:
            with scandir(top) as entries:
                for entry in entries:
                    try:
                        if followlinks is _walk_symlinks_as_files:
                            is_dir = entry.is_dir(follow_symlinks=False) and not entry.is_junction()
                        else:
                            is_dir = entry.is_dir()
                    except OSError:
                        # If is_dir() raises an OSError, consider the entry not to
                        # be a directory, same behaviour as os.path.isdir().
                        is_dir = False

                    if is_dir:
                        dirs.append(entry.name)
                    else:
                        nondirs.append(entry.name)

                    if not topdown and is_dir:
                        # Bottom-up: traverse into sub-directory, but exclude
                        # symlinks to directories if followlinks is False
                        if followlinks:
                            walk_into = True
                        else:
                            try:
                                is_symlink = entry.is_symlink()
                            except OSError:
                                # If is_symlink() raises an OSError, consider the
                                # entry not to be a symbolic link, same behaviour
                                # as os.path.islink().
                                is_symlink = False
                            walk_into = not is_symlink

                        if walk_into:
                            walk_dirs.append(entry.path)
        except OSError as error:
            if onerror is not None:
                onerror(error)
            continue

        if topdown:
            # Yield before sub-directory traversal if going top down
            yield top, dirs, nondirs
            # Traverse into sub-directories
            for dirname in reversed(dirs):
                new_path = join(top, dirname)
                # bpo-23605: os.path.islink() is used instead of caching
                # entry.is_symlink() result during the loop on os.scandir() because
                # the caller can replace the directory entry during the "yield"
                # above.
                if followlinks or not islink(new_path):
                    stack.append(new_path)
        else:
            # Yield after sub-directory traversal if going bottom up
            stack.append((top, dirs, nondirs))
            # Traverse into sub-directories
            for new_path in reversed(walk_dirs):
                stack.append(new_path)

__all__.append("walk")

if {open, stat} <= supports_dir_fd and {scandir, stat} <= supports_fd:

    def fwalk(top=".", topdown=True, onerror=None, *, follow_symlinks=False, dir_fd=None):
        """Directory tree generator.

        This behaves exactly like walk(), except that it yields a 4-tuple

            dirpath, dirnames, filenames, dirfd

        `dirpath`, `dirnames` and `filenames` are identical to walk() output,
        and `dirfd` is a file descriptor referring to the directory `dirpath`.

        The advantage of fwalk() over walk() is that it's safe against symlink
        races (when follow_symlinks is False).

        If dir_fd is not None, it should be a file descriptor open to
        a directory, and top should be relative; top will then be relative to
        that directory.  (dir_fd is always supported for fwalk.)

        Caution:
        Since fwalk() yields file descriptors, those are only valid until the
        next iteration step, so you should dup() them if you want to keep them
        for a longer period.

        Example:

        import os
        for root, dirs, files, rootfd in os.fwalk('python/Lib/xml'):
            print(root, "consumes", end="")
            print(sum(os.stat(name, dir_fd=rootfd).st_size for name in files),
                  end="")
            print("bytes in", len(files), "non-directory files")
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')  # don't visit __pycache__ directories
        """
        sys.audit("os.fwalk", top, topdown, onerror, follow_symlinks, dir_fd)
        top = fspath(top)
        stack = [(_fwalk_walk, (True, dir_fd, top, top, None))]
        isbytes = isinstance(top, bytes)
        try:
            while stack:
                yield from _fwalk(stack, isbytes, topdown, onerror, follow_symlinks)
        finally:
            # Close any file descriptors still on the stack.
            while stack:
                action, value = stack.pop()
                if action == _fwalk_close:
                    close(value)

    # Each item in the _fwalk() stack is a pair (action, args).
    _fwalk_walk = 0  # args: (isroot, dirfd, toppath, topname, entry)
    _fwalk_yield = 1  # args: (toppath, dirnames, filenames, topfd)
    _fwalk_close = 2  # args: dirfd

    def _fwalk(stack, isbytes, topdown, onerror, follow_symlinks):
        # Note: This uses O(depth of the directory tree) file descriptors: if
        # necessary, it can be adapted to only require O(1) FDs, see issue
        # #13734.

        action, value = stack.pop()
        if action == _fwalk_close:
            close(value)
            return
        elif action == _fwalk_yield:
            yield value
            return
        assert action == _fwalk_walk
        isroot, dirfd, toppath, topname, entry = value
        try:
            if not follow_symlinks:
                # Note: To guard against symlink races, we use the standard
                # lstat()/open()/fstat() trick.
                if entry is None:
                    orig_st = stat(topname, follow_symlinks=False, dir_fd=dirfd)
                else:
                    orig_st = entry.stat(follow_symlinks=False)
            topfd = open(topname, O_RDONLY | O_NONBLOCK, dir_fd=dirfd)
        except OSError as err:
            if isroot:
                raise
            if onerror is not None:
                onerror(err)
            return
        stack.append((_fwalk_close, topfd))
        if not follow_symlinks:
            if isroot and not st.S_ISDIR(orig_st.st_mode):
                return
            if not path.samestat(orig_st, stat(topfd)):
                return

        scandir_it = scandir(topfd)
        dirs = []
        nondirs = []
        entries = None if topdown or follow_symlinks else []
        for entry in scandir_it:
            name = entry.name
            if isbytes:
                name = fsencode(name)
            try:
                if entry.is_dir():
                    dirs.append(name)
                    if entries is not None:
                        entries.append(entry)
                else:
                    nondirs.append(name)
            except OSError:
                try:
                    # Add dangling symlinks, ignore disappeared files
                    if entry.is_symlink():
                        nondirs.append(name)
                except OSError:
                    pass

        if topdown:
            yield toppath, dirs, nondirs, topfd
        else:
            stack.append((_fwalk_yield, (toppath, dirs, nondirs, topfd)))

        toppath = path.join(toppath, toppath[:0])  # Add trailing slash.
        if entries is None:
            stack.extend(
                (_fwalk_walk, (False, topfd, toppath + name, name, None))
                for name in dirs[::-1])
        else:
            stack.extend(
                (_fwalk_walk, (False, topfd, toppath + name, name, entry))
                for name, entry in zip(dirs[::-1], entries[::-1]))

    __all__.append("fwalk")

def execl(file, *args):
    """execl(file, *args)

    Execute the executable file with argument list args, replacing the
    current process. """
    execv(file, args)

def execle(file, *args):
    """execle(file, *args, env)

    Execute the executable file with argument list args and
    environment env, replacing the current process. """
    env = args[-1]
    execve(file, args[:-1], env)

def execlp(file, *args):
    """execlp(file, *args)

    Execute the executable file (which is searched for along $PATH)
    with argument list args, replacing the current process. """
    execvp(file, args)

def execlpe(file, *args):
    """execlpe(file, *args, env)

    Execute the executable file (which is searched for along $PATH)
    with argument list args and environment env, replacing the current
    process. """
    env = args[-1]
    execvpe(file, args[:-1], env)

def execvp(file, args):
    """execvp(file, args)

    Execute the executable file (which is searched for along $PATH)
    with argument list args, replacing the current process.
    args may be a list or tuple of strings. """
    _execvpe(file, args)

def execvpe(file, args, env):
    """execvpe(file, args, env)

    Execute the executable file (which is searched for along $PATH)
    with argument list args and environment env, replacing the
    current process.
    args may be a list or tuple of strings. """
    _execvpe(file, args, env)

__all__.extend(["execl","execle","execlp","execlpe","execvp","execvpe"])

def _execvpe(file, args, env=None):
    if env is not None:
        exec_func = execve
        argrest = (args, env)
    else:
        exec_func = execv
        argrest = (args,)
        env = environ

    if path.dirname(file):
        exec_func(file, *argrest)
        return
    saved_exc = None
    path_list = get_exec_path(env)
    if name != 'nt':
        file = fsencode(file)
        path_list = map(fsencode, path_list)
    for dir in path_list:
        fullname = path.join(dir, file)
        try:
            exec_func(fullname, *argrest)
        except (FileNotFoundError, NotADirectoryError) as e:
            last_exc = e
        except OSError as e:
            last_exc = e
            if saved_exc is None:
                saved_exc = e
    if saved_exc is not None:
        raise saved_exc
    raise last_exc


def get_exec_path(env=None):
    """Returns the sequence of directories that will be searched for the
    named executable (similar to a shell) when launching a process.

    *env* must be an environment variable dict or None.  If *env* is None,
    os.environ will be used.
    """
    # Use a local import instead of a global import to limit the number of
    # modules loaded at startup: the os module is always loaded at startup by
    # Python. It may also avoid a bootstrap issue.
    import warnings

    if env is None:
        env = environ

    # {b'PATH': ...}.get('PATH') and {'PATH': ...}.get(b'PATH') emit a
    # BytesWarning when using python -b or python -bb: ignore the warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BytesWarning)

        try:
            path_list = env.get('PATH')
        except TypeError:
            path_list = None

        if supports_bytes_environ:
            try:
                path_listb = env[b'PATH']
            except (KeyError, TypeError):
                pass
            else:
                if path_list is not None:
                    raise ValueError(
                        "env cannot contain 'PATH' and b'PATH' keys")
                path_list = path_listb

            if path_list is not None and isinstance(path_list, bytes):
                path_list = fsdecode(path_list)

    if path_list is None:
        path_list = defpath
    return path_list.split(pathsep)


# Change environ to automatically call putenv() and unsetenv()
from _collections_abc import MutableMapping, Mapping

class _Environ(MutableMapping):
    def __init__(self, data, encodekey, decodekey, encodevalue, decodevalue):
        self.encodekey = encodekey
        self.decodekey = decodekey
        self.encodevalue = encodevalue
        self.decodevalue = decodevalue
        self._data = data

    def __getitem__(self, key):
        try:
            value = self._data[self.encodekey(key)]
        except KeyError:
            # raise KeyError with the original key value
            raise KeyError(key) from None
        return self.decodevalue(value)

    def __setitem__(self, key, value):
        key = self.encodekey(key)
        value = self.encodevalue(value)
        putenv(key, value)
        self._data[key] = value

    def __delitem__(self, key):
        encodedkey = self.encodekey(key)
        unsetenv(encodedkey)
        try:
            del self._data[encodedkey]
        except KeyError:
            # raise KeyError with the original key value
            raise KeyError(key) from None

    def __iter__(self):
        # list() from dict object is an atomic operation
        keys = list(self._data)
        for key in keys:
            yield self.decodekey(key)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        formatted_items = ", ".join(
            f"{self.decodekey(key)!r}: {self.decodevalue(value)!r}"
            for key, value in self._data.items()
        )
        return f"environ({{{formatted_items}}})"

    def copy(self):
        return dict(self)

    def setdefault(self, key, value):
        if key not in self:
            self[key] = value
        return self[key]

    def __ior__(self, other):
        self.update(other)
        return self

    def __or__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        new = dict(self)
        new.update(other)
        return new

    def __ror__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        new = dict(other)
        new.update(self)
        return new

def _create_environ_mapping():
    if name == 'nt':
        # Where Env Var Names Must Be UPPERCASE
        def check_str(value):
            if not isinstance(value, str):
                raise TypeError("str expected, not %s" % type(value).__name__)
            return value
        encode = check_str
        decode = str
        def encodekey(key):
            return encode(key).upper()
        data = {}
        for key, value in environ.items():
            data[encodekey(key)] = value
    else:
        # Where Env Var Names Can Be Mixed Case
        encoding = sys.getfilesystemencoding()
        def encode(value):
            if not isinstance(value, str):
                raise TypeError("str expected, not %s" % type(value).__name__)
            return value.encode(encoding, 'surrogateescape')
        def decode(value):
            return value.decode(encoding, 'surrogateescape')
        encodekey = encode
        data = environ
    return _Environ(data,
        encodekey, decode,
        encode, decode)

# unicode environ
environ = _create_environ_mapping()
del _create_environ_mapping


if _exists("_create_environ"):
    def reload_environ():
        data = _create_environ()
        if name == 'nt':
            encodekey = environ.encodekey
            data = {encodekey(key): value
                    for key, value in data.items()}

        # modify in-place to keep os.environb in sync
        env_data = environ._data
        env_data.clear()
        env_data.update(data)

    __all__.append("reload_environ")

def getenv(key, default=None):
    """Get an environment variable, return None if it doesn't exist.
    The optional second argument can specify an alternate default.
    key, default and the result are str."""
    return environ.get(key, default)

supports_bytes_environ = (name != 'nt')
__all__.extend(("getenv", "supports_bytes_environ"))

if supports_bytes_environ:
    def _check_bytes(value):
        if not isinstance(value, bytes):
            raise TypeError("bytes expected, not %s" % type(value).__name__)
        return value

    # bytes environ
    environb = _Environ(environ._data,
        _check_bytes, bytes,
        _check_bytes, bytes)
    del _check_bytes

    def getenvb(key, default=None):
        """Get an environment variable, return None if it doesn't exist.
        The optional second argument can specify an alternate default.
        key, default and the result are bytes."""
        return environb.get(key, default)

    __all__.extend(("environb", "getenvb"))

def _fscodec():
    encoding = sys.getfilesystemencoding()
    errors = sys.getfilesystemencodeerrors()

    def fsencode(filename):
        """Encode filename (an os.PathLike, bytes, or str) to the filesystem
        encoding with 'surrogateescape' error handler, return bytes unchanged.
        On Windows, use 'strict' error handler if the file system encoding is
        'mbcs' (which is the default encoding).
        """
        filename = fspath(filename)  # Does type-checking of `filename`.
        if isinstance(filename, str):
            return filename.encode(encoding, errors)
        else:
            return filename

    def fsdecode(filename):
        """Decode filename (an os.PathLike, bytes, or str) from the filesystem
        encoding with 'surrogateescape' error handler, return str unchanged. On
        Windows, use 'strict' error handler if the file system encoding is
        'mbcs' (which is the default encoding).
        """
        filename = fspath(filename)  # Does type-checking of `filename`.
        if isinstance(filename, bytes):
            return filename.decode(encoding, errors)
        else:
            return filename

    return fsencode, fsdecode

fsencode, fsdecode = _fscodec()
del _fscodec

# Supply spawn*() (probably only for Unix)
if _exists("fork") and not _exists("spawnv") and _exists("execv"):

    P_WAIT = 0
    P_NOWAIT = P_NOWAITO = 1

    __all__.extend(["P_WAIT", "P_NOWAIT", "P_NOWAITO"])

    # XXX Should we support P_DETACH?  I suppose it could fork()**2
    # and close the std I/O streams.  Also, P_OVERLAY is the same
    # as execv*()?

    def _spawnvef(mode, file, args, env, func):
        # Internal helper; func is the exec*() function to use
        if not isinstance(args, (tuple, list)):
            raise TypeError('argv must be a tuple or a list')
        if not args or not args[0]:
            raise ValueError('argv first element cannot be empty')
        pid = fork()
        if not pid:
            # Child
            try:
                if env is None:
                    func(file, args)
                else:
                    func(file, args, env)
            except:
                _exit(127)
        else:
            # Parent
            if mode == P_NOWAIT:
                return pid # Caller is responsible for waiting!
            while 1:
                wpid, sts = waitpid(pid, 0)
                if WIFSTOPPED(sts):
                    continue

                return waitstatus_to_exitcode(sts)

    def spawnv(mode, file, args):
        """spawnv(mode, file, args) -> integer

Execute file with arguments from args in a subprocess.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        return _spawnvef(mode, file, args, None, execv)

    def spawnve(mode, file, args, env):
        """spawnve(mode, file, args, env) -> integer

Execute file with arguments from args in a subprocess with the
specified environment.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        return _spawnvef(mode, file, args, env, execve)

    # Note: spawnvp[e] isn't currently supported on Windows

    def spawnvp(mode, file, args):
        """spawnvp(mode, file, args) -> integer

Execute file (which is looked for along $PATH) with arguments from
args in a subprocess.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        return _spawnvef(mode, file, args, None, execvp)

    def spawnvpe(mode, file, args, env):
        """spawnvpe(mode, file, args, env) -> integer

Execute file (which is looked for along $PATH) with arguments from
args in a subprocess with the supplied environment.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        return _spawnvef(mode, file, args, env, execvpe)


    __all__.extend(["spawnv", "spawnve", "spawnvp", "spawnvpe"])


if _exists("spawnv"):
    # These aren't supplied by the basic Windows code
    # but can be easily implemented in Python

    def spawnl(mode, file, *args):
        """spawnl(mode, file, *args) -> integer

Execute file with arguments from args in a subprocess.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        return spawnv(mode, file, args)

    def spawnle(mode, file, *args):
        """spawnle(mode, file, *args, env) -> integer

Execute file with arguments from args in a subprocess with the
supplied environment.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        env = args[-1]
        return spawnve(mode, file, args[:-1], env)


    __all__.extend(["spawnl", "spawnle"])


if _exists("spawnvp"):
    # At the moment, Windows doesn't implement spawnvp[e],
    # so it won't have spawnlp[e] either.
    def spawnlp(mode, file, *args):
        """spawnlp(mode, file, *args) -> integer

Execute file (which is looked for along $PATH) with arguments from
args in a subprocess with the supplied environment.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        return spawnvp(mode, file, args)

    def spawnlpe(mode, file, *args):
        """spawnlpe(mode, file, *args, env) -> integer

Execute file (which is looked for along $PATH) with arguments from
args in a subprocess with the supplied environment.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        env = args[-1]
        return spawnvpe(mode, file, args[:-1], env)


    __all__.extend(["spawnlp", "spawnlpe"])

# VxWorks has no user space shell provided. As a result, running
# command in a shell can't be supported.
if sys.platform != 'vxworks':
    # Supply os.popen()
    def popen(cmd, mode="r", buffering=-1):
        if not isinstance(cmd, str):
            raise TypeError("invalid cmd type (%s, expected string)" % type(cmd))
        if mode not in ("r", "w"):
            raise ValueError("invalid mode %r" % mode)
        if buffering == 0 or buffering is None:
            raise ValueError("popen() does not support unbuffered streams")
        import subprocess
        if mode == "r":
            proc = subprocess.Popen(cmd,
                                    shell=True, text=True,
                                    stdout=subprocess.PIPE,
                                    bufsize=buffering)
            return _wrap_close(proc.stdout, proc)
        else:
            proc = subprocess.Popen(cmd,
                                    shell=True, text=True,
                                    stdin=subprocess.PIPE,
                                    bufsize=buffering)
            return _wrap_close(proc.stdin, proc)

    # Helper for popen() -- a proxy for a file whose close waits for the process
    class _wrap_close:
        def __init__(self, stream, proc):
            self._stream = stream
            self._proc = proc
        def close(self):
            self._stream.close()
            returncode = self._proc.wait()
            if returncode == 0:
                return None
            if name == 'nt':
                return returncode
            else:
                return returncode << 8  # Shift left to match old behavior
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def __getattr__(self, name):
            return getattr(self._stream, name)
        def __iter__(self):
            return iter(self._stream)

    __all__.append("popen")

# Supply os.fdopen()
def fdopen(fd, mode="r", buffering=-1, encoding=None, *args, **kwargs):
    if not isinstance(fd, int):
        raise TypeError("invalid fd type (%s, expected integer)" % type(fd))
    import io
    if "b" not in mode:
        encoding = io.text_encoding(encoding)
    return io.open(fd, mode, buffering, encoding, *args, **kwargs)


# For testing purposes, make sure the function is available when the C
# implementation exists.
def _fspath(path):
    """Return the path representation of a path-like object.

    If str or bytes is passed in, it is returned unchanged. Otherwise the
    os.PathLike interface is used to get the path representation. If the
    path representation is not str or bytes, TypeError is raised. If the
    provided path is not str, bytes, or os.PathLike, TypeError is raised.
    """
    if isinstance(path, (str, bytes)):
        return path

    # Work from the object's type to match method resolution of other magic
    # methods.
    path_type = type(path)
    try:
        path_repr = path_type.__fspath__(path)
    except AttributeError:
        if hasattr(path_type, '__fspath__'):
            raise
        else:
            raise TypeError("expected str, bytes or os.PathLike object, "
                            "not " + path_type.__name__)
    except TypeError:
        if path_type.__fspath__ is None:
            raise TypeError("expected str, bytes or os.PathLike object, "
                            "not " + path_type.__name__) from None
        else:
            raise
    if isinstance(path_repr, (str, bytes)):
        return path_repr
    else:
        raise TypeError("expected {}.__fspath__() to return str or bytes, "
                        "not {}".format(path_type.__name__,
                                        type(path_repr).__name__))

# If there is no C implementation, make the pure Python version the
# implementation as transparently as possible.
if not _exists('fspath'):
    fspath = _fspath
    fspath.__name__ = "fspath"


class PathLike(abc.ABC):

    """Abstract base class for implementing the file system path protocol."""

    __slots__ = ()

    @abc.abstractmethod
    def __fspath__(self):
        """Return the file system path representation of the object."""
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is PathLike:
            return _check_methods(subclass, '__fspath__')
        return NotImplemented

    __class_getitem__ = classmethod(GenericAlias)


if name == 'nt':
    class _AddedDllDirectory:
        def __init__(self, path, cookie, remove_dll_directory):
            self.path = path
            self._cookie = cookie
            self._remove_dll_directory = remove_dll_directory
        def close(self):
            self._remove_dll_directory(self._cookie)
            self.path = None
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def __repr__(self):
            if self.path:
                return "<AddedDllDirectory({!r})>".format(self.path)
            return "<AddedDllDirectory()>"

    def add_dll_directory(path):
        """Add a path to the DLL search path.

        This search path is used when resolving dependencies for imported
        extension modules (the module itself is resolved through sys.path),
        and also by ctypes.

        Remove the directory by calling close() on the returned object or
        using it in a with statement.
        """
        import nt
        cookie = nt._add_dll_directory(path)
        return _AddedDllDirectory(
            path,
            cookie,
            nt._remove_dll_directory
        )


if _exists('sched_getaffinity') and sys._get_cpu_count_config() < 0:
    def process_cpu_count():
        """
        Get the number of CPUs of the current process.

        Return the number of logical CPUs usable by the calling thread of the
        current process. Return None if indeterminable.
        """
        return len(sched_getaffinity(0))
else:
    # Just an alias to cpu_count() (same docstring)
    process_cpu_count = cpu_count

# Module 'ntpath' -- common operations on WinNT/Win95 pathnames
"""Common pathname manipulations, WindowsNT/95 version.

Instead of importing this module directly, import os and refer to this
module as os.path.
"""

# strings representing various path-related bits and pieces
# These are primarily for export; internally, they are hardcoded.
# Should be set before imports for resolving cyclic dependency.
curdir = '.'
pardir = '..'
extsep = '.'
sep = '\\'
pathsep = ';'
altsep = '/'
defpath = '.;C:\\bin'
devnull = 'nul'

import os
import sys
import genericpath
from genericpath import *

__all__ = ["normcase","isabs","join","splitdrive","splitroot","split","splitext",
           "basename","dirname","commonprefix","getsize","getmtime",
           "getatime","getctime", "islink","exists","lexists","isdir","isfile",
           "ismount","isreserved","expanduser","expandvars","normpath",
           "abspath","curdir","pardir","sep","pathsep","defpath","altsep",
           "extsep","devnull","realpath","supports_unicode_filenames","relpath",
           "samefile", "sameopenfile", "samestat", "commonpath", "isjunction",
           "isdevdrive", "ALLOW_MISSING"]

def _get_bothseps(path):
    if isinstance(path, bytes):
        return b'\\/'
    else:
        return '\\/'

# Normalize the case of a pathname and map slashes to backslashes.
# Other normalizations (such as optimizing '../' away) are not done
# (this is done by normpath).

try:
    from _winapi import (
        LCMapStringEx as _LCMapStringEx,
        LOCALE_NAME_INVARIANT as _LOCALE_NAME_INVARIANT,
        LCMAP_LOWERCASE as _LCMAP_LOWERCASE)

    def normcase(s):
        """Normalize case of pathname.

        Makes all characters lowercase and all slashes into backslashes.
        """
        s = os.fspath(s)
        if not s:
            return s
        if isinstance(s, bytes):
            encoding = sys.getfilesystemencoding()
            s = s.decode(encoding, 'surrogateescape').replace('/', '\\')
            s = _LCMapStringEx(_LOCALE_NAME_INVARIANT,
                               _LCMAP_LOWERCASE, s)
            return s.encode(encoding, 'surrogateescape')
        else:
            return _LCMapStringEx(_LOCALE_NAME_INVARIANT,
                                  _LCMAP_LOWERCASE,
                                  s.replace('/', '\\'))
except ImportError:
    def normcase(s):
        """Normalize case of pathname.

        Makes all characters lowercase and all slashes into backslashes.
        """
        s = os.fspath(s)
        if isinstance(s, bytes):
            return os.fsencode(os.fsdecode(s).replace('/', '\\').lower())
        return s.replace('/', '\\').lower()


def isabs(s):
    """Test whether a path is absolute"""
    s = os.fspath(s)
    if isinstance(s, bytes):
        sep = b'\\'
        altsep = b'/'
        colon_sep = b':\\'
        double_sep = b'\\\\'
    else:
        sep = '\\'
        altsep = '/'
        colon_sep = ':\\'
        double_sep = '\\\\'
    s = s[:3].replace(altsep, sep)
    # Absolute: UNC, device, and paths with a drive and root.
    return s.startswith(colon_sep, 1) or s.startswith(double_sep)


# Join two (or more) paths.
def join(path, *paths):
    path = os.fspath(path)
    if isinstance(path, bytes):
        sep = b'\\'
        seps = b'\\/'
        colon_seps = b':\\/'
    else:
        sep = '\\'
        seps = '\\/'
        colon_seps = ':\\/'
    try:
        result_drive, result_root, result_path = splitroot(path)
        for p in paths:
            p_drive, p_root, p_path = splitroot(p)
            if p_root:
                # Second path is absolute
                if p_drive or not result_drive:
                    result_drive = p_drive
                result_root = p_root
                result_path = p_path
                continue
            elif p_drive and p_drive != result_drive:
                if p_drive.lower() != result_drive.lower():
                    # Different drives => ignore the first path entirely
                    result_drive = p_drive
                    result_root = p_root
                    result_path = p_path
                    continue
                # Same drive in different case
                result_drive = p_drive
            # Second path is relative to the first
            if result_path and result_path[-1] not in seps:
                result_path = result_path + sep
            result_path = result_path + p_path
        ## add separator between UNC and non-absolute path
        if (result_path and not result_root and
            result_drive and result_drive[-1] not in colon_seps):
            return result_drive + sep + result_path
        return result_drive + result_root + result_path
    except (TypeError, AttributeError, BytesWarning):
        genericpath._check_arg_types('join', path, *paths)
        raise


# Split a path in a drive specification (a drive letter followed by a
# colon) and the path specification.
# It is always true that drivespec + pathspec == p
def splitdrive(p):
    """Split a pathname into drive/UNC sharepoint and relative path specifiers.
    Returns a 2-tuple (drive_or_unc, path); either part may be empty.

    If you assign
        result = splitdrive(p)
    It is always true that:
        result[0] + result[1] == p

    If the path contained a drive letter, drive_or_unc will contain
    everything up to and including the colon.  e.g. splitdrive("c:/dir")
    returns ("c:", "/dir")

    If the path contained a UNC path, the drive_or_unc will contain the
    host name and share up to but not including the fourth directory
    separator character.  e.g. splitdrive("//host/computer/dir") returns
    ("//host/computer", "/dir")

    Paths cannot contain both a drive letter and a UNC path.

    """
    drive, root, tail = splitroot(p)
    return drive, root + tail


try:
    from nt import _path_splitroot_ex as splitroot
except ImportError:
    def splitroot(p):
        """Split a pathname into drive, root and tail.

        The tail contains anything after the root."""
        p = os.fspath(p)
        if isinstance(p, bytes):
            sep = b'\\'
            altsep = b'/'
            colon = b':'
            unc_prefix = b'\\\\?\\UNC\\'
            empty = b''
        else:
            sep = '\\'
            altsep = '/'
            colon = ':'
            unc_prefix = '\\\\?\\UNC\\'
            empty = ''
        normp = p.replace(altsep, sep)
        if normp[:1] == sep:
            if normp[1:2] == sep:
                # UNC drives, e.g. \\server\share or \\?\UNC\server\share
                # Device drives, e.g. \\.\device or \\?\device
                start = 8 if normp[:8].upper() == unc_prefix else 2
                index = normp.find(sep, start)
                if index == -1:
                    return p, empty, empty
                index2 = normp.find(sep, index + 1)
                if index2 == -1:
                    return p, empty, empty
                return p[:index2], p[index2:index2 + 1], p[index2 + 1:]
            else:
                # Relative path with root, e.g. \Windows
                return empty, p[:1], p[1:]
        elif normp[1:2] == colon:
            if normp[2:3] == sep:
                # Absolute drive-letter path, e.g. X:\Windows
                return p[:2], p[2:3], p[3:]
            else:
                # Relative path with drive, e.g. X:Windows
                return p[:2], empty, p[2:]
        else:
            # Relative path, e.g. Windows
            return empty, empty, p


# Split a path in head (everything up to the last '/') and tail (the
# rest).  After the trailing '/' is stripped, the invariant
# join(head, tail) == p holds.
# The resulting head won't end in '/' unless it is the root.

def split(p):
    """Split a pathname.

    Return tuple (head, tail) where tail is everything after the final
    slash.  Either part may be empty."""
    p = os.fspath(p)
    seps = _get_bothseps(p)
    d, r, p = splitroot(p)
    # set i to index beyond p's last slash
    i = len(p)
    while i and p[i-1] not in seps:
        i -= 1
    head, tail = p[:i], p[i:]  # now tail has no slashes
    return d + r + head.rstrip(seps), tail


# Split a path in root and extension.
# The extension is everything starting at the last dot in the last
# pathname component; the root is everything before that.
# It is always true that root + ext == p.

def splitext(p):
    p = os.fspath(p)
    if isinstance(p, bytes):
        return genericpath._splitext(p, b'\\', b'/', b'.')
    else:
        return genericpath._splitext(p, '\\', '/', '.')
splitext.__doc__ = genericpath._splitext.__doc__


# Return the tail (basename) part of a path.

def basename(p):
    """Returns the final component of a pathname"""
    return split(p)[1]


# Return the head (dirname) part of a path.

def dirname(p):
    """Returns the directory component of a pathname"""
    return split(p)[0]


# Is a path a mount point?
# Any drive letter root (eg c:\)
# Any share UNC (eg \\server\share)
# Any volume mounted on a filesystem folder
#
# No one method detects all three situations. Historically we've lexically
# detected drive letter roots and share UNCs. The canonical approach to
# detecting mounted volumes (querying the reparse tag) fails for the most
# common case: drive letter roots. The alternative which uses GetVolumePathName
# fails if the drive letter is the result of a SUBST.
try:
    from nt import _getvolumepathname
except ImportError:
    _getvolumepathname = None
def ismount(path):
    """Test whether a path is a mount point (a drive root, the root of a
    share, or a mounted volume)"""
    path = os.fspath(path)
    seps = _get_bothseps(path)
    path = abspath(path)
    drive, root, rest = splitroot(path)
    if drive and drive[0] in seps:
        return not rest
    if root and not rest:
        return True

    if _getvolumepathname:
        x = path.rstrip(seps)
        y =_getvolumepathname(path).rstrip(seps)
        return x.casefold() == y.casefold()
    else:
        return False


_reserved_chars = frozenset(
    {chr(i) for i in range(32)} |
    {'"', '*', ':', '<', '>', '?', '|', '/', '\\'}
)

_reserved_names = frozenset(
    {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'} |
    {f'COM{c}' for c in '123456789\xb9\xb2\xb3'} |
    {f'LPT{c}' for c in '123456789\xb9\xb2\xb3'}
)

def isreserved(path):
    """Return true if the pathname is reserved by the system."""
    # Refer to "Naming Files, Paths, and Namespaces":
    # https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file
    path = os.fsdecode(splitroot(path)[2]).replace(altsep, sep)
    return any(_isreservedname(name) for name in reversed(path.split(sep)))

def _isreservedname(name):
    """Return true if the filename is reserved by the system."""
    # Trailing dots and spaces are reserved.
    if name[-1:] in ('.', ' '):
        return name not in ('.', '..')
    # Wildcards, separators, colon, and pipe (*?"<>/\:|) are reserved.
    # ASCII control characters (0-31) are reserved.
    # Colon is reserved for file streams (e.g. "name:stream[:type]").
    if _reserved_chars.intersection(name):
        return True
    # DOS device names are reserved (e.g. "nul" or "nul .txt"). The rules
    # are complex and vary across Windows versions. On the side of
    # caution, return True for names that may not be reserved.
    return name.partition('.')[0].rstrip(' ').upper() in _reserved_names


# Expand paths beginning with '~' or '~user'.
# '~' means $HOME; '~user' means that user's home directory.
# If the path doesn't begin with '~', or if the user or $HOME is unknown,
# the path is returned unchanged (leaving error reporting to whatever
# function is called with the expanded path as argument).
# See also module 'glob' for expansion of *, ? and [...] in pathnames.
# (A function should also be defined to do full *sh-style environment
# variable expansion.)

def expanduser(path):
    """Expand ~ and ~user constructs.

    If user or home directory is unknown, do nothing."""
    path = os.fspath(path)
    if isinstance(path, bytes):
        seps = b'\\/'
        tilde = b'~'
    else:
        seps = '\\/'
        tilde = '~'
    if not path.startswith(tilde):
        return path
    i, n = 1, len(path)
    while i < n and path[i] not in seps:
        i += 1

    if 'USERPROFILE' in os.environ:
        userhome = os.environ['USERPROFILE']
    elif 'HOMEPATH' not in os.environ:
        return path
    else:
        drive = os.environ.get('HOMEDRIVE', '')
        userhome = join(drive, os.environ['HOMEPATH'])

    if i != 1: #~user
        target_user = path[1:i]
        if isinstance(target_user, bytes):
            target_user = os.fsdecode(target_user)
        current_user = os.environ.get('USERNAME')

        if target_user != current_user:
            # Try to guess user home directory.  By default all user
            # profile directories are located in the same place and are
            # named by corresponding usernames.  If userhome isn't a
            # normal profile directory, this guess is likely wrong,
            # so we bail out.
            if current_user != basename(userhome):
                return path
            userhome = join(dirname(userhome), target_user)

    if isinstance(path, bytes):
        userhome = os.fsencode(userhome)

    return userhome + path[i:]


# Expand paths containing shell variable substitutions.
# The following rules apply:
#       - no expansion within single quotes
#       - '$$' is translated into '$'
#       - '%%' is translated into '%' if '%%' are not seen in %var1%%var2%
#       - ${varname} is accepted.
#       - $varname is accepted.
#       - %varname% is accepted.
#       - varnames can be made out of letters, digits and the characters '_-'
#         (though is not verified in the ${varname} and %varname% cases)
# XXX With COMMAND.COM you can use any characters in a variable name,
# XXX except '^|<>='.

_varpattern = r"'[^']*'?|%(%|[^%]*%?)|\$(\$|[-\w]+|\{[^}]*\}?)"
_varsub = None
_varsubb = None

def expandvars(path):
    """Expand shell variables of the forms $var, ${var} and %var%.

    Unknown variables are left unchanged."""
    path = os.fspath(path)
    global _varsub, _varsubb
    if isinstance(path, bytes):
        if b'$' not in path and b'%' not in path:
            return path
        if not _varsubb:
            import re
            _varsubb = re.compile(_varpattern.encode(), re.ASCII).sub
        sub = _varsubb
        percent = b'%'
        brace = b'{'
        rbrace = b'}'
        dollar = b'$'
        environ = getattr(os, 'environb', None)
    else:
        if '$' not in path and '%' not in path:
            return path
        if not _varsub:
            import re
            _varsub = re.compile(_varpattern, re.ASCII).sub
        sub = _varsub
        percent = '%'
        brace = '{'
        rbrace = '}'
        dollar = '$'
        environ = os.environ

    def repl(m):
        lastindex = m.lastindex
        if lastindex is None:
            return m[0]
        name = m[lastindex]
        if lastindex == 1:
            if name == percent:
                return name
            if not name.endswith(percent):
                return m[0]
            name = name[:-1]
        else:
            if name == dollar:
                return name
            if name.startswith(brace):
                if not name.endswith(rbrace):
                    return m[0]
                name = name[1:-1]

        try:
            if environ is None:
                return os.fsencode(os.environ[os.fsdecode(name)])
            else:
                return environ[name]
        except KeyError:
            return m[0]

    return sub(repl, path)


# Normalize a path, e.g. A//B, A/./B and A/foo/../B all become A\B.
# Previously, this function also truncated pathnames to 8+3 format,
# but as this module is called "ntpath", that's obviously wrong!
try:
    from nt import _path_normpath as normpath

except ImportError:
    def normpath(path):
        """Normalize path, eliminating double slashes, etc."""
        path = os.fspath(path)
        if isinstance(path, bytes):
            sep = b'\\'
            altsep = b'/'
            curdir = b'.'
            pardir = b'..'
        else:
            sep = '\\'
            altsep = '/'
            curdir = '.'
            pardir = '..'
        path = path.replace(altsep, sep)
        drive, root, path = splitroot(path)
        prefix = drive + root
        comps = path.split(sep)
        i = 0
        while i < len(comps):
            if not comps[i] or comps[i] == curdir:
                del comps[i]
            elif comps[i] == pardir:
                if i > 0 and comps[i-1] != pardir:
                    del comps[i-1:i+1]
                    i -= 1
                elif i == 0 and root:
                    del comps[i]
                else:
                    i += 1
            else:
                i += 1
        # If the path is now empty, substitute '.'
        if not prefix and not comps:
            comps.append(curdir)
        return prefix + sep.join(comps)


# Return an absolute path.
try:
    from nt import _getfullpathname

except ImportError: # not running on Windows - mock up something sensible
    def abspath(path):
        """Return the absolute version of a path."""
        path = os.fspath(path)
        if not isabs(path):
            if isinstance(path, bytes):
                cwd = os.getcwdb()
            else:
                cwd = os.getcwd()
            path = join(cwd, path)
        return normpath(path)

else:  # use native Windows method on Windows
    def abspath(path):
        """Return the absolute version of a path."""
        try:
            return _getfullpathname(normpath(path))
        except (OSError, ValueError):
            # See gh-75230, handle outside for cleaner traceback
            pass
        path = os.fspath(path)
        if not isabs(path):
            if isinstance(path, bytes):
                sep = b'\\'
                getcwd = os.getcwdb
            else:
                sep = '\\'
                getcwd = os.getcwd
            drive, root, path = splitroot(path)
            # Either drive or root can be nonempty, but not both.
            if drive or root:
                try:
                    path = join(_getfullpathname(drive + root), path)
                except (OSError, ValueError):
                    # Drive "\0:" cannot exist; use the root directory.
                    path = drive + sep + path
            else:
                path = join(getcwd(), path)
        return normpath(path)

try:
    from nt import _findfirstfile, _getfinalpathname, readlink as _nt_readlink
except ImportError:
    # realpath is a no-op on systems without _getfinalpathname support.
    def realpath(path, *, strict=False):
        return abspath(path)
else:
    def _readlink_deep(path, ignored_error=OSError):
        # These error codes indicate that we should stop reading links and
        # return the path we currently have.
        # 1: ERROR_INVALID_FUNCTION
        # 2: ERROR_FILE_NOT_FOUND
        # 3: ERROR_DIRECTORY_NOT_FOUND
        # 5: ERROR_ACCESS_DENIED
        # 21: ERROR_NOT_READY (implies drive with no media)
        # 32: ERROR_SHARING_VIOLATION (probably an NTFS paging file)
        # 50: ERROR_NOT_SUPPORTED (implies no support for reparse points)
        # 67: ERROR_BAD_NET_NAME (implies remote server unavailable)
        # 87: ERROR_INVALID_PARAMETER
        # 4390: ERROR_NOT_A_REPARSE_POINT
        # 4392: ERROR_INVALID_REPARSE_DATA
        # 4393: ERROR_REPARSE_TAG_INVALID
        allowed_winerror = 1, 2, 3, 5, 21, 32, 50, 67, 87, 4390, 4392, 4393

        seen = set()
        while normcase(path) not in seen:
            seen.add(normcase(path))
            try:
                old_path = path
                path = _nt_readlink(path)
                # Links may be relative, so resolve them against their
                # own location
                if not isabs(path):
                    # If it's something other than a symlink, we don't know
                    # what it's actually going to be resolved against, so
                    # just return the old path.
                    if not islink(old_path):
                        path = old_path
                        break
                    path = normpath(join(dirname(old_path), path))
            except ignored_error as ex:
                if ex.winerror in allowed_winerror:
                    break
                raise
            except ValueError:
                # Stop on reparse points that are not symlinks
                break
        return path

    def _getfinalpathname_nonstrict(path, ignored_error=OSError):
        # These error codes indicate that we should stop resolving the path
        # and return the value we currently have.
        # 1: ERROR_INVALID_FUNCTION
        # 2: ERROR_FILE_NOT_FOUND
        # 3: ERROR_DIRECTORY_NOT_FOUND
        # 5: ERROR_ACCESS_DENIED
        # 21: ERROR_NOT_READY (implies drive with no media)
        # 32: ERROR_SHARING_VIOLATION (probably an NTFS paging file)
        # 50: ERROR_NOT_SUPPORTED
        # 53: ERROR_BAD_NETPATH
        # 65: ERROR_NETWORK_ACCESS_DENIED
        # 67: ERROR_BAD_NET_NAME (implies remote server unavailable)
        # 87: ERROR_INVALID_PARAMETER
        # 123: ERROR_INVALID_NAME
        # 161: ERROR_BAD_PATHNAME
        # 1005: ERROR_UNRECOGNIZED_VOLUME
        # 1920: ERROR_CANT_ACCESS_FILE
        # 1921: ERROR_CANT_RESOLVE_FILENAME (implies unfollowable symlink)
        allowed_winerror = 1, 2, 3, 5, 21, 32, 50, 53, 65, 67, 87, 123, 161, 1005, 1920, 1921

        # Non-strict algorithm is to find as much of the target directory
        # as we can and join the rest.
        tail = path[:0]
        while path:
            try:
                path = _getfinalpathname(path)
                return join(path, tail) if tail else path
            except ignored_error as ex:
                if ex.winerror not in allowed_winerror:
                    raise
                try:
                    # The OS could not resolve this path fully, so we attempt
                    # to follow the link ourselves. If we succeed, join the tail
                    # and return.
                    new_path = _readlink_deep(path,
                                              ignored_error=ignored_error)
                    if new_path != path:
                        return join(new_path, tail) if tail else new_path
                except ignored_error:
                    # If we fail to readlink(), let's keep traversing
                    pass
                # If we get these errors, try to get the real name of the file without accessing it.
                if ex.winerror in (1, 5, 32, 50, 87, 1920, 1921):
                    try:
                        name = _findfirstfile(path)
                        path, _ = split(path)
                    except ignored_error:
                        path, name = split(path)
                else:
                    path, name = split(path)
                if path and not name:
                    return path + tail
                tail = join(name, tail) if tail else name
        return tail

    def realpath(path, *, strict=False):
        path = normpath(path)
        if isinstance(path, bytes):
            prefix = b'\\\\?\\'
            unc_prefix = b'\\\\?\\UNC\\'
            new_unc_prefix = b'\\\\'
            cwd = os.getcwdb()
            # bpo-38081: Special case for realpath(b'nul')
            devnull = b'nul'
            if normcase(path) == devnull:
                return b'\\\\.\\NUL'
        else:
            prefix = '\\\\?\\'
            unc_prefix = '\\\\?\\UNC\\'
            new_unc_prefix = '\\\\'
            cwd = os.getcwd()
            # bpo-38081: Special case for realpath('nul')
            devnull = 'nul'
            if normcase(path) == devnull:
                return '\\\\.\\NUL'
        had_prefix = path.startswith(prefix)

        if strict is ALLOW_MISSING:
            ignored_error = FileNotFoundError
            strict = True
        elif strict:
            ignored_error = ()
        else:
            ignored_error = OSError

        if not had_prefix and not isabs(path):
            path = join(cwd, path)
        try:
            path = _getfinalpathname(path)
            initial_winerror = 0
        except ValueError as ex:
            # gh-106242: Raised for embedded null characters
            # In strict modes, we convert into an OSError.
            # Non-strict mode returns the path as-is, since we've already
            # made it absolute.
            if strict:
                raise OSError(str(ex)) from None
            path = normpath(path)
        except ignored_error as ex:
            initial_winerror = ex.winerror
            path = _getfinalpathname_nonstrict(path,
                                               ignored_error=ignored_error)
        # The path returned by _getfinalpathname will always start with \\?\ -
        # strip off that prefix unless it was already provided on the original
        # path.
        if not had_prefix and path.startswith(prefix):
            # For UNC paths, the prefix will actually be \\?\UNC\
            # Handle that case as well.
            if path.startswith(unc_prefix):
                spath = new_unc_prefix + path[len(unc_prefix):]
            else:
                spath = path[len(prefix):]
            # Ensure that the non-prefixed path resolves to the same path
            try:
                if _getfinalpathname(spath) == path:
                    path = spath
            except ValueError as ex:
                # Unexpected, as an invalid path should not have gained a prefix
                # at any point, but we ignore this error just in case.
                pass
            except OSError as ex:
                # If the path does not exist and originally did not exist, then
                # strip the prefix anyway.
                if ex.winerror == initial_winerror:
                    path = spath
        return path


# All supported version have Unicode filename support.
supports_unicode_filenames = True

def relpath(path, start=None):
    """Return a relative version of a path"""
    path = os.fspath(path)
    if not path:
        raise ValueError("no path specified")

    if isinstance(path, bytes):
        sep = b'\\'
        curdir = b'.'
        pardir = b'..'
    else:
        sep = '\\'
        curdir = '.'
        pardir = '..'

    if start is None:
        start = curdir
    else:
        start = os.fspath(start)

    try:
        start_abs = abspath(start)
        path_abs = abspath(path)
        start_drive, _, start_rest = splitroot(start_abs)
        path_drive, _, path_rest = splitroot(path_abs)
        if normcase(start_drive) != normcase(path_drive):
            raise ValueError("path is on mount %r, start on mount %r" % (
                path_drive, start_drive))

        start_list = start_rest.split(sep) if start_rest else []
        path_list = path_rest.split(sep) if path_rest else []
        # Work out how much of the filepath is shared by start and path.
        i = 0
        for e1, e2 in zip(start_list, path_list):
            if normcase(e1) != normcase(e2):
                break
            i += 1

        rel_list = [pardir] * (len(start_list)-i) + path_list[i:]
        if not rel_list:
            return curdir
        return sep.join(rel_list)
    except (TypeError, ValueError, AttributeError, BytesWarning, DeprecationWarning):
        genericpath._check_arg_types('relpath', path, start)
        raise


# Return the longest common sub-path of the iterable of paths given as input.
# The function is case-insensitive and 'separator-insensitive', i.e. if the
# only difference between two paths is the use of '\' versus '/' as separator,
# they are deemed to be equal.
#
# However, the returned path will have the standard '\' separator (even if the
# given paths had the alternative '/' separator) and will have the case of the
# first path given in the iterable. Additionally, any trailing separator is
# stripped from the returned path.

def commonpath(paths):
    """Given an iterable of path names, returns the longest common sub-path."""
    paths = tuple(map(os.fspath, paths))
    if not paths:
        raise ValueError('commonpath() arg is an empty iterable')

    if isinstance(paths[0], bytes):
        sep = b'\\'
        altsep = b'/'
        curdir = b'.'
    else:
        sep = '\\'
        altsep = '/'
        curdir = '.'

    try:
        drivesplits = [splitroot(p.replace(altsep, sep).lower()) for p in paths]
        split_paths = [p.split(sep) for d, r, p in drivesplits]

        # Check that all drive letters or UNC paths match. The check is made only
        # now otherwise type errors for mixing strings and bytes would not be
        # caught.
        if len({d for d, r, p in drivesplits}) != 1:
            raise ValueError("Paths don't have the same drive")

        drive, root, path = splitroot(paths[0].replace(altsep, sep))
        if len({r for d, r, p in drivesplits}) != 1:
            if drive:
                raise ValueError("Can't mix absolute and relative paths")
            else:
                raise ValueError("Can't mix rooted and not-rooted paths")

        common = path.split(sep)
        common = [c for c in common if c and c != curdir]

        split_paths = [[c for c in s if c and c != curdir] for s in split_paths]
        s1 = min(split_paths)
        s2 = max(split_paths)
        for i, c in enumerate(s1):
            if c != s2[i]:
                common = common[:i]
                break
        else:
            common = common[:len(s1)]

        return drive + root + sep.join(common)
    except (TypeError, AttributeError):
        genericpath._check_arg_types('commonpath', *paths)
        raise


try:
    # The isdir(), isfile(), islink(), exists() and lexists() implementations
    # in genericpath use os.stat(). This is overkill on Windows. Use simpler
    # builtin functions if they are available.
    from nt import _path_isdir as isdir
    from nt import _path_isfile as isfile
    from nt import _path_islink as islink
    from nt import _path_isjunction as isjunction
    from nt import _path_exists as exists
    from nt import _path_lexists as lexists
except ImportError:
    # Use genericpath.* as imported above
    pass


try:
    from nt import _path_isdevdrive
    def isdevdrive(path):
        """Determines whether the specified path is on a Windows Dev Drive."""
        try:
            return _path_isdevdrive(abspath(path))
        except OSError:
            return False
except ImportError:
    # Use genericpath.isdevdrive as imported above
    pass

"""Interfaces for launching and remotely controlling web browsers."""
# Maintained by Georg Brandl.

import os
import shlex
import shutil
import sys
import subprocess
import threading

__all__ = ["Error", "open", "open_new", "open_new_tab", "get", "register"]


class Error(Exception):
    pass


_lock = threading.RLock()
_browsers = {}                  # Dictionary of available browser controllers
_tryorder = None                # Preference order of available browsers
_os_preferred_browser = None    # The preferred browser


def register(name, klass, instance=None, *, preferred=False):
    """Register a browser connector."""
    with _lock:
        if _tryorder is None:
            register_standard_browsers()
        _browsers[name.lower()] = [klass, instance]

        # Preferred browsers go to the front of the list.
        # Need to match to the default browser returned by xdg-settings, which
        # may be of the form e.g. "firefox.desktop".
        if preferred or (_os_preferred_browser and f'{name}.desktop' == _os_preferred_browser):
            _tryorder.insert(0, name)
        else:
            _tryorder.append(name)


def get(using=None):
    """Return a browser launcher instance appropriate for the environment."""
    if _tryorder is None:
        with _lock:
            if _tryorder is None:
                register_standard_browsers()
    if using is not None:
        alternatives = [using]
    else:
        alternatives = _tryorder
    for browser in alternatives:
        if '%s' in browser:
            # User gave us a command line, split it into name and args
            browser = shlex.split(browser)
            if browser[-1] == '&':
                return BackgroundBrowser(browser[:-1])
            else:
                return GenericBrowser(browser)
        else:
            # User gave us a browser name or path.
            try:
                command = _browsers[browser.lower()]
            except KeyError:
                command = _synthesize(browser)
            if command[1] is not None:
                return command[1]
            elif command[0] is not None:
                return command[0]()
    raise Error("could not locate runnable browser")


# Please note: the following definition hides a builtin function.
# It is recommended one does "import webbrowser" and uses webbrowser.open(url)
# instead of "from webbrowser import *".

def open(url, new=0, autoraise=True):
    """Display url using the default browser.

    If possible, open url in a location determined by new.
    - 0: the same browser window (the default).
    - 1: a new browser window.
    - 2: a new browser page ("tab").
    If possible, autoraise raises the window (the default) or not.

    If opening the browser succeeds, return True.
    If there is a problem, return False.
    """
    if _tryorder is None:
        with _lock:
            if _tryorder is None:
                register_standard_browsers()
    for name in _tryorder:
        browser = get(name)
        if browser.open(url, new, autoraise):
            return True
    return False


def open_new(url):
    """Open url in a new window of the default browser.

    If not possible, then open url in the only browser window.
    """
    return open(url, 1)


def open_new_tab(url):
    """Open url in a new page ("tab") of the default browser.

    If not possible, then the behavior becomes equivalent to open_new().
    """
    return open(url, 2)


def _synthesize(browser, *, preferred=False):
    """Attempt to synthesize a controller based on existing controllers.

    This is useful to create a controller when a user specifies a path to
    an entry in the BROWSER environment variable -- we can copy a general
    controller to operate using a specific installation of the desired
    browser in this way.

    If we can't create a controller in this way, or if there is no
    executable for the requested browser, return [None, None].

    """
    cmd = browser.split()[0]
    if not shutil.which(cmd):
        return [None, None]
    name = os.path.basename(cmd)
    try:
        command = _browsers[name.lower()]
    except KeyError:
        return [None, None]
    # now attempt to clone to fit the new name:
    controller = command[1]
    if controller and name.lower() == controller.basename:
        import copy
        controller = copy.copy(controller)
        controller.name = browser
        controller.basename = os.path.basename(browser)
        register(browser, None, instance=controller, preferred=preferred)
        return [None, controller]
    return [None, None]


# General parent classes

class BaseBrowser:
    """Parent class for all browsers. Do not use directly."""

    args = ['%s']

    def __init__(self, name=""):
        self.name = name
        self.basename = name

    def open(self, url, new=0, autoraise=True):
        raise NotImplementedError

    def open_new(self, url):
        return self.open(url, 1)

    def open_new_tab(self, url):
        return self.open(url, 2)

    @staticmethod
    def _check_url(url):
        """Ensures that the URL is safe to pass to subprocesses as a parameter"""
        if url and url.lstrip().startswith("-"):
            raise ValueError(f"Invalid URL (leading dash disallowed): {url!r}")


class GenericBrowser(BaseBrowser):
    """Class for all browsers started with a command
       and without remote functionality."""

    def __init__(self, name):
        if isinstance(name, str):
            self.name = name
            self.args = ["%s"]
        else:
            # name should be a list with arguments
            self.name = name[0]
            self.args = name[1:]
        self.basename = os.path.basename(self.name)

    def open(self, url, new=0, autoraise=True):
        sys.audit("webbrowser.open", url)
        self._check_url(url)
        cmdline = [self.name] + [arg.replace("%s", url)
                                 for arg in self.args]
        try:
            if sys.platform[:3] == 'win':
                p = subprocess.Popen(cmdline)
            else:
                p = subprocess.Popen(cmdline, close_fds=True)
            return not p.wait()
        except OSError:
            return False


class BackgroundBrowser(GenericBrowser):
    """Class for all browsers which are to be started in the
       background."""

    def open(self, url, new=0, autoraise=True):
        cmdline = [self.name] + [arg.replace("%s", url)
                                 for arg in self.args]
        sys.audit("webbrowser.open", url)
        self._check_url(url)
        try:
            if sys.platform[:3] == 'win':
                p = subprocess.Popen(cmdline)
            else:
                p = subprocess.Popen(cmdline, close_fds=True,
                                     start_new_session=True)
            return p.poll() is None
        except OSError:
            return False


class UnixBrowser(BaseBrowser):
    """Parent class for all Unix browsers with remote functionality."""

    raise_opts = None
    background = False
    redirect_stdout = True
    # In remote_args, %s will be replaced with the requested URL.  %action will
    # be replaced depending on the value of 'new' passed to open.
    # remote_action is used for new=0 (open).  If newwin is not None, it is
    # used for new=1 (open_new).  If newtab is not None, it is used for
    # new=3 (open_new_tab).  After both substitutions are made, any empty
    # strings in the transformed remote_args list will be removed.
    remote_args = ['%action', '%s']
    remote_action = None
    remote_action_newwin = None
    remote_action_newtab = None

    def _invoke(self, args, remote, autoraise, url=None):
        raise_opt = []
        if remote and self.raise_opts:
            # use autoraise argument only for remote invocation
            autoraise = int(autoraise)
            opt = self.raise_opts[autoraise]
            if opt:
                raise_opt = [opt]

        cmdline = [self.name] + raise_opt + args

        if remote or self.background:
            inout = subprocess.DEVNULL
        else:
            # for TTY browsers, we need stdin/out
            inout = None
        p = subprocess.Popen(cmdline, close_fds=True, stdin=inout,
                             stdout=(self.redirect_stdout and inout or None),
                             stderr=inout, start_new_session=True)
        if remote:
            # wait at most five seconds. If the subprocess is not finished, the
            # remote invocation has (hopefully) started a new instance.
            try:
                rc = p.wait(5)
                # if remote call failed, open() will try direct invocation
                return not rc
            except subprocess.TimeoutExpired:
                return True
        elif self.background:
            if p.poll() is None:
                return True
            else:
                return False
        else:
            return not p.wait()

    def open(self, url, new=0, autoraise=True):
        sys.audit("webbrowser.open", url)
        if new == 0:
            action = self.remote_action
        elif new == 1:
            action = self.remote_action_newwin
        elif new == 2:
            if self.remote_action_newtab is None:
                action = self.remote_action_newwin
            else:
                action = self.remote_action_newtab
        else:
            raise Error("Bad 'new' parameter to open(); "
                        f"expected 0, 1, or 2, got {new}")

        self._check_url(url.replace("%action", action))

        args = [arg.replace("%action", action).replace("%s", url)
                for arg in self.remote_args]
        args = [arg for arg in args if arg]
        success = self._invoke(args, True, autoraise, url)
        if not success:
            # remote invocation failed, try straight way
            args = [arg.replace("%s", url) for arg in self.args]
            return self._invoke(args, False, False)
        else:
            return True


class Mozilla(UnixBrowser):
    """Launcher class for Mozilla browsers."""

    remote_args = ['%action', '%s']
    remote_action = ""
    remote_action_newwin = "-new-window"
    remote_action_newtab = "-new-tab"
    background = True


class Epiphany(UnixBrowser):
    """Launcher class for Epiphany browser."""

    raise_opts = ["-noraise", ""]
    remote_args = ['%action', '%s']
    remote_action = "-n"
    remote_action_newwin = "-w"
    background = True


class Chrome(UnixBrowser):
    """Launcher class for Google Chrome browser."""

    remote_args = ['%action', '%s']
    remote_action = ""
    remote_action_newwin = "--new-window"
    remote_action_newtab = ""
    background = True


Chromium = Chrome


class Opera(UnixBrowser):
    """Launcher class for Opera browser."""

    remote_args = ['%action', '%s']
    remote_action = ""
    remote_action_newwin = "--new-window"
    remote_action_newtab = ""
    background = True


class Elinks(UnixBrowser):
    """Launcher class for Elinks browsers."""

    remote_args = ['-remote', 'openURL(%s%action)']
    remote_action = ""
    remote_action_newwin = ",new-window"
    remote_action_newtab = ",new-tab"
    background = False

    # elinks doesn't like its stdout to be redirected -
    # it uses redirected stdout as a signal to do -dump
    redirect_stdout = False


class Konqueror(BaseBrowser):
    """Controller for the KDE File Manager (kfm, or Konqueror).

    See the output of ``kfmclient --commands``
    for more information on the Konqueror remote-control interface.
    """

    def open(self, url, new=0, autoraise=True):
        sys.audit("webbrowser.open", url)
        self._check_url(url)
        # XXX Currently I know no way to prevent KFM from opening a new win.
        if new == 2:
            action = "newTab"
        else:
            action = "openURL"

        devnull = subprocess.DEVNULL

        try:
            p = subprocess.Popen(["kfmclient", action, url],
                                 close_fds=True, stdin=devnull,
                                 stdout=devnull, stderr=devnull)
        except OSError:
            # fall through to next variant
            pass
        else:
            p.wait()
            # kfmclient's return code unfortunately has no meaning as it seems
            return True

        try:
            p = subprocess.Popen(["konqueror", "--silent", url],
                                 close_fds=True, stdin=devnull,
                                 stdout=devnull, stderr=devnull,
                                 start_new_session=True)
        except OSError:
            # fall through to next variant
            pass
        else:
            if p.poll() is None:
                # Should be running now.
                return True

        try:
            p = subprocess.Popen(["kfm", "-d", url],
                                 close_fds=True, stdin=devnull,
                                 stdout=devnull, stderr=devnull,
                                 start_new_session=True)
        except OSError:
            return False
        else:
            return p.poll() is None


class Edge(UnixBrowser):
    """Launcher class for Microsoft Edge browser."""

    remote_args = ['%action', '%s']
    remote_action = ""
    remote_action_newwin = "--new-window"
    remote_action_newtab = ""
    background = True


#
# Platform support for Unix
#

# These are the right tests because all these Unix browsers require either
# a console terminal or an X display to run.

def register_X_browsers():

    # use xdg-open if around
    if shutil.which("xdg-open"):
        register("xdg-open", None, BackgroundBrowser("xdg-open"))

    # Opens an appropriate browser for the URL scheme according to
    # freedesktop.org settings (GNOME, KDE, XFCE, etc.)
    if shutil.which("gio"):
        register("gio", None, BackgroundBrowser(["gio", "open", "--", "%s"]))

    xdg_desktop = os.getenv("XDG_CURRENT_DESKTOP", "").split(":")

    # The default GNOME3 browser
    if (("GNOME" in xdg_desktop or
         "GNOME_DESKTOP_SESSION_ID" in os.environ) and
            shutil.which("gvfs-open")):
        register("gvfs-open", None, BackgroundBrowser("gvfs-open"))

    # The default KDE browser
    if (("KDE" in xdg_desktop or
         "KDE_FULL_SESSION" in os.environ) and
            shutil.which("kfmclient")):
        register("kfmclient", Konqueror, Konqueror("kfmclient"))

    # Common symbolic link for the default X11 browser
    if shutil.which("x-www-browser"):
        register("x-www-browser", None, BackgroundBrowser("x-www-browser"))

    # The Mozilla browsers
    for browser in ("firefox", "iceweasel", "seamonkey", "mozilla-firefox",
                    "mozilla"):
        if shutil.which(browser):
            register(browser, None, Mozilla(browser))

    # Konqueror/kfm, the KDE browser.
    if shutil.which("kfm"):
        register("kfm", Konqueror, Konqueror("kfm"))
    elif shutil.which("konqueror"):
        register("konqueror", Konqueror, Konqueror("konqueror"))

    # Gnome's Epiphany
    if shutil.which("epiphany"):
        register("epiphany", None, Epiphany("epiphany"))

    # Google Chrome/Chromium browsers
    for browser in ("google-chrome", "chrome", "chromium", "chromium-browser"):
        if shutil.which(browser):
            register(browser, None, Chrome(browser))

    # Opera, quite popular
    if shutil.which("opera"):
        register("opera", None, Opera("opera"))

    if shutil.which("microsoft-edge"):
        register("microsoft-edge", None, Edge("microsoft-edge"))


def register_standard_browsers():
    global _tryorder
    _tryorder = []

    if sys.platform == 'darwin':
        register("MacOSX", None, MacOSXOSAScript('default'))
        register("chrome", None, MacOSXOSAScript('google chrome'))
        register("firefox", None, MacOSXOSAScript('firefox'))
        register("safari", None, MacOSXOSAScript('safari'))
        # macOS can use below Unix support (but we prefer using the macOS
        # specific stuff)

    if sys.platform == "ios":
        register("iosbrowser", None, IOSBrowser(), preferred=True)

    if sys.platform == "serenityos":
        # SerenityOS webbrowser, simply called "Browser".
        register("Browser", None, BackgroundBrowser("Browser"))

    if sys.platform[:3] == "win":
        # First try to use the default Windows browser
        register("windows-default", WindowsDefault)

        # Detect some common Windows browsers, fallback to Microsoft Edge
        # location in 64-bit Windows
        edge64 = os.path.join(os.environ.get("PROGRAMFILES(x86)", "C:\\Program Files (x86)"),
                              "Microsoft\\Edge\\Application\\msedge.exe")
        # location in 32-bit Windows
        edge32 = os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"),
                              "Microsoft\\Edge\\Application\\msedge.exe")
        for browser in ("firefox", "seamonkey", "mozilla", "chrome",
                        "opera", edge64, edge32):
            if shutil.which(browser):
                register(browser, None, BackgroundBrowser(browser))
        if shutil.which("MicrosoftEdge.exe"):
            register("microsoft-edge", None, Edge("MicrosoftEdge.exe"))
    else:
        # Prefer X browsers if present
        #
        # NOTE: Do not check for X11 browser on macOS,
        # XQuartz installation sets a DISPLAY environment variable and will
        # autostart when someone tries to access the display. Mac users in
        # general don't need an X11 browser.
        if sys.platform != "darwin" and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            try:
                cmd = "xdg-settings get default-web-browser".split()
                raw_result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                result = raw_result.decode().strip()
            except (FileNotFoundError, subprocess.CalledProcessError,
                    PermissionError, NotADirectoryError):
                pass
            else:
                global _os_preferred_browser
                _os_preferred_browser = result

            register_X_browsers()

        # Also try console browsers
        if os.environ.get("TERM"):
            # Common symbolic link for the default text-based browser
            if shutil.which("www-browser"):
                register("www-browser", None, GenericBrowser("www-browser"))
            # The Links/elinks browsers <http://links.twibright.com/>
            if shutil.which("links"):
                register("links", None, GenericBrowser("links"))
            if shutil.which("elinks"):
                register("elinks", None, Elinks("elinks"))
            # The Lynx browser <https://lynx.invisible-island.net/>, <http://lynx.browser.org/>
            if shutil.which("lynx"):
                register("lynx", None, GenericBrowser("lynx"))
            # The w3m browser <http://w3m.sourceforge.net/>
            if shutil.which("w3m"):
                register("w3m", None, GenericBrowser("w3m"))

    # OK, now that we know what the default preference orders for each
    # platform are, allow user to override them with the BROWSER variable.
    if "BROWSER" in os.environ:
        userchoices = os.environ["BROWSER"].split(os.pathsep)
        userchoices.reverse()

        # Treat choices in same way as if passed into get() but do register
        # and prepend to _tryorder
        for cmdline in userchoices:
            if all(x not in cmdline for x in " \t"):
                # Assume this is the name of a registered command, use
                # that unless it is a GenericBrowser.
                try:
                    command = _browsers[cmdline.lower()]
                except KeyError:
                    pass

                else:
                    if not isinstance(command[1], GenericBrowser):
                        _tryorder.insert(0, cmdline.lower())
                        continue

            if cmdline != '':
                cmd = _synthesize(cmdline, preferred=True)
                if cmd[1] is None:
                    register(cmdline, None, GenericBrowser(cmdline), preferred=True)

    # what to do if _tryorder is now empty?


#
# Platform support for Windows
#

if sys.platform[:3] == "win":
    class WindowsDefault(BaseBrowser):
        def open(self, url, new=0, autoraise=True):
            sys.audit("webbrowser.open", url)
            self._check_url(url)
            try:
                os.startfile(url)
            except OSError:
                # [Error 22] No application is associated with the specified
                # file for this operation: '<URL>'
                return False
            else:
                return True

#
# Platform support for macOS
#

if sys.platform == 'darwin':
    class MacOSXOSAScript(BaseBrowser):
        def __init__(self, name='default'):
            super().__init__(name)

        def open(self, url, new=0, autoraise=True):
            sys.audit("webbrowser.open", url)
            self._check_url(url)
            url = url.replace('"', '%22')
            if self.name == 'default':
                proto, _sep, _rest = url.partition(":")
                if _sep and proto.lower() in {"http", "https"}:
                    # default web URL, don't need to lookup browser
                    script = f'open location "{url}"'
                else:
                    # if not a web URL, need to lookup default browser to ensure a browser is launched
                    # this should always work, but is overkill to lookup http handler
                    # before launching http
                    script = f"""
                        use framework "AppKit"
                        use AppleScript version "2.4"
                        use scripting additions

                        property NSWorkspace : a reference to current application's NSWorkspace
                        property NSURL : a reference to current application's NSURL

                        set http_url to NSURL's URLWithString:"https://python.org"
                        set browser_url to (NSWorkspace's sharedWorkspace)'s ¬
                            URLForApplicationToOpenURL:http_url
                        set app_path to browser_url's relativePath as text -- NSURL to absolute path '/Applications/Safari.app'

                        tell application app_path
                            activate
                            open location "{url}"
                        end tell
                    """
            else:
                script = f'''
                   tell application "{self.name}"
                       activate
                       open location "{url}"
                   end
                   '''

            osapipe = os.popen("/usr/bin/osascript", "w")
            if osapipe is None:
                return False

            osapipe.write(script)
            rc = osapipe.close()
            return not rc

#
# Platform support for iOS
#
if sys.platform == "ios":
    from _ios_support import objc
    if objc:
        # If objc exists, we know ctypes is also importable.
        from ctypes import c_void_p, c_char_p, c_ulong

    class IOSBrowser(BaseBrowser):
        def open(self, url, new=0, autoraise=True):
            sys.audit("webbrowser.open", url)
            self._check_url(url)
            # If ctypes isn't available, we can't open a browser
            if objc is None:
                return False

            # All the messages in this call return object references.
            objc.objc_msgSend.restype = c_void_p

            # This is the equivalent of:
            #    NSString url_string =
            #        [NSString stringWithCString:url.encode("utf-8")
            #                           encoding:NSUTF8StringEncoding];
            NSString = objc.objc_getClass(b"NSString")
            constructor = objc.sel_registerName(b"stringWithCString:encoding:")
            objc.objc_msgSend.argtypes = [c_void_p, c_void_p, c_char_p, c_ulong]
            url_string = objc.objc_msgSend(
                NSString,
                constructor,
                url.encode("utf-8"),
                4,  # NSUTF8StringEncoding = 4
            )

            # Create an NSURL object representing the URL
            # This is the equivalent of:
            #   NSURL *nsurl = [NSURL URLWithString:url];
            NSURL = objc.objc_getClass(b"NSURL")
            urlWithString_ = objc.sel_registerName(b"URLWithString:")
            objc.objc_msgSend.argtypes = [c_void_p, c_void_p, c_void_p]
            ns_url = objc.objc_msgSend(NSURL, urlWithString_, url_string)

            # Get the shared UIApplication instance
            # This code is the equivalent of:
            # UIApplication shared_app = [UIApplication sharedApplication]
            UIApplication = objc.objc_getClass(b"UIApplication")
            sharedApplication = objc.sel_registerName(b"sharedApplication")
            objc.objc_msgSend.argtypes = [c_void_p, c_void_p]
            shared_app = objc.objc_msgSend(UIApplication, sharedApplication)

            # Open the URL on the shared application
            # This code is the equivalent of:
            #   [shared_app openURL:ns_url
            #               options:NIL
            #     completionHandler:NIL];
            openURL_ = objc.sel_registerName(b"openURL:options:completionHandler:")
            objc.objc_msgSend.argtypes = [
                c_void_p, c_void_p, c_void_p, c_void_p, c_void_p
            ]
            # Method returns void
            objc.objc_msgSend.restype = None
            objc.objc_msgSend(shared_app, openURL_, ns_url, None, None)

            return True


def parse_args(arg_list: list[str] | None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Open URL in a web browser.", color=True,
    )
    parser.add_argument("url", help="URL to open")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-n", "--new-window", action="store_const",
                       const=1, default=0, dest="new_win",
                       help="open new window")
    group.add_argument("-t", "--new-tab", action="store_const",
                       const=2, default=0, dest="new_win",
                       help="open new tab")

    args = parser.parse_args(arg_list)

    return args


def main(arg_list: list[str] | None = None):
    args = parse_args(arg_list)

    open(args.url, args.new_win)

    print("\a")


if __name__ == "__main__":
    main()

"""Convert a NT pathname to a file URL and vice versa.

This module only exists to provide OS-specific code
for urllib.requests, thus do not use directly.
"""
# Testing is done through test_nturl2path.

import warnings


warnings._deprecated(
    __name__,
    message=f"{warnings._DEPRECATED_MSG}; use 'urllib.request' instead",
    remove=(3, 19))

def url2pathname(url):
    """OS-specific conversion from a relative URL of the 'file' scheme
    to a file system path; not recommended for general use."""
    # e.g.
    #   ///C|/foo/bar/spam.foo
    # and
    #   ///C:/foo/bar/spam.foo
    # become
    #   C:\foo\bar\spam.foo
    import urllib.parse
    if url[:3] == '///':
        # URL has an empty authority section, so the path begins on the third
        # character.
        url = url[2:]
    elif url[:12] == '//localhost/':
        # Skip past 'localhost' authority.
        url = url[11:]
    if url[:3] == '///':
        # Skip past extra slash before UNC drive in URL path.
        url = url[1:]
    else:
        if url[:1] == '/' and url[2:3] in (':', '|'):
            # Skip past extra slash before DOS drive in URL path.
            url = url[1:]
        if url[1:2] == '|':
            # Older URLs use a pipe after a drive letter
            url = url[:1] + ':' + url[2:]
    return urllib.parse.unquote(url.replace('/', '\\'))

def pathname2url(p):
    """OS-specific conversion from a file system path to a relative URL
    of the 'file' scheme; not recommended for general use."""
    # e.g.
    #   C:\foo\bar\spam.foo
    # becomes
    #   ///C:/foo/bar/spam.foo
    import ntpath
    import urllib.parse
    # First, clean up some special forms. We are going to sacrifice
    # the additional information anyway
    p = p.replace('\\', '/')
    if p[:4] == '//?/':
        p = p[4:]
        if p[:4].upper() == 'UNC/':
            p = '//' + p[4:]
    drive, root, tail = ntpath.splitroot(p)
    if drive:
        if drive[1:] == ':':
            # DOS drive specified. Add three slashes to the start, producing
            # an authority section with a zero-length authority, and a path
            # section starting with a single slash.
            drive = f'///{drive}'
        drive = urllib.parse.quote(drive, safe='/:')
    elif root:
        # Add explicitly empty authority to path beginning with one slash.
        root = f'//{root}'

    tail = urllib.parse.quote(tail)
    return drive + root + tail

"""Temporary files.

This module provides generic, low- and high-level interfaces for
creating temporary files and directories.  All of the interfaces
provided by this module can be used without fear of race conditions
except for 'mktemp'.  'mktemp' is subject to race conditions and
should not be used; it is provided for backward compatibility only.

The default path names are returned as str.  If you supply bytes as
input, all return values will be in bytes.  Ex:

    >>> tempfile.mkstemp()
    (4, '/tmp/tmptpu9nin8')
    >>> tempfile.mkdtemp(suffix=b'')
    b'/tmp/tmppbi8f0hy'

This module also provides some data items to the user:

  TMP_MAX  - maximum number of names that will be tried before
             giving up.
  tempdir  - If this is set to a string before the first use of
             any routine from this module, it will be considered as
             another candidate location to store temporary files.
"""

__all__ = [
    "NamedTemporaryFile", "TemporaryFile", # high level safe interfaces
    "SpooledTemporaryFile", "TemporaryDirectory",
    "mkstemp", "mkdtemp",                  # low level safe interfaces
    "mktemp",                              # deprecated unsafe interface
    "TMP_MAX", "gettempprefix",            # constants
    "tempdir", "gettempdir",
    "gettempprefixb", "gettempdirb",
   ]


# Imports.

import functools as _functools
import warnings as _warnings
import io as _io
import os as _os
import shutil as _shutil
import errno as _errno
from random import Random as _Random
import sys as _sys
import types as _types
import weakref as _weakref
import _thread
_allocate_lock = _thread.allocate_lock

_text_openflags = _os.O_RDWR | _os.O_CREAT | _os.O_EXCL
if hasattr(_os, 'O_NOFOLLOW'):
    _text_openflags |= _os.O_NOFOLLOW

_bin_openflags = _text_openflags
if hasattr(_os, 'O_BINARY'):
    _bin_openflags |= _os.O_BINARY

# This is more than enough.
# Each name contains over 40 random bits.  Even with a million temporary
# files, the chance of a conflict is less than 1 in a million, and with
# 20 attempts, it is less than 1e-120.
TMP_MAX = 20

# This variable _was_ unused for legacy reasons, see issue 10354.
# But as of 3.5 we actually use it at runtime so changing it would
# have a possibly desirable side effect...  But we do not want to support
# that as an API.  It is undocumented on purpose.  Do not depend on this.
template = "tmp"

# Internal routines.

_once_lock = _allocate_lock()


def _exists(fn):
    try:
        _os.lstat(fn)
    except OSError:
        return False
    else:
        return True


def _infer_return_type(*args):
    """Look at the type of all args and divine their implied return type."""
    return_type = None
    for arg in args:
        if arg is None:
            continue

        if isinstance(arg, _os.PathLike):
            arg = _os.fspath(arg)

        if isinstance(arg, bytes):
            if return_type is str:
                raise TypeError("Can't mix bytes and non-bytes in "
                                "path components.")
            return_type = bytes
        else:
            if return_type is bytes:
                raise TypeError("Can't mix bytes and non-bytes in "
                                "path components.")
            return_type = str
    if return_type is None:
        if tempdir is None or isinstance(tempdir, str):
            return str  # tempfile APIs return a str by default.
        else:
            # we could check for bytes but it'll fail later on anyway
            return bytes
    return return_type


def _sanitize_params(prefix, suffix, dir):
    """Common parameter processing for most APIs in this module."""
    output_type = _infer_return_type(prefix, suffix, dir)
    if suffix is None:
        suffix = output_type()
    if prefix is None:
        if output_type is str:
            prefix = template
        else:
            prefix = _os.fsencode(template)
    if dir is None:
        if output_type is str:
            dir = gettempdir()
        else:
            dir = gettempdirb()
    return prefix, suffix, dir, output_type


class _RandomNameSequence:
    """An instance of _RandomNameSequence generates an endless
    sequence of unpredictable strings which can safely be incorporated
    into file names.  Each string is eight characters long.  Multiple
    threads can safely use the same instance at the same time.

    _RandomNameSequence is an iterator."""

    characters = "abcdefghijklmnopqrstuvwxyz0123456789_"

    @property
    def rng(self):
        cur_pid = _os.getpid()
        if cur_pid != getattr(self, '_rng_pid', None):
            self._rng = _Random()
            self._rng_pid = cur_pid
        return self._rng

    def __iter__(self):
        return self

    def __next__(self):
        return ''.join(self.rng.choices(self.characters, k=8))

def _candidate_tempdir_list():
    """Generate a list of candidate temporary directories which
    _get_default_tempdir will try."""

    dirlist = []

    # First, try the environment.
    for envname in 'TMPDIR', 'TEMP', 'TMP':
        dirname = _os.getenv(envname)
        if dirname: dirlist.append(dirname)

    # Failing that, try OS-specific locations.
    if _os.name == 'nt':
        dirlist.extend([ _os.path.expanduser(r'~\AppData\Local\Temp'),
                         _os.path.expandvars(r'%SYSTEMROOT%\Temp'),
                         r'c:\temp', r'c:\tmp', r'\temp', r'\tmp' ])
    else:
        dirlist.extend([ '/tmp', '/var/tmp', '/usr/tmp' ])

    # As a last resort, the current directory.
    try:
        dirlist.append(_os.getcwd())
    except (AttributeError, OSError):
        dirlist.append(_os.curdir)

    return dirlist

def _get_default_tempdir(dirlist=None):
    """Calculate the default directory to use for temporary files.
    This routine should be called exactly once.

    We determine whether or not a candidate temp dir is usable by
    trying to create and write to a file in that directory.  If this
    is successful, the test file is deleted.  To prevent denial of
    service, the name of the test file must be randomized."""

    namer = _RandomNameSequence()
    if dirlist is None:
        dirlist = _candidate_tempdir_list()

    for dir in dirlist:
        if dir != _os.curdir:
            dir = _os.path.abspath(dir)
        for seq in range(TMP_MAX):
            name = next(namer)
            filename = _os.path.join(dir, name)
            try:
                fd = _os.open(filename, _bin_openflags, 0o600)
                try:
                    try:
                        _os.write(fd, b'blat')
                    finally:
                        _os.close(fd)
                finally:
                    _os.unlink(filename)
                return dir
            except FileExistsError:
                pass
            except PermissionError:
                # See the comment in mkdtemp().
                if _os.name == 'nt' and _os.path.isdir(dir):
                    continue
                break   # no point trying more names in this directory
            except OSError:
                break   # no point trying more names in this directory
    raise FileNotFoundError(_errno.ENOENT,
                            "No usable temporary directory found in %s" %
                            dirlist)

_name_sequence = None

def _get_candidate_names():
    """Common setup sequence for all user-callable interfaces."""

    global _name_sequence
    if _name_sequence is None:
        _once_lock.acquire()
        try:
            if _name_sequence is None:
                _name_sequence = _RandomNameSequence()
        finally:
            _once_lock.release()
    return _name_sequence


def _mkstemp_inner(dir, pre, suf, flags, output_type):
    """Code common to mkstemp, TemporaryFile, and NamedTemporaryFile."""

    dir = _os.path.abspath(dir)
    names = _get_candidate_names()
    if output_type is bytes:
        names = map(_os.fsencode, names)

    for seq in range(TMP_MAX):
        name = next(names)
        file = _os.path.join(dir, pre + name + suf)
        _sys.audit("tempfile.mkstemp", file)
        try:
            fd = _os.open(file, flags, 0o600)
        except FileExistsError:
            continue    # try again
        except PermissionError:
            # See the comment in mkdtemp().
            if _os.name == 'nt' and _os.path.isdir(dir) and seq < TMP_MAX - 1:
                continue
            else:
                raise
        return fd, file

    raise FileExistsError(_errno.EEXIST,
                          "No usable temporary file name found")

def _dont_follow_symlinks(func, path, *args):
    # Pass follow_symlinks=False, unless not supported on this platform.
    if func in _os.supports_follow_symlinks:
        func(path, *args, follow_symlinks=False)
    elif not _os.path.islink(path):
        func(path, *args)

def _resetperms(path):
    try:
        chflags = _os.chflags
    except AttributeError:
        pass
    else:
        _dont_follow_symlinks(chflags, path, 0)
    _dont_follow_symlinks(_os.chmod, path, 0o700)


# User visible interfaces.

def gettempprefix():
    """The default prefix for temporary directories as string."""
    return _os.fsdecode(template)

def gettempprefixb():
    """The default prefix for temporary directories as bytes."""
    return _os.fsencode(template)

tempdir = None

def _gettempdir():
    """Private accessor for tempfile.tempdir."""
    global tempdir
    if tempdir is None:
        _once_lock.acquire()
        try:
            if tempdir is None:
                tempdir = _get_default_tempdir()
        finally:
            _once_lock.release()
    return tempdir

def gettempdir():
    """Returns tempfile.tempdir as str."""
    return _os.fsdecode(_gettempdir())

def gettempdirb():
    """Returns tempfile.tempdir as bytes."""
    return _os.fsencode(_gettempdir())

def mkstemp(suffix=None, prefix=None, dir=None, text=False):
    """User-callable function to create and return a unique temporary
    file.  The return value is a pair (fd, name) where fd is the
    file descriptor returned by os.open, and name is the filename.

    If 'suffix' is not None, the file name will end with that suffix,
    otherwise there will be no suffix.

    If 'prefix' is not None, the file name will begin with that prefix,
    otherwise a default prefix is used.

    If 'dir' is not None, the file will be created in that directory,
    otherwise a default directory is used.

    If 'text' is specified and true, the file is opened in text
    mode.  Else (the default) the file is opened in binary mode.

    If any of 'suffix', 'prefix' and 'dir' are not None, they must be the
    same type.  If they are bytes, the returned name will be bytes; str
    otherwise.

    The file is readable and writable only by the creating user ID.
    If the operating system uses permission bits to indicate whether a
    file is executable, the file is executable by no one. The file
    descriptor is not inherited by children of this process.

    Caller is responsible for deleting the file when done with it.
    """

    prefix, suffix, dir, output_type = _sanitize_params(prefix, suffix, dir)

    if text:
        flags = _text_openflags
    else:
        flags = _bin_openflags

    return _mkstemp_inner(dir, prefix, suffix, flags, output_type)


def mkdtemp(suffix=None, prefix=None, dir=None):
    """User-callable function to create and return a unique temporary
    directory.  The return value is the pathname of the directory.

    Arguments are as for mkstemp, except that the 'text' argument is
    not accepted.

    The directory is readable, writable, and searchable only by the
    creating user.

    Caller is responsible for deleting the directory when done with it.
    """

    prefix, suffix, dir, output_type = _sanitize_params(prefix, suffix, dir)

    names = _get_candidate_names()
    if output_type is bytes:
        names = map(_os.fsencode, names)

    for seq in range(TMP_MAX):
        name = next(names)
        file = _os.path.join(dir, prefix + name + suffix)
        _sys.audit("tempfile.mkdtemp", file)
        try:
            _os.mkdir(file, 0o700)
        except FileExistsError:
            continue    # try again
        except PermissionError:
            # On Posix, this exception is raised when the user has no
            # write access to the parent directory.
            # On Windows, it is also raised when a directory with
            # the chosen name already exists, or if the parent directory
            # is not a directory.
            # We cannot distinguish between "directory-exists-error" and
            # "access-denied-error".
            if _os.name == 'nt' and _os.path.isdir(dir) and seq < TMP_MAX - 1:
                continue
            else:
                raise
        return _os.path.abspath(file)

    raise FileExistsError(_errno.EEXIST,
                          "No usable temporary directory name found")

def mktemp(suffix="", prefix=template, dir=None):
    """User-callable function to return a unique temporary file name.  The
    file is not created.

    Arguments are similar to mkstemp, except that the 'text' argument is
    not accepted, and suffix=None, prefix=None and bytes file names are not
    supported.

    THIS FUNCTION IS UNSAFE AND SHOULD NOT BE USED.  The file name may
    refer to a file that did not exist at some point, but by the time
    you get around to creating it, someone else may have beaten you to
    the punch.
    """

##    from warnings import warn as _warn
##    _warn("mktemp is a potential security risk to your program",
##          RuntimeWarning, stacklevel=2)

    if dir is None:
        dir = gettempdir()

    names = _get_candidate_names()
    for seq in range(TMP_MAX):
        name = next(names)
        file = _os.path.join(dir, prefix + name + suffix)
        if not _exists(file):
            return file

    raise FileExistsError(_errno.EEXIST,
                          "No usable temporary filename found")


class _TemporaryFileCloser:
    """A separate object allowing proper closing of a temporary file's
    underlying file object, without adding a __del__ method to the
    temporary file."""

    cleanup_called = False
    close_called = False

    def __init__(
        self,
        file,
        name,
        delete=True,
        delete_on_close=True,
        warn_message="Implicitly cleaning up unknown file",
    ):
        self.file = file
        self.name = name
        self.delete = delete
        self.delete_on_close = delete_on_close
        self.warn_message = warn_message

    def cleanup(self, windows=(_os.name == 'nt'), unlink=_os.unlink):
        if not self.cleanup_called:
            self.cleanup_called = True
            try:
                if not self.close_called:
                    self.close_called = True
                    self.file.close()
            finally:
                # Windows provides delete-on-close as a primitive, in which
                # case the file was deleted by self.file.close().
                if self.delete and not (windows and self.delete_on_close):
                    try:
                        unlink(self.name)
                    except FileNotFoundError:
                        pass

    def close(self):
        if not self.close_called:
            self.close_called = True
            try:
                self.file.close()
            finally:
                if self.delete and self.delete_on_close:
                    self.cleanup()

    def __del__(self):
        close_called = self.close_called
        self.cleanup()
        if not close_called:
            _warnings.warn(self.warn_message, ResourceWarning)


class _TemporaryFileWrapper:
    """Temporary file wrapper

    This class provides a wrapper around files opened for
    temporary use.  In particular, it seeks to automatically
    remove the file when it is no longer needed.
    """

    def __init__(self, file, name, delete=True, delete_on_close=True):
        self.file = file
        self.name = name
        self._closer = _TemporaryFileCloser(
            file,
            name,
            delete,
            delete_on_close,
            warn_message=f"Implicitly cleaning up {self!r}",
        )

    def __repr__(self):
        file = self.__dict__['file']
        return f"<{type(self).__name__} {file=}>"

    def __getattr__(self, name):
        # Attribute lookups are delegated to the underlying file
        # and cached for non-numeric results
        # (i.e. methods are cached, closed and friends are not)
        file = self.__dict__['file']
        a = getattr(file, name)
        if hasattr(a, '__call__'):
            func = a
            @_functools.wraps(func)
            def func_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            # Avoid closing the file as long as the wrapper is alive,
            # see issue #18879.
            func_wrapper._closer = self._closer
            a = func_wrapper
        if not isinstance(a, int):
            setattr(self, name, a)
        return a

    # The underlying __enter__ method returns the wrong object
    # (self.file) so override it to return the wrapper
    def __enter__(self):
        self.file.__enter__()
        return self

    # Need to trap __exit__ as well to ensure the file gets
    # deleted when used in a with statement
    def __exit__(self, exc, value, tb):
        result = self.file.__exit__(exc, value, tb)
        self._closer.cleanup()
        return result

    def close(self):
        """
        Close the temporary file, possibly deleting it.
        """
        self._closer.close()

    # iter() doesn't use __getattr__ to find the __iter__ method
    def __iter__(self):
        # Don't return iter(self.file), but yield from it to avoid closing
        # file as long as it's being used as iterator (see issue #23700).  We
        # can't use 'yield from' here because iter(file) returns the file
        # object itself, which has a close method, and thus the file would get
        # closed when the generator is finalized, due to PEP380 semantics.
        for line in self.file:
            yield line

def NamedTemporaryFile(mode='w+b', buffering=-1, encoding=None,
                       newline=None, suffix=None, prefix=None,
                       dir=None, delete=True, *, errors=None,
                       delete_on_close=True):
    """Create and return a temporary file.
    Arguments:
    'prefix', 'suffix', 'dir' -- as for mkstemp.
    'mode' -- the mode argument to io.open (default "w+b").
    'buffering' -- the buffer size argument to io.open (default -1).
    'encoding' -- the encoding argument to io.open (default None)
    'newline' -- the newline argument to io.open (default None)
    'delete' -- whether the file is automatically deleted (default True).
    'delete_on_close' -- if 'delete', whether the file is deleted on close
       (default True) or otherwise either on context manager exit
       (if context manager was used) or on object finalization. .
    'errors' -- the errors argument to io.open (default None)
    The file is created as mkstemp() would do it.

    Returns an object with a file-like interface; the name of the file
    is accessible as its 'name' attribute.  The file will be automatically
    deleted when it is closed unless the 'delete' argument is set to False.

    On POSIX, NamedTemporaryFiles cannot be automatically deleted if
    the creating process is terminated abruptly with a SIGKILL signal.
    Windows can delete the file even in this case.
    """

    prefix, suffix, dir, output_type = _sanitize_params(prefix, suffix, dir)

    flags = _bin_openflags

    # Setting O_TEMPORARY in the flags causes the OS to delete
    # the file when it is closed.  This is only supported by Windows.
    if _os.name == 'nt' and delete and delete_on_close:
        flags |= _os.O_TEMPORARY

    if "b" not in mode:
        encoding = _io.text_encoding(encoding)

    name = None
    def opener(*args):
        nonlocal name
        fd, name = _mkstemp_inner(dir, prefix, suffix, flags, output_type)
        return fd
    try:
        file = _io.open(dir, mode, buffering=buffering,
                        newline=newline, encoding=encoding, errors=errors,
                        opener=opener)
        try:
            raw = getattr(file, 'buffer', file)
            raw = getattr(raw, 'raw', raw)
            raw.name = name
            return _TemporaryFileWrapper(file, name, delete, delete_on_close)
        except:
            file.close()
            raise
    except:
        if name is not None and not (
            _os.name == 'nt' and delete and delete_on_close):
            _os.unlink(name)
        raise

if _os.name != 'posix' or _sys.platform == 'cygwin':
    # On non-POSIX and Cygwin systems, assume that we cannot unlink a file
    # while it is open.
    TemporaryFile = NamedTemporaryFile

else:
    # Is the O_TMPFILE flag available and does it work?
    # The flag is set to False if os.open(dir, os.O_TMPFILE) raises an
    # IsADirectoryError exception
    _O_TMPFILE_WORKS = hasattr(_os, 'O_TMPFILE')

    def TemporaryFile(mode='w+b', buffering=-1, encoding=None,
                      newline=None, suffix=None, prefix=None,
                      dir=None, *, errors=None):
        """Create and return a temporary file.
        Arguments:
        'prefix', 'suffix', 'dir' -- as for mkstemp.
        'mode' -- the mode argument to io.open (default "w+b").
        'buffering' -- the buffer size argument to io.open (default -1).
        'encoding' -- the encoding argument to io.open (default None)
        'newline' -- the newline argument to io.open (default None)
        'errors' -- the errors argument to io.open (default None)
        The file is created as mkstemp() would do it.

        Returns an object with a file-like interface.  The file has no
        name, and will cease to exist when it is closed.
        """
        global _O_TMPFILE_WORKS

        if "b" not in mode:
            encoding = _io.text_encoding(encoding)

        prefix, suffix, dir, output_type = _sanitize_params(prefix, suffix, dir)

        flags = _bin_openflags
        if _O_TMPFILE_WORKS:
            fd = None
            def opener(*args):
                nonlocal fd
                flags2 = (flags | _os.O_TMPFILE) & ~_os.O_CREAT
                fd = _os.open(dir, flags2, 0o600)
                return fd
            try:
                file = _io.open(dir, mode, buffering=buffering,
                                newline=newline, encoding=encoding,
                                errors=errors, opener=opener)
                raw = getattr(file, 'buffer', file)
                raw = getattr(raw, 'raw', raw)
                raw.name = fd
                return file
            except IsADirectoryError:
                # Linux kernel older than 3.11 ignores the O_TMPFILE flag:
                # O_TMPFILE is read as O_DIRECTORY. Trying to open a directory
                # with O_RDWR|O_DIRECTORY fails with IsADirectoryError, a
                # directory cannot be open to write. Set flag to False to not
                # try again.
                _O_TMPFILE_WORKS = False
            except OSError:
                # The filesystem of the directory does not support O_TMPFILE.
                # For example, OSError(95, 'Operation not supported').
                #
                # On Linux kernel older than 3.11, trying to open a regular
                # file (or a symbolic link to a regular file) with O_TMPFILE
                # fails with NotADirectoryError, because O_TMPFILE is read as
                # O_DIRECTORY.
                pass
            # Fallback to _mkstemp_inner().

        fd = None
        def opener(*args):
            nonlocal fd
            fd, name = _mkstemp_inner(dir, prefix, suffix, flags, output_type)
            try:
                _os.unlink(name)
            except BaseException as e:
                _os.close(fd)
                raise
            return fd
        file = _io.open(dir, mode, buffering=buffering,
                        newline=newline, encoding=encoding, errors=errors,
                        opener=opener)
        raw = getattr(file, 'buffer', file)
        raw = getattr(raw, 'raw', raw)
        raw.name = fd
        return file

class SpooledTemporaryFile(_io.IOBase):
    """Temporary file wrapper, specialized to switch from BytesIO
    or StringIO to a real file when it exceeds a certain size or
    when a fileno is needed.
    """
    _rolled = False

    def __init__(self, max_size=0, mode='w+b', buffering=-1,
                 encoding=None, newline=None,
                 suffix=None, prefix=None, dir=None, *, errors=None):
        if 'b' in mode:
            self._file = _io.BytesIO()
        else:
            encoding = _io.text_encoding(encoding)
            self._file = _io.TextIOWrapper(_io.BytesIO(),
                            encoding=encoding, errors=errors,
                            newline=newline)
        self._max_size = max_size
        self._rolled = False
        self._TemporaryFileArgs = {'mode': mode, 'buffering': buffering,
                                   'suffix': suffix, 'prefix': prefix,
                                   'encoding': encoding, 'newline': newline,
                                   'dir': dir, 'errors': errors}

    __class_getitem__ = classmethod(_types.GenericAlias)

    def _check(self, file):
        if self._rolled: return
        max_size = self._max_size
        if max_size and file.tell() > max_size:
            self.rollover()

    def rollover(self):
        if self._rolled: return
        file = self._file
        newfile = self._file = TemporaryFile(**self._TemporaryFileArgs)
        del self._TemporaryFileArgs

        pos = file.tell()
        if hasattr(newfile, 'buffer'):
            newfile.buffer.write(file.detach().getvalue())
        else:
            newfile.write(file.getvalue())
        newfile.seek(pos, 0)

        self._rolled = True

    # The method caching trick from NamedTemporaryFile
    # won't work here, because _file may change from a
    # BytesIO/StringIO instance to a real file. So we list
    # all the methods directly.

    # Context management protocol
    def __enter__(self):
        if self._file.closed:
            raise ValueError("Cannot enter context with closed file")
        return self

    def __exit__(self, exc, value, tb):
        self._file.close()

    # file protocol
    def __iter__(self):
        return self._file.__iter__()

    def __del__(self):
        if not self.closed:
            _warnings.warn(
                "Unclosed file {!r}".format(self),
                ResourceWarning,
                stacklevel=2,
                source=self
            )
            self.close()

    def close(self):
        self._file.close()

    @property
    def closed(self):
        return self._file.closed

    @property
    def encoding(self):
        return self._file.encoding

    @property
    def errors(self):
        return self._file.errors

    def fileno(self):
        self.rollover()
        return self._file.fileno()

    def flush(self):
        self._file.flush()

    def isatty(self):
        return self._file.isatty()

    @property
    def mode(self):
        try:
            return self._file.mode
        except AttributeError:
            return self._TemporaryFileArgs['mode']

    @property
    def name(self):
        try:
            return self._file.name
        except AttributeError:
            return None

    @property
    def newlines(self):
        return self._file.newlines

    def readable(self):
        return self._file.readable()

    def read(self, *args):
        return self._file.read(*args)

    def read1(self, *args):
        return self._file.read1(*args)

    def readinto(self, b):
        return self._file.readinto(b)

    def readinto1(self, b):
        return self._file.readinto1(b)

    def readline(self, *args):
        return self._file.readline(*args)

    def readlines(self, *args):
        return self._file.readlines(*args)

    def seekable(self):
        return self._file.seekable()

    def seek(self, *args):
        return self._file.seek(*args)

    def tell(self):
        return self._file.tell()

    def truncate(self, size=None):
        if size is None:
            return self._file.truncate()
        else:
            if size > self._max_size:
                self.rollover()
            return self._file.truncate(size)

    def writable(self):
        return self._file.writable()

    def write(self, s):
        file = self._file
        rv = file.write(s)
        self._check(file)
        return rv

    def writelines(self, iterable):
        if self._max_size == 0 or self._rolled:
            return self._file.writelines(iterable)

        it = iter(iterable)
        for line in it:
            self.write(line)
            if self._rolled:
                return self._file.writelines(it)

    def detach(self):
        return self._file.detach()


class TemporaryDirectory:
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed (unless delete=False is passed or an exception
    is raised during cleanup and ignore_cleanup_errors is not True).

    Optional Arguments:
        suffix - A str suffix for the directory name.  (see mkdtemp)
        prefix - A str prefix for the directory name.  (see mkdtemp)
        dir - A directory to create this temp dir in.  (see mkdtemp)
        ignore_cleanup_errors - False; ignore exceptions during cleanup?
        delete - True; whether the directory is automatically deleted.
    """

    def __init__(self, suffix=None, prefix=None, dir=None,
                 ignore_cleanup_errors=False, *, delete=True):
        self.name = mkdtemp(suffix, prefix, dir)
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._delete = delete
        self._finalizer = _weakref.finalize(
            self, self._cleanup, self.name,
            warn_message="Implicitly cleaning up {!r}".format(self),
            ignore_errors=self._ignore_cleanup_errors, delete=self._delete)

    @classmethod
    def _rmtree(cls, name, ignore_errors=False, repeated=False):
        def onexc(func, path, exc):
            # On DragonFly BSD, UF_NOUNLINK removal fails with EISDIR, not EPERM.
            if isinstance(exc, (PermissionError, IsADirectoryError)):
                if repeated and path == name:
                    if ignore_errors:
                        return
                    raise

                try:
                    if path != name:
                        _resetperms(_os.path.dirname(path))
                    _resetperms(path)

                    try:
                        _os.unlink(path)
                    except IsADirectoryError:
                        cls._rmtree(path, ignore_errors=ignore_errors)
                    except PermissionError:
                        # The PermissionError handler was originally added for
                        # FreeBSD in directories, but it seems that it is raised
                        # on Windows too.
                        # bpo-43153: Calling _rmtree again may
                        # raise NotADirectoryError and mask the PermissionError.
                        # So we must re-raise the current PermissionError if
                        # path is not a directory.
                        if not _os.path.isdir(path) or _os.path.isjunction(path):
                            if ignore_errors:
                                return
                            raise
                        cls._rmtree(path, ignore_errors=ignore_errors,
                                    repeated=(path == name))
                except FileNotFoundError:
                    pass
            elif isinstance(exc, FileNotFoundError):
                pass
            else:
                if not ignore_errors:
                    raise

        _shutil.rmtree(name, onexc=onexc)

    @classmethod
    def _cleanup(cls, name, warn_message, ignore_errors=False, delete=True):
        if delete:
            cls._rmtree(name, ignore_errors=ignore_errors)
            _warnings.warn(warn_message, ResourceWarning)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def __exit__(self, exc, value, tb):
        if self._delete:
            self.cleanup()

    def cleanup(self):
        if self._finalizer.detach() or _os.path.exists(self.name):
            self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)

    __class_getitem__ = classmethod(_types.GenericAlias)

#  Author:      Fred L. Drake, Jr.
#               fdrake@acm.org
#
#  This is a simple little module I wrote to make life easier.  I didn't
#  see anything quite like it in the library, though I may have overlooked
#  something.  I wrote this when I was trying to read some heavily nested
#  tuples with fairly non-descriptive content.  This is modeled very much
#  after Lisp/Scheme - style pretty-printing of lists.  If you find it
#  useful, thank small children who sleep at night.

"""Support to pretty-print lists, tuples, & dictionaries recursively.

Very simple, but useful, especially in debugging data structures.

Classes
-------

PrettyPrinter()
    Handle pretty-printing operations onto a stream using a configured
    set of formatting parameters.

Functions
---------

pformat()
    Format a Python object into a pretty-printed representation.

pprint()
    Pretty-print a Python object to a stream [default is sys.stdout].

saferepr()
    Generate a 'standard' repr()-like value, but protect against recursive
    data structures.

"""

import collections as _collections
import sys as _sys
import types as _types
from io import StringIO as _StringIO

__all__ = ["pprint","pformat","isreadable","isrecursive","saferepr",
           "PrettyPrinter", "pp"]


def pprint(object, stream=None, indent=1, width=80, depth=None, *,
           compact=False, sort_dicts=True, underscore_numbers=False):
    """Pretty-print a Python object to a stream [default is sys.stdout]."""
    printer = PrettyPrinter(
        stream=stream, indent=indent, width=width, depth=depth,
        compact=compact, sort_dicts=sort_dicts,
        underscore_numbers=underscore_numbers)
    printer.pprint(object)


def pformat(object, indent=1, width=80, depth=None, *,
            compact=False, sort_dicts=True, underscore_numbers=False):
    """Format a Python object into a pretty-printed representation."""
    return PrettyPrinter(indent=indent, width=width, depth=depth,
                         compact=compact, sort_dicts=sort_dicts,
                         underscore_numbers=underscore_numbers).pformat(object)


def pp(object, *args, sort_dicts=False, **kwargs):
    """Pretty-print a Python object"""
    pprint(object, *args, sort_dicts=sort_dicts, **kwargs)


def saferepr(object):
    """Version of repr() which can handle recursive data structures."""
    return PrettyPrinter()._safe_repr(object, {}, None, 0)[0]


def isreadable(object):
    """Determine if saferepr(object) is readable by eval()."""
    return PrettyPrinter()._safe_repr(object, {}, None, 0)[1]


def isrecursive(object):
    """Determine if object requires a recursive representation."""
    return PrettyPrinter()._safe_repr(object, {}, None, 0)[2]


class _safe_key:
    """Helper function for key functions when sorting unorderable objects.

    The wrapped-object will fallback to a Py2.x style comparison for
    unorderable types (sorting first comparing the type name and then by
    the obj ids).  Does not work recursively, so dict.items() must have
    _safe_key applied to both the key and the value.

    """

    __slots__ = ['obj']

    def __init__(self, obj):
        self.obj = obj

    def __lt__(self, other):
        try:
            return self.obj < other.obj
        except TypeError:
            return ((str(type(self.obj)), id(self.obj)) < \
                    (str(type(other.obj)), id(other.obj)))


def _safe_tuple(t):
    "Helper function for comparing 2-tuples"
    return _safe_key(t[0]), _safe_key(t[1])


class PrettyPrinter:
    def __init__(self, indent=1, width=80, depth=None, stream=None, *,
                 compact=False, sort_dicts=True, underscore_numbers=False):
        """Handle pretty printing operations onto a stream using a set of
        configured parameters.

        indent
            Number of spaces to indent for each level of nesting.

        width
            Attempted maximum number of columns in the output.

        depth
            The maximum depth to print out nested structures.

        stream
            The desired output stream.  If omitted (or false), the standard
            output stream available at construction will be used.

        compact
            If true, several items will be combined in one line.

        sort_dicts
            If true, dict keys are sorted.

        underscore_numbers
            If true, digit groups are separated with underscores.

        """
        indent = int(indent)
        width = int(width)
        if indent < 0:
            raise ValueError('indent must be >= 0')
        if depth is not None and depth <= 0:
            raise ValueError('depth must be > 0')
        if not width:
            raise ValueError('width must be != 0')
        self._depth = depth
        self._indent_per_level = indent
        self._width = width
        if stream is not None:
            self._stream = stream
        else:
            self._stream = _sys.stdout
        self._compact = bool(compact)
        self._sort_dicts = sort_dicts
        self._underscore_numbers = underscore_numbers

    def pprint(self, object):
        if self._stream is not None:
            self._format(object, self._stream, 0, 0, {}, 0)
            self._stream.write("\n")

    def pformat(self, object):
        sio = _StringIO()
        self._format(object, sio, 0, 0, {}, 0)
        return sio.getvalue()

    def isrecursive(self, object):
        return self.format(object, {}, 0, 0)[2]

    def isreadable(self, object):
        s, readable, recursive = self.format(object, {}, 0, 0)
        return readable and not recursive

    def _format(self, object, stream, indent, allowance, context, level):
        objid = id(object)
        if objid in context:
            stream.write(_recursion(object))
            self._recursive = True
            self._readable = False
            return
        rep = self._repr(object, context, level)
        max_width = self._width - indent - allowance
        if len(rep) > max_width:
            p = self._dispatch.get(type(object).__repr__, None)
            # Lazy import to improve module import time
            from dataclasses import is_dataclass

            if p is not None:
                context[objid] = 1
                p(self, object, stream, indent, allowance, context, level + 1)
                del context[objid]
                return
            elif (is_dataclass(object) and
                  not isinstance(object, type) and
                  object.__dataclass_params__.repr and
                  # Check dataclass has generated repr method.
                  hasattr(object.__repr__, "__wrapped__") and
                  "__create_fn__" in object.__repr__.__wrapped__.__qualname__):
                context[objid] = 1
                self._pprint_dataclass(object, stream, indent, allowance, context, level + 1)
                del context[objid]
                return
        stream.write(rep)

    def _pprint_dataclass(self, object, stream, indent, allowance, context, level):
        # Lazy import to improve module import time
        from dataclasses import fields as dataclass_fields

        cls_name = object.__class__.__name__
        indent += len(cls_name) + 1
        items = [(f.name, getattr(object, f.name)) for f in dataclass_fields(object) if f.repr]
        stream.write(cls_name + '(')
        self._format_namespace_items(items, stream, indent, allowance, context, level)
        stream.write(')')

    _dispatch = {}

    def _pprint_dict(self, object, stream, indent, allowance, context, level):
        write = stream.write
        write('{')
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * ' ')
        length = len(object)
        if length:
            if self._sort_dicts:
                items = sorted(object.items(), key=_safe_tuple)
            else:
                items = object.items()
            self._format_dict_items(items, stream, indent, allowance + 1,
                                    context, level)
        write('}')

    _dispatch[dict.__repr__] = _pprint_dict

    def _pprint_ordered_dict(self, object, stream, indent, allowance, context, level):
        if not len(object):
            stream.write(repr(object))
            return
        cls = object.__class__
        stream.write(cls.__name__ + '(')
        self._format(list(object.items()), stream,
                     indent + len(cls.__name__) + 1, allowance + 1,
                     context, level)
        stream.write(')')

    _dispatch[_collections.OrderedDict.__repr__] = _pprint_ordered_dict

    def _pprint_list(self, object, stream, indent, allowance, context, level):
        stream.write('[')
        self._format_items(object, stream, indent, allowance + 1,
                           context, level)
        stream.write(']')

    _dispatch[list.__repr__] = _pprint_list

    def _pprint_tuple(self, object, stream, indent, allowance, context, level):
        stream.write('(')
        endchar = ',)' if len(object) == 1 else ')'
        self._format_items(object, stream, indent, allowance + len(endchar),
                           context, level)
        stream.write(endchar)

    _dispatch[tuple.__repr__] = _pprint_tuple

    def _pprint_set(self, object, stream, indent, allowance, context, level):
        if not len(object):
            stream.write(repr(object))
            return
        typ = object.__class__
        if typ is set:
            stream.write('{')
            endchar = '}'
        else:
            stream.write(typ.__name__ + '({')
            endchar = '})'
            indent += len(typ.__name__) + 1
        object = sorted(object, key=_safe_key)
        self._format_items(object, stream, indent, allowance + len(endchar),
                           context, level)
        stream.write(endchar)

    _dispatch[set.__repr__] = _pprint_set
    _dispatch[frozenset.__repr__] = _pprint_set

    def _pprint_str(self, object, stream, indent, allowance, context, level):
        write = stream.write
        if not len(object):
            write(repr(object))
            return
        chunks = []
        lines = object.splitlines(True)
        if level == 1:
            indent += 1
            allowance += 1
        max_width1 = max_width = self._width - indent
        for i, line in enumerate(lines):
            rep = repr(line)
            if i == len(lines) - 1:
                max_width1 -= allowance
            if len(rep) <= max_width1:
                chunks.append(rep)
            else:
                # Lazy import to improve module import time
                import re

                # A list of alternating (non-space, space) strings
                parts = re.findall(r'\S*\s*', line)
                assert parts
                assert not parts[-1]
                parts.pop()  # drop empty last part
                max_width2 = max_width
                current = ''
                for j, part in enumerate(parts):
                    candidate = current + part
                    if j == len(parts) - 1 and i == len(lines) - 1:
                        max_width2 -= allowance
                    if len(repr(candidate)) > max_width2:
                        if current:
                            chunks.append(repr(current))
                        current = part
                    else:
                        current = candidate
                if current:
                    chunks.append(repr(current))
        if len(chunks) == 1:
            write(rep)
            return
        if level == 1:
            write('(')
        for i, rep in enumerate(chunks):
            if i > 0:
                write('\n' + ' '*indent)
            write(rep)
        if level == 1:
            write(')')

    _dispatch[str.__repr__] = _pprint_str

    def _pprint_bytes(self, object, stream, indent, allowance, context, level):
        write = stream.write
        if len(object) <= 4:
            write(repr(object))
            return
        parens = level == 1
        if parens:
            indent += 1
            allowance += 1
            write('(')
        delim = ''
        for rep in _wrap_bytes_repr(object, self._width - indent, allowance):
            write(delim)
            write(rep)
            if not delim:
                delim = '\n' + ' '*indent
        if parens:
            write(')')

    _dispatch[bytes.__repr__] = _pprint_bytes

    def _pprint_bytearray(self, object, stream, indent, allowance, context, level):
        write = stream.write
        write('bytearray(')
        self._pprint_bytes(bytes(object), stream, indent + 10,
                           allowance + 1, context, level + 1)
        write(')')

    _dispatch[bytearray.__repr__] = _pprint_bytearray

    def _pprint_mappingproxy(self, object, stream, indent, allowance, context, level):
        stream.write('mappingproxy(')
        self._format(object.copy(), stream, indent + 13, allowance + 1,
                     context, level)
        stream.write(')')

    _dispatch[_types.MappingProxyType.__repr__] = _pprint_mappingproxy

    def _pprint_simplenamespace(self, object, stream, indent, allowance, context, level):
        if type(object) is _types.SimpleNamespace:
            # The SimpleNamespace repr is "namespace" instead of the class
            # name, so we do the same here. For subclasses; use the class name.
            cls_name = 'namespace'
        else:
            cls_name = object.__class__.__name__
        indent += len(cls_name) + 1
        items = object.__dict__.items()
        stream.write(cls_name + '(')
        self._format_namespace_items(items, stream, indent, allowance, context, level)
        stream.write(')')

    _dispatch[_types.SimpleNamespace.__repr__] = _pprint_simplenamespace

    def _format_dict_items(self, items, stream, indent, allowance, context,
                           level):
        write = stream.write
        indent += self._indent_per_level
        delimnl = ',\n' + ' ' * indent
        last_index = len(items) - 1
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            rep = self._repr(key, context, level)
            write(rep)
            write(': ')
            self._format(ent, stream, indent + len(rep) + 2,
                         allowance if last else 1,
                         context, level)
            if not last:
                write(delimnl)

    def _format_namespace_items(self, items, stream, indent, allowance, context, level):
        write = stream.write
        delimnl = ',\n' + ' ' * indent
        last_index = len(items) - 1
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            write(key)
            write('=')
            if id(ent) in context:
                # Special-case representation of recursion to match standard
                # recursive dataclass repr.
                write("...")
            else:
                self._format(ent, stream, indent + len(key) + 1,
                             allowance if last else 1,
                             context, level)
            if not last:
                write(delimnl)

    def _format_items(self, items, stream, indent, allowance, context, level):
        write = stream.write
        indent += self._indent_per_level
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * ' ')
        delimnl = ',\n' + ' ' * indent
        delim = ''
        width = max_width = self._width - indent + 1
        it = iter(items)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        while not last:
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                rep = self._repr(ent, context, level)
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ', '
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            self._format(ent, stream, indent,
                         allowance if last else 1,
                         context, level)

    def _repr(self, object, context, level):
        repr, readable, recursive = self.format(object, context.copy(),
                                                self._depth, level)
        if not readable:
            self._readable = False
        if recursive:
            self._recursive = True
        return repr

    def format(self, object, context, maxlevels, level):
        """Format object for a specific context, returning a string
        and flags indicating whether the representation is 'readable'
        and whether the object represents a recursive construct.
        """
        return self._safe_repr(object, context, maxlevels, level)

    def _pprint_default_dict(self, object, stream, indent, allowance, context, level):
        if not len(object):
            stream.write(repr(object))
            return
        rdf = self._repr(object.default_factory, context, level)
        cls = object.__class__
        indent += len(cls.__name__) + 1
        stream.write('%s(%s,\n%s' % (cls.__name__, rdf, ' ' * indent))
        self._pprint_dict(object, stream, indent, allowance + 1, context, level)
        stream.write(')')

    _dispatch[_collections.defaultdict.__repr__] = _pprint_default_dict

    def _pprint_counter(self, object, stream, indent, allowance, context, level):
        if not len(object):
            stream.write(repr(object))
            return
        cls = object.__class__
        stream.write(cls.__name__ + '({')
        if self._indent_per_level > 1:
            stream.write((self._indent_per_level - 1) * ' ')
        items = object.most_common()
        self._format_dict_items(items, stream,
                                indent + len(cls.__name__) + 1, allowance + 2,
                                context, level)
        stream.write('})')

    _dispatch[_collections.Counter.__repr__] = _pprint_counter

    def _pprint_chain_map(self, object, stream, indent, allowance, context, level):
        if not len(object.maps):
            stream.write(repr(object))
            return
        cls = object.__class__
        stream.write(cls.__name__ + '(')
        indent += len(cls.__name__) + 1
        for i, m in enumerate(object.maps):
            if i == len(object.maps) - 1:
                self._format(m, stream, indent, allowance + 1, context, level)
                stream.write(')')
            else:
                self._format(m, stream, indent, 1, context, level)
                stream.write(',\n' + ' ' * indent)

    _dispatch[_collections.ChainMap.__repr__] = _pprint_chain_map

    def _pprint_deque(self, object, stream, indent, allowance, context, level):
        if not len(object):
            stream.write(repr(object))
            return
        cls = object.__class__
        stream.write(cls.__name__ + '(')
        indent += len(cls.__name__) + 1
        stream.write('[')
        if object.maxlen is None:
            self._format_items(object, stream, indent, allowance + 2,
                               context, level)
            stream.write('])')
        else:
            self._format_items(object, stream, indent, 2,
                               context, level)
            rml = self._repr(object.maxlen, context, level)
            stream.write('],\n%smaxlen=%s)' % (' ' * indent, rml))

    _dispatch[_collections.deque.__repr__] = _pprint_deque

    def _pprint_user_dict(self, object, stream, indent, allowance, context, level):
        self._format(object.data, stream, indent, allowance, context, level - 1)

    _dispatch[_collections.UserDict.__repr__] = _pprint_user_dict

    def _pprint_user_list(self, object, stream, indent, allowance, context, level):
        self._format(object.data, stream, indent, allowance, context, level - 1)

    _dispatch[_collections.UserList.__repr__] = _pprint_user_list

    def _pprint_user_string(self, object, stream, indent, allowance, context, level):
        self._format(object.data, stream, indent, allowance, context, level - 1)

    _dispatch[_collections.UserString.__repr__] = _pprint_user_string

    def _safe_repr(self, object, context, maxlevels, level):
        # Return triple (repr_string, isreadable, isrecursive).
        typ = type(object)
        if typ in _builtin_scalars:
            return repr(object), True, False

        r = getattr(typ, "__repr__", None)

        if issubclass(typ, int) and r is int.__repr__:
            if self._underscore_numbers:
                return f"{object:_d}", True, False
            else:
                return repr(object), True, False

        if issubclass(typ, dict) and r is dict.__repr__:
            if not object:
                return "{}", True, False
            objid = id(object)
            if maxlevels and level >= maxlevels:
                return "{...}", False, objid in context
            if objid in context:
                return _recursion(object), False, True
            context[objid] = 1
            readable = True
            recursive = False
            components = []
            append = components.append
            level += 1
            if self._sort_dicts:
                items = sorted(object.items(), key=_safe_tuple)
            else:
                items = object.items()
            for k, v in items:
                krepr, kreadable, krecur = self.format(
                    k, context, maxlevels, level)
                vrepr, vreadable, vrecur = self.format(
                    v, context, maxlevels, level)
                append("%s: %s" % (krepr, vrepr))
                readable = readable and kreadable and vreadable
                if krecur or vrecur:
                    recursive = True
            del context[objid]
            return "{%s}" % ", ".join(components), readable, recursive

        if (issubclass(typ, list) and r is list.__repr__) or \
           (issubclass(typ, tuple) and r is tuple.__repr__):
            if issubclass(typ, list):
                if not object:
                    return "[]", True, False
                format = "[%s]"
            elif len(object) == 1:
                format = "(%s,)"
            else:
                if not object:
                    return "()", True, False
                format = "(%s)"
            objid = id(object)
            if maxlevels and level >= maxlevels:
                return format % "...", False, objid in context
            if objid in context:
                return _recursion(object), False, True
            context[objid] = 1
            readable = True
            recursive = False
            components = []
            append = components.append
            level += 1
            for o in object:
                orepr, oreadable, orecur = self.format(
                    o, context, maxlevels, level)
                append(orepr)
                if not oreadable:
                    readable = False
                if orecur:
                    recursive = True
            del context[objid]
            return format % ", ".join(components), readable, recursive

        rep = repr(object)
        return rep, (rep and not rep.startswith('<')), False


_builtin_scalars = frozenset({str, bytes, bytearray, float, complex,
                              bool, type(None)})


def _recursion(object):
    return ("<Recursion on %s with id=%s>"
            % (type(object).__name__, id(object)))


def _wrap_bytes_repr(object, width, allowance):
    current = b''
    last = len(object) // 4 * 4
    for i in range(0, len(object), 4):
        part = object[i: i+4]
        candidate = current + part
        if i == last:
            width -= allowance
        if len(repr(candidate)) > width:
            if current:
                yield repr(current)
            current = part
        else:
            current = candidate
    if current:
        yield repr(current)

import warnings
warnings.warn(f"module {__name__!r} is deprecated",
              DeprecationWarning,
              stacklevel=2)

from re import _parser as _
globals().update({k: v for k, v in vars(_).items() if k[:2] != '__'})

#
# Class for profiling python code. rev 1.0  6/2/94
#
# Written by James Roskind
# Based on prior profile module by Sjoerd Mullender...
#   which was hacked somewhat by: Guido van Rossum

"""Class for profiling Python code."""

# Copyright Disney Enterprises, Inc.  All Rights Reserved.
# Licensed to PSF under a Contributor Agreement
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.  See the License for the specific language
# governing permissions and limitations under the License.


import importlib.machinery
import io
import sys
import time
import marshal

__all__ = ["run", "runctx", "Profile"]

# Sample timer for use with
#i_count = 0
#def integer_timer():
#       global i_count
#       i_count = i_count + 1
#       return i_count
#itimes = integer_timer # replace with C coded timer returning integers

class _Utils:
    """Support class for utility functions which are shared by
    profile.py and cProfile.py modules.
    Not supposed to be used directly.
    """

    def __init__(self, profiler):
        self.profiler = profiler

    def run(self, statement, filename, sort):
        prof = self.profiler()
        try:
            prof.run(statement)
        except SystemExit:
            pass
        finally:
            self._show(prof, filename, sort)

    def runctx(self, statement, globals, locals, filename, sort):
        prof = self.profiler()
        try:
            prof.runctx(statement, globals, locals)
        except SystemExit:
            pass
        finally:
            self._show(prof, filename, sort)

    def _show(self, prof, filename, sort):
        if filename is not None:
            prof.dump_stats(filename)
        else:
            prof.print_stats(sort)


#**************************************************************************
# The following are the static member functions for the profiler class
# Note that an instance of Profile() is *not* needed to call them.
#**************************************************************************

def run(statement, filename=None, sort=-1):
    """Run statement under profiler optionally saving results in filename

    This function takes a single argument that can be passed to the
    "exec" statement, and an optional file name.  In all cases this
    routine attempts to "exec" its first argument and gather profiling
    statistics from the execution. If no file name is present, then this
    function automatically prints a simple profiling report, sorted by the
    standard name string (file/line/function-name) that is presented in
    each line.
    """
    return _Utils(Profile).run(statement, filename, sort)

def runctx(statement, globals, locals, filename=None, sort=-1):
    """Run statement under profiler, supplying your own globals and locals,
    optionally saving results in filename.

    statement and filename have the same semantics as profile.run
    """
    return _Utils(Profile).runctx(statement, globals, locals, filename, sort)


class Profile:
    """Profiler class.

    self.cur is always a tuple.  Each such tuple corresponds to a stack
    frame that is currently active (self.cur[-2]).  The following are the
    definitions of its members.  We use this external "parallel stack" to
    avoid contaminating the program that we are profiling. (old profiler
    used to write into the frames local dictionary!!) Derived classes
    can change the definition of some entries, as long as they leave
    [-2:] intact (frame and previous tuple).  In case an internal error is
    detected, the -3 element is used as the function name.

    [ 0] = Time that needs to be charged to the parent frame's function.
           It is used so that a function call will not have to access the
           timing data for the parent frame.
    [ 1] = Total time spent in this frame's function, excluding time in
           subfunctions (this latter is tallied in cur[2]).
    [ 2] = Total time spent in subfunctions, excluding time executing the
           frame's function (this latter is tallied in cur[1]).
    [-3] = Name of the function that corresponds to this frame.
    [-2] = Actual frame that we correspond to (used to sync exception handling).
    [-1] = Our parent 6-tuple (corresponds to frame.f_back).

    Timing data for each function is stored as a 5-tuple in the dictionary
    self.timings[].  The index is always the name stored in self.cur[-3].
    The following are the definitions of the members:

    [0] = The number of times this function was called, not counting direct
          or indirect recursion,
    [1] = Number of times this function appears on the stack, minus one
    [2] = Total time spent internal to this function
    [3] = Cumulative time that this function was present on the stack.  In
          non-recursive functions, this is the total execution time from start
          to finish of each invocation of a function, including time spent in
          all subfunctions.
    [4] = A dictionary indicating for each function name, the number of times
          it was called by us.
    """

    bias = 0  # calibration constant

    def __init__(self, timer=None, bias=None):
        self.timings = {}
        self.cur = None
        self.cmd = ""
        self.c_func_name = ""

        if bias is None:
            bias = self.bias
        self.bias = bias     # Materialize in local dict for lookup speed.

        if not timer:
            self.timer = self.get_time = time.process_time
            self.dispatcher = self.trace_dispatch_i
        else:
            self.timer = timer
            t = self.timer() # test out timer function
            try:
                length = len(t)
            except TypeError:
                self.get_time = timer
                self.dispatcher = self.trace_dispatch_i
            else:
                if length == 2:
                    self.dispatcher = self.trace_dispatch
                else:
                    self.dispatcher = self.trace_dispatch_l
                # This get_time() implementation needs to be defined
                # here to capture the passed-in timer in the parameter
                # list (for performance).  Note that we can't assume
                # the timer() result contains two values in all
                # cases.
                def get_time_timer(timer=timer, sum=sum):
                    return sum(timer())
                self.get_time = get_time_timer
        self.t = self.get_time()
        self.simulate_call('profiler')

    # Heavily optimized dispatch routine for time.process_time() timer

    def trace_dispatch(self, frame, event, arg):
        timer = self.timer
        t = timer()
        t = t[0] + t[1] - self.t - self.bias

        if event == "c_call":
            self.c_func_name = arg.__name__

        if self.dispatch[event](self, frame,t):
            t = timer()
            self.t = t[0] + t[1]
        else:
            r = timer()
            self.t = r[0] + r[1] - t # put back unrecorded delta

    # Dispatch routine for best timer program (return = scalar, fastest if
    # an integer but float works too -- and time.process_time() relies on that).

    def trace_dispatch_i(self, frame, event, arg):
        timer = self.timer
        t = timer() - self.t - self.bias

        if event == "c_call":
            self.c_func_name = arg.__name__

        if self.dispatch[event](self, frame, t):
            self.t = timer()
        else:
            self.t = timer() - t  # put back unrecorded delta

    # Dispatch routine for macintosh (timer returns time in ticks of
    # 1/60th second)

    def trace_dispatch_mac(self, frame, event, arg):
        timer = self.timer
        t = timer()/60.0 - self.t - self.bias

        if event == "c_call":
            self.c_func_name = arg.__name__

        if self.dispatch[event](self, frame, t):
            self.t = timer()/60.0
        else:
            self.t = timer()/60.0 - t  # put back unrecorded delta

    # SLOW generic dispatch routine for timer returning lists of numbers

    def trace_dispatch_l(self, frame, event, arg):
        get_time = self.get_time
        t = get_time() - self.t - self.bias

        if event == "c_call":
            self.c_func_name = arg.__name__

        if self.dispatch[event](self, frame, t):
            self.t = get_time()
        else:
            self.t = get_time() - t # put back unrecorded delta

    # In the event handlers, the first 3 elements of self.cur are unpacked
    # into vrbls w/ 3-letter names.  The last two characters are meant to be
    # mnemonic:
    #     _pt  self.cur[0] "parent time"   time to be charged to parent frame
    #     _it  self.cur[1] "internal time" time spent directly in the function
    #     _et  self.cur[2] "external time" time spent in subfunctions

    def trace_dispatch_exception(self, frame, t):
        rpt, rit, ret, rfn, rframe, rcur = self.cur
        if (rframe is not frame) and rcur:
            return self.trace_dispatch_return(rframe, t)
        self.cur = rpt, rit+t, ret, rfn, rframe, rcur
        return 1


    def trace_dispatch_call(self, frame, t):
        if self.cur and frame.f_back is not self.cur[-2]:
            rpt, rit, ret, rfn, rframe, rcur = self.cur
            if not isinstance(rframe, Profile.fake_frame):
                assert rframe.f_back is frame.f_back, ("Bad call", rfn,
                                                       rframe, rframe.f_back,
                                                       frame, frame.f_back)
                self.trace_dispatch_return(rframe, 0)
                assert (self.cur is None or \
                        frame.f_back is self.cur[-2]), ("Bad call",
                                                        self.cur[-3])
        fcode = frame.f_code
        fn = (fcode.co_filename, fcode.co_firstlineno, fcode.co_name)
        self.cur = (t, 0, 0, fn, frame, self.cur)
        timings = self.timings
        if fn in timings:
            cc, ns, tt, ct, callers = timings[fn]
            timings[fn] = cc, ns + 1, tt, ct, callers
        else:
            timings[fn] = 0, 0, 0, 0, {}
        return 1

    def trace_dispatch_c_call (self, frame, t):
        fn = ("", 0, self.c_func_name)
        self.cur = (t, 0, 0, fn, frame, self.cur)
        timings = self.timings
        if fn in timings:
            cc, ns, tt, ct, callers = timings[fn]
            timings[fn] = cc, ns+1, tt, ct, callers
        else:
            timings[fn] = 0, 0, 0, 0, {}
        return 1

    def trace_dispatch_return(self, frame, t):
        if frame is not self.cur[-2]:
            assert frame is self.cur[-2].f_back, ("Bad return", self.cur[-3])
            self.trace_dispatch_return(self.cur[-2], 0)

        # Prefix "r" means part of the Returning or exiting frame.
        # Prefix "p" means part of the Previous or Parent or older frame.

        rpt, rit, ret, rfn, frame, rcur = self.cur
        rit = rit + t
        frame_total = rit + ret

        ppt, pit, pet, pfn, pframe, pcur = rcur
        self.cur = ppt, pit + rpt, pet + frame_total, pfn, pframe, pcur

        timings = self.timings
        cc, ns, tt, ct, callers = timings[rfn]
        if not ns:
            # This is the only occurrence of the function on the stack.
            # Else this is a (directly or indirectly) recursive call, and
            # its cumulative time will get updated when the topmost call to
            # it returns.
            ct = ct + frame_total
            cc = cc + 1

        if pfn in callers:
            callers[pfn] = callers[pfn] + 1  # hack: gather more
            # stats such as the amount of time added to ct courtesy
            # of this specific call, and the contribution to cc
            # courtesy of this call.
        else:
            callers[pfn] = 1

        timings[rfn] = cc, ns - 1, tt + rit, ct, callers

        return 1


    dispatch = {
        "call": trace_dispatch_call,
        "exception": trace_dispatch_exception,
        "return": trace_dispatch_return,
        "c_call": trace_dispatch_c_call,
        "c_exception": trace_dispatch_return,  # the C function returned
        "c_return": trace_dispatch_return,
        }


    # The next few functions play with self.cmd. By carefully preloading
    # our parallel stack, we can force the profiled result to include
    # an arbitrary string as the name of the calling function.
    # We use self.cmd as that string, and the resulting stats look
    # very nice :-).

    def set_cmd(self, cmd):
        if self.cur[-1]: return   # already set
        self.cmd = cmd
        self.simulate_call(cmd)

    class fake_code:
        def __init__(self, filename, line, name):
            self.co_filename = filename
            self.co_line = line
            self.co_name = name
            self.co_firstlineno = 0

        def __repr__(self):
            return repr((self.co_filename, self.co_line, self.co_name))

    class fake_frame:
        def __init__(self, code, prior):
            self.f_code = code
            self.f_back = prior

    def simulate_call(self, name):
        code = self.fake_code('profile', 0, name)
        if self.cur:
            pframe = self.cur[-2]
        else:
            pframe = None
        frame = self.fake_frame(code, pframe)
        self.dispatch['call'](self, frame, 0)

    # collect stats from pending stack, including getting final
    # timings for self.cmd frame.

    def simulate_cmd_complete(self):
        get_time = self.get_time
        t = get_time() - self.t
        while self.cur[-1]:
            # We *can* cause assertion errors here if
            # dispatch_trace_return checks for a frame match!
            self.dispatch['return'](self, self.cur[-2], t)
            t = 0
        self.t = get_time() - t


    def print_stats(self, sort=-1):
        import pstats
        if not isinstance(sort, tuple):
            sort = (sort,)
        pstats.Stats(self).strip_dirs().sort_stats(*sort).print_stats()

    def dump_stats(self, file):
        with open(file, 'wb') as f:
            self.create_stats()
            marshal.dump(self.stats, f)

    def create_stats(self):
        self.simulate_cmd_complete()
        self.snapshot_stats()

    def snapshot_stats(self):
        self.stats = {}
        for func, (cc, ns, tt, ct, callers) in self.timings.items():
            callers = callers.copy()
            nc = 0
            for callcnt in callers.values():
                nc += callcnt
            self.stats[func] = cc, nc, tt, ct, callers


    # The following two methods can be called by clients to use
    # a profiler to profile a statement, given as a string.

    def run(self, cmd):
        import __main__
        dict = __main__.__dict__
        return self.runctx(cmd, dict, dict)

    def runctx(self, cmd, globals, locals):
        self.set_cmd(cmd)
        sys.setprofile(self.dispatcher)
        try:
            exec(cmd, globals, locals)
        finally:
            sys.setprofile(None)
        return self

    # This method is more useful to profile a single function call.
    def runcall(self, func, /, *args, **kw):
        self.set_cmd(repr(func))
        sys.setprofile(self.dispatcher)
        try:
            return func(*args, **kw)
        finally:
            sys.setprofile(None)


    #******************************************************************
    # The following calculates the overhead for using a profiler.  The
    # problem is that it takes a fair amount of time for the profiler
    # to stop the stopwatch (from the time it receives an event).
    # Similarly, there is a delay from the time that the profiler
    # re-starts the stopwatch before the user's code really gets to
    # continue.  The following code tries to measure the difference on
    # a per-event basis.
    #
    # Note that this difference is only significant if there are a lot of
    # events, and relatively little user code per event.  For example,
    # code with small functions will typically benefit from having the
    # profiler calibrated for the current platform.  This *could* be
    # done on the fly during init() time, but it is not worth the
    # effort.  Also note that if too large a value specified, then
    # execution time on some functions will actually appear as a
    # negative number.  It is *normal* for some functions (with very
    # low call counts) to have such negative stats, even if the
    # calibration figure is "correct."
    #
    # One alternative to profile-time calibration adjustments (i.e.,
    # adding in the magic little delta during each event) is to track
    # more carefully the number of events (and cumulatively, the number
    # of events during sub functions) that are seen.  If this were
    # done, then the arithmetic could be done after the fact (i.e., at
    # display time).  Currently, we track only call/return events.
    # These values can be deduced by examining the callees and callers
    # vectors for each functions.  Hence we *can* almost correct the
    # internal time figure at print time (note that we currently don't
    # track exception event processing counts).  Unfortunately, there
    # is currently no similar information for cumulative sub-function
    # time.  It would not be hard to "get all this info" at profiler
    # time.  Specifically, we would have to extend the tuples to keep
    # counts of this in each frame, and then extend the defs of timing
    # tuples to include the significant two figures. I'm a bit fearful
    # that this additional feature will slow the heavily optimized
    # event/time ratio (i.e., the profiler would run slower, fur a very
    # low "value added" feature.)
    #**************************************************************

    def calibrate(self, m, verbose=0):
        if self.__class__ is not Profile:
            raise TypeError("Subclasses must override .calibrate().")

        saved_bias = self.bias
        self.bias = 0
        try:
            return self._calibrate_inner(m, verbose)
        finally:
            self.bias = saved_bias

    def _calibrate_inner(self, m, verbose):
        get_time = self.get_time

        # Set up a test case to be run with and without profiling.  Include
        # lots of calls, because we're trying to quantify stopwatch overhead.
        # Do not raise any exceptions, though, because we want to know
        # exactly how many profile events are generated (one call event, +
        # one return event, per Python-level call).

        def f1(n):
            for i in range(n):
                x = 1

        def f(m, f1=f1):
            for i in range(m):
                f1(100)

        f(m)    # warm up the cache

        # elapsed_noprofile <- time f(m) takes without profiling.
        t0 = get_time()
        f(m)
        t1 = get_time()
        elapsed_noprofile = t1 - t0
        if verbose:
            print("elapsed time without profiling =", elapsed_noprofile)

        # elapsed_profile <- time f(m) takes with profiling.  The difference
        # is profiling overhead, only some of which the profiler subtracts
        # out on its own.
        p = Profile()
        t0 = get_time()
        p.runctx('f(m)', globals(), locals())
        t1 = get_time()
        elapsed_profile = t1 - t0
        if verbose:
            print("elapsed time with profiling =", elapsed_profile)

        # reported_time <- "CPU seconds" the profiler charged to f and f1.
        total_calls = 0.0
        reported_time = 0.0
        for (filename, line, funcname), (cc, ns, tt, ct, callers) in \
                p.timings.items():
            if funcname in ("f", "f1"):
                total_calls += cc
                reported_time += tt

        if verbose:
            print("'CPU seconds' profiler reported =", reported_time)
            print("total # calls =", total_calls)
        if total_calls != m + 1:
            raise ValueError("internal error: total calls = %d" % total_calls)

        # reported_time - elapsed_noprofile = overhead the profiler wasn't
        # able to measure.  Divide by twice the number of calls (since there
        # are two profiler events per call in this test) to get the hidden
        # overhead per event.
        mean = (reported_time - elapsed_noprofile) / 2.0 / total_calls
        if verbose:
            print("mean stopwatch overhead per profile event =", mean)
        return mean

#****************************************************************************

def main():
    import os
    from optparse import OptionParser

    usage = "profile.py [-o output_file_path] [-s sort] [-m module | scriptfile] [arg] ..."
    parser = OptionParser(usage=usage)
    parser.allow_interspersed_args = False
    parser.add_option('-o', '--outfile', dest="outfile",
        help="Save stats to <outfile>", default=None)
    parser.add_option('-m', dest="module", action="store_true",
        help="Profile a library module.", default=False)
    parser.add_option('-s', '--sort', dest="sort",
        help="Sort order when printing to stdout, based on pstats.Stats class",
        default=-1)

    if not sys.argv[1:]:
        parser.print_usage()
        sys.exit(2)

    (options, args) = parser.parse_args()
    sys.argv[:] = args

    # The script that we're profiling may chdir, so capture the absolute path
    # to the output file at startup.
    if options.outfile is not None:
        options.outfile = os.path.abspath(options.outfile)

    if len(args) > 0:
        if options.module:
            import runpy
            code = "run_module(modname, run_name='__main__')"
            globs = {
                'run_module': runpy.run_module,
                'modname': args[0]
            }
        else:
            progname = args[0]
            sys.path.insert(0, os.path.dirname(progname))
            with io.open_code(progname) as fp:
                code = compile(fp.read(), progname, 'exec')
            spec = importlib.machinery.ModuleSpec(name='__main__', loader=None,
                                                  origin=progname)
            globs = {
                '__spec__': spec,
                '__file__': spec.origin,
                '__name__': spec.name,
                '__package__': None,
                '__cached__': None,
            }
        try:
            runctx(code, globs, None, options.outfile, options.sort)
        except BrokenPipeError as exc:
            # Prevent "Exception ignored" during interpreter shutdown.
            sys.stdout = None
            sys.exit(exc.errno)
    else:
        parser.print_usage()
    return parser

# When invoked as main program, invoke the profiler on a script
if __name__ == '__main__':
    main()

"""Routine to "compile" a .py file to a .pyc file.

This module has intimate knowledge of the format of .pyc files.
"""

import enum
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import os
import os.path
import sys
import traceback

__all__ = ["compile", "main", "PyCompileError", "PycInvalidationMode"]


class PyCompileError(Exception):
    """Exception raised when an error occurs while attempting to
    compile the file.

    To raise this exception, use

        raise PyCompileError(exc_type,exc_value,file[,msg])

    where

        exc_type:   exception type to be used in error message
                    type name can be accesses as class variable
                    'exc_type_name'

        exc_value:  exception value to be used in error message
                    can be accesses as class variable 'exc_value'

        file:       name of file being compiled to be used in error message
                    can be accesses as class variable 'file'

        msg:        string message to be written as error message
                    If no value is given, a default exception message will be
                    given, consistent with 'standard' py_compile output.
                    message (or default) can be accesses as class variable
                    'msg'

    """

    def __init__(self, exc_type, exc_value, file, msg=''):
        exc_type_name = exc_type.__name__
        if exc_type is SyntaxError:
            tbtext = ''.join(traceback.format_exception_only(
                exc_type, exc_value))
            errmsg = tbtext.replace('File "<string>"', 'File "%s"' % file)
        else:
            errmsg = "Sorry: %s: %s" % (exc_type_name,exc_value)

        Exception.__init__(self,msg or errmsg,exc_type_name,exc_value,file)

        self.exc_type_name = exc_type_name
        self.exc_value = exc_value
        self.file = file
        self.msg = msg or errmsg

    def __str__(self):
        return self.msg


class PycInvalidationMode(enum.Enum):
    TIMESTAMP = 1
    CHECKED_HASH = 2
    UNCHECKED_HASH = 3


def _get_default_invalidation_mode():
    if os.environ.get('SOURCE_DATE_EPOCH'):
        return PycInvalidationMode.CHECKED_HASH
    else:
        return PycInvalidationMode.TIMESTAMP


def compile(file, cfile=None, dfile=None, doraise=False, optimize=-1,
            invalidation_mode=None, quiet=0):
    """Byte-compile one Python source file to Python bytecode.

    :param file: The source file name.
    :param cfile: The target byte compiled file name.  When not given, this
        defaults to the PEP 3147/PEP 488 location.
    :param dfile: Purported file name, i.e. the file name that shows up in
        error messages.  Defaults to the source file name.
    :param doraise: Flag indicating whether or not an exception should be
        raised when a compile error is found.  If an exception occurs and this
        flag is set to False, a string indicating the nature of the exception
        will be printed, and the function will return to the caller. If an
        exception occurs and this flag is set to True, a PyCompileError
        exception will be raised.
    :param optimize: The optimization level for the compiler.  Valid values
        are -1, 0, 1 and 2.  A value of -1 means to use the optimization
        level of the current interpreter, as given by -O command line options.
    :param invalidation_mode:
    :param quiet: Return full output with False or 0, errors only with 1,
        and no output with 2.

    :return: Path to the resulting byte compiled file.

    Note that it isn't necessary to byte-compile Python modules for
    execution efficiency -- Python itself byte-compiles a module when
    it is loaded, and if it can, writes out the bytecode to the
    corresponding .pyc file.

    However, if a Python installation is shared between users, it is a
    good idea to byte-compile all modules upon installation, since
    other users may not be able to write in the source directories,
    and thus they won't be able to write the .pyc file, and then
    they would be byte-compiling every module each time it is loaded.
    This can slow down program start-up considerably.

    See compileall.py for a script/module that uses this module to
    byte-compile all installed files (or all files in selected
    directories).

    Do note that FileExistsError is raised if cfile ends up pointing at a
    non-regular file or symlink. Because the compilation uses a file renaming,
    the resulting file would be regular and thus not the same type of file as
    it was previously.
    """
    if invalidation_mode is None:
        invalidation_mode = _get_default_invalidation_mode()
    if cfile is None:
        if optimize >= 0:
            optimization = optimize if optimize >= 1 else ''
            cfile = importlib.util.cache_from_source(file,
                                                     optimization=optimization)
        else:
            cfile = importlib.util.cache_from_source(file)
    if os.path.islink(cfile):
        msg = ('{} is a symlink and will be changed into a regular file if '
               'import writes a byte-compiled file to it')
        raise FileExistsError(msg.format(cfile))
    elif os.path.exists(cfile) and not os.path.isfile(cfile):
        msg = ('{} is a non-regular file and will be changed into a regular '
               'one if import writes a byte-compiled file to it')
        raise FileExistsError(msg.format(cfile))
    loader = importlib.machinery.SourceFileLoader('<py_compile>', file)
    source_bytes = loader.get_data(file)
    try:
        code = loader.source_to_code(source_bytes, dfile or file,
                                     _optimize=optimize)
    except Exception as err:
        py_exc = PyCompileError(err.__class__, err, dfile or file)
        if quiet < 2:
            if doraise:
                raise py_exc
            else:
                sys.stderr.write(py_exc.msg + '\n')
        return
    try:
        dirname = os.path.dirname(cfile)
        if dirname:
            os.makedirs(dirname)
    except FileExistsError:
        pass
    if invalidation_mode == PycInvalidationMode.TIMESTAMP:
        source_stats = loader.path_stats(file)
        bytecode = importlib._bootstrap_external._code_to_timestamp_pyc(
            code, source_stats['mtime'], source_stats['size'])
    else:
        source_hash = importlib.util.source_hash(source_bytes)
        bytecode = importlib._bootstrap_external._code_to_hash_pyc(
            code,
            source_hash,
            (invalidation_mode == PycInvalidationMode.CHECKED_HASH),
        )
    mode = importlib._bootstrap_external._calc_mode(file)
    importlib._bootstrap_external._write_atomic(cfile, bytecode, mode)
    return cfile


def main():
    import argparse

    description = 'A simple command-line interface for py_compile module.'
    parser = argparse.ArgumentParser(description=description, color=True)
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress error output',
    )
    parser.add_argument(
        'filenames',
        nargs='+',
        help='Files to compile',
    )
    args = parser.parse_args()
    if args.filenames == ['-']:
        filenames = [filename.rstrip('\n') for filename in sys.stdin.readlines()]
    else:
        filenames = args.filenames
    for filename in filenames:
        try:
            compile(filename, doraise=True)
        except PyCompileError as error:
            if args.quiet:
                parser.exit(1)
            else:
                parser.exit(1, error.msg)
        except OSError as error:
            if args.quiet:
                parser.exit(1)
            else:
                parser.exit(1, str(error))


if __name__ == "__main__":
    main()

"""HMAC (Keyed-Hashing for Message Authentication) module.

Implements the HMAC algorithm as described by RFC 2104.
"""

try:
    import _hashlib as _hashopenssl
except ImportError:
    _hashopenssl = None
    _functype = None
    from _operator import _compare_digest as compare_digest
else:
    compare_digest = _hashopenssl.compare_digest
    _functype = type(_hashopenssl.openssl_sha256)  # builtin type

try:
    import _hmac
except ImportError:
    _hmac = None

trans_5C = bytes((x ^ 0x5C) for x in range(256))
trans_36 = bytes((x ^ 0x36) for x in range(256))

# The size of the digests returned by HMAC depends on the underlying
# hashing module used.  Use digest_size from the instance of HMAC instead.
digest_size = None


def _get_digest_constructor(digest_like):
    if callable(digest_like):
        return digest_like
    if isinstance(digest_like, str):
        def digest_wrapper(d=b''):
            import hashlib
            return hashlib.new(digest_like, d)
    else:
        def digest_wrapper(d=b''):
            return digest_like.new(d)
    return digest_wrapper


class HMAC:
    """RFC 2104 HMAC class.  Also complies with RFC 4231.

    This supports the API for Cryptographic Hash Functions (PEP 247).
    """

    # Note: self.blocksize is the default blocksize; self.block_size
    # is effective block size as well as the public API attribute.
    blocksize = 64  # 512-bit HMAC; can be changed in subclasses.

    __slots__ = (
        "_hmac", "_inner", "_outer", "block_size", "digest_size"
    )

    def __init__(self, key, msg=None, digestmod=''):
        """Create a new HMAC object.

        key: bytes or buffer, key for the keyed hash object.
        msg: bytes or buffer, Initial input for the hash or None.
        digestmod: A hash name suitable for hashlib.new(). *OR*
                   A hashlib constructor returning a new hash object. *OR*
                   A module supporting PEP 247.

                   Required as of 3.8, despite its position after the optional
                   msg argument.  Passing it as a keyword argument is
                   recommended, though not required for legacy API reasons.
        """

        if not isinstance(key, (bytes, bytearray)):
            raise TypeError(f"key: expected bytes or bytearray, "
                            f"but got {type(key).__name__!r}")

        if not digestmod:
            raise TypeError("Missing required argument 'digestmod'.")

        self.__init(key, msg, digestmod)

    def __init(self, key, msg, digestmod):
        if _hashopenssl and isinstance(digestmod, (str, _functype)):
            try:
                self._init_openssl_hmac(key, msg, digestmod)
                return
            except _hashopenssl.UnsupportedDigestmodError:  # pragma: no cover
                pass
        if _hmac and isinstance(digestmod, str):
            try:
                self._init_builtin_hmac(key, msg, digestmod)
                return
            except _hmac.UnknownHashError:  # pragma: no cover
                pass
        self._init_old(key, msg, digestmod)

    def _init_openssl_hmac(self, key, msg, digestmod):
        self._hmac = _hashopenssl.hmac_new(key, msg, digestmod=digestmod)
        self._inner = self._outer = None  # because the slots are defined
        self.digest_size = self._hmac.digest_size
        self.block_size = self._hmac.block_size

    _init_hmac = _init_openssl_hmac  # for backward compatibility (if any)

    def _init_builtin_hmac(self, key, msg, digestmod):
        self._hmac = _hmac.new(key, msg, digestmod=digestmod)
        self._inner = self._outer = None  # because the slots are defined
        self.digest_size = self._hmac.digest_size
        self.block_size = self._hmac.block_size

    def _init_old(self, key, msg, digestmod):
        import warnings

        digest_cons = _get_digest_constructor(digestmod)

        self._hmac = None
        self._outer = digest_cons()
        self._inner = digest_cons()
        self.digest_size = self._inner.digest_size

        if hasattr(self._inner, 'block_size'):
            blocksize = self._inner.block_size
            if blocksize < 16:
                warnings.warn(f"block_size of {blocksize} seems too small; "
                              f"using our default of {self.blocksize}.",
                              RuntimeWarning, 2)
                blocksize = self.blocksize  # pragma: no cover
        else:
            warnings.warn("No block_size attribute on given digest object; "
                          f"Assuming {self.blocksize}.",
                          RuntimeWarning, 2)
            blocksize = self.blocksize  # pragma: no cover

        if len(key) > blocksize:
            key = digest_cons(key).digest()

        self.block_size = blocksize

        key = key.ljust(blocksize, b'\0')
        self._outer.update(key.translate(trans_5C))
        self._inner.update(key.translate(trans_36))
        if msg is not None:
            self.update(msg)

    @property
    def name(self):
        if self._hmac:
            return self._hmac.name
        else:
            return f"hmac-{self._inner.name}"

    def update(self, msg):
        """Feed data from msg into this hashing object."""
        inst = self._hmac or self._inner
        inst.update(msg)

    def copy(self):
        """Return a separate copy of this hashing object.

        An update to this copy won't affect the original object.
        """
        # Call __new__ directly to avoid the expensive __init__.
        other = self.__class__.__new__(self.__class__)
        other.digest_size = self.digest_size
        other.block_size = self.block_size
        if self._hmac:
            other._hmac = self._hmac.copy()
            other._inner = other._outer = None
        else:
            other._hmac = None
            other._inner = self._inner.copy()
            other._outer = self._outer.copy()
        return other

    def _current(self):
        """Return a hash object for the current state.

        To be used only internally with digest() and hexdigest().
        """
        if self._hmac:
            return self._hmac
        else:
            h = self._outer.copy()
            h.update(self._inner.digest())
            return h

    def digest(self):
        """Return the hash value of this hashing object.

        This returns the hmac value as bytes.  The object is
        not altered in any way by this function; you can continue
        updating the object after calling this function.
        """
        h = self._current()
        return h.digest()

    def hexdigest(self):
        """Like digest(), but returns a string of hexadecimal digits instead.
        """
        h = self._current()
        return h.hexdigest()


def new(key, msg=None, digestmod=''):
    """Create a new hashing object and return it.

    key: bytes or buffer, The starting key for the hash.
    msg: bytes or buffer, Initial input for the hash, or None.
    digestmod: A hash name suitable for hashlib.new(). *OR*
               A hashlib constructor returning a new hash object. *OR*
               A module supporting PEP 247.

               Required as of 3.8, despite its position after the optional
               msg argument.  Passing it as a keyword argument is
               recommended, though not required for legacy API reasons.

    You can now feed arbitrary bytes into the object using its update()
    method, and can ask for the hash value at any time by calling its digest()
    or hexdigest() methods.
    """
    return HMAC(key, msg, digestmod)


def digest(key, msg, digest):
    """Fast inline implementation of HMAC.

    key: bytes or buffer, The key for the keyed hash object.
    msg: bytes or buffer, Input message.
    digest: A hash name suitable for hashlib.new() for best performance. *OR*
            A hashlib constructor returning a new hash object. *OR*
            A module supporting PEP 247.
    """
    if _hashopenssl and isinstance(digest, (str, _functype)):
        try:
            return _hashopenssl.hmac_digest(key, msg, digest)
        except OverflowError:
            # OpenSSL's HMAC limits the size of the key to INT_MAX.
            # Instead of falling back to HACL* implementation which
            # may still not be supported due to a too large key, we
            # directly switch to the pure Python fallback instead
            # even if we could have used streaming HMAC for small keys
            # but large messages.
            return _compute_digest_fallback(key, msg, digest)
        except _hashopenssl.UnsupportedDigestmodError:
            pass

    if _hmac and isinstance(digest, str):
        try:
            return _hmac.compute_digest(key, msg, digest)
        except (OverflowError, _hmac.UnknownHashError):
            # HACL* HMAC limits the size of the key to UINT32_MAX
            # so we fallback to the pure Python implementation even
            # if streaming HMAC may have been used for small keys
            # and large messages.
            pass

    return _compute_digest_fallback(key, msg, digest)


def _compute_digest_fallback(key, msg, digest):
    digest_cons = _get_digest_constructor(digest)
    inner = digest_cons()
    outer = digest_cons()
    blocksize = getattr(inner, 'block_size', 64)
    if len(key) > blocksize:
        key = digest_cons(key).digest()
    key = key.ljust(blocksize, b'\0')
    inner.update(key.translate(trans_36))
    outer.update(key.translate(trans_5C))
    inner.update(msg)
    outer.update(inner.digest())
    return outer.digest()

initialized = True

class TestFrozenUtf8_1:
    """\u00b6"""

class TestFrozenUtf8_2:
    """\u03c0"""

class TestFrozenUtf8_4:
    """\U0001f600"""

def main():
    print("Hello world!")

if __name__ == '__main__':
    main()

r"""plistlib.py -- a tool to generate and parse MacOSX .plist files.

The property list (.plist) file format is a simple XML pickle supporting
basic object types, like dictionaries, lists, numbers and strings.
Usually the top level object is a dictionary.

To write out a plist file, use the dump(value, file)
function. 'value' is the top level object, 'file' is
a (writable) file object.

To parse a plist from a file, use the load(file) function,
with a (readable) file object as the only argument. It
returns the top level object (again, usually a dictionary).

To work with plist data in bytes objects, you can use loads()
and dumps().

Values can be strings, integers, floats, booleans, tuples, lists,
dictionaries (but only with string keys), Data, bytes, bytearray, or
datetime.datetime objects.

Generate Plist example:

    import datetime as dt
    import plistlib

    pl = dict(
        aString = "Doodah",
        aList = ["A", "B", 12, 32.1, [1, 2, 3]],
        aFloat = 0.1,
        anInt = 728,
        aDict = dict(
            anotherString = "<hello & hi there!>",
            aThirdString = "M\xe4ssig, Ma\xdf",
            aTrueValue = True,
            aFalseValue = False,
        ),
        someData = b"<binary gunk>",
        someMoreData = b"<lots of binary gunk>" * 10,
        aDate = dt.datetime.now()
    )
    print(plistlib.dumps(pl).decode())

Parse Plist example:

    import plistlib

    plist = b'''<plist version="1.0">
    <dict>
        <key>foo</key>
        <string>bar</string>
    </dict>
    </plist>'''
    pl = plistlib.loads(plist)
    print(pl["foo"])
"""
__all__ = [
    "InvalidFileException", "FMT_XML", "FMT_BINARY", "load", "dump", "loads", "dumps", "UID"
]

import binascii
import codecs
import datetime
import enum
from io import BytesIO
import itertools
import os
import re
import struct
from xml.parsers.expat import ParserCreate


PlistFormat = enum.Enum('PlistFormat', 'FMT_XML FMT_BINARY', module=__name__)
globals().update(PlistFormat.__members__)

# Data larger than this will be read in chunks, to prevent extreme
# overallocation.
_MIN_READ_BUF_SIZE = 1 << 20

class UID:
    def __init__(self, data):
        if not isinstance(data, int):
            raise TypeError("data must be an int")
        if data >= 1 << 64:
            raise ValueError("UIDs cannot be >= 2**64")
        if data < 0:
            raise ValueError("UIDs must be positive")
        self.data = data

    def __index__(self):
        return self.data

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self.data))

    def __reduce__(self):
        return self.__class__, (self.data,)

    def __eq__(self, other):
        if not isinstance(other, UID):
            return NotImplemented
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

#
# XML support
#


# XML 'header'
PLISTHEADER = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
"""


# Regex to find any control chars, except for \t \n and \r
_controlCharPat = re.compile(
    r"[\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f"
    r"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f]")

def _encode_base64(s, maxlinelength=76):
    # copied from base64.encodebytes(), with added maxlinelength argument
    maxbinsize = (maxlinelength//4)*3
    pieces = []
    for i in range(0, len(s), maxbinsize):
        chunk = s[i : i + maxbinsize]
        pieces.append(binascii.b2a_base64(chunk))
    return b''.join(pieces)

def _decode_base64(s):
    if isinstance(s, str):
        return binascii.a2b_base64(s.encode("utf-8"))

    else:
        return binascii.a2b_base64(s)

# Contents should conform to a subset of ISO 8601
# (in particular, YYYY '-' MM '-' DD 'T' HH ':' MM ':' SS 'Z'.  Smaller units
# may be omitted with #  a loss of precision)
_dateParser = re.compile(r"(?P<year>\d\d\d\d)(?:-(?P<month>\d\d)(?:-(?P<day>\d\d)(?:T(?P<hour>\d\d)(?::(?P<minute>\d\d)(?::(?P<second>\d\d))?)?)?)?)?Z", re.ASCII)


def _date_from_string(s, aware_datetime):
    order = ('year', 'month', 'day', 'hour', 'minute', 'second')
    gd = _dateParser.match(s).groupdict()
    lst = []
    for key in order:
        val = gd[key]
        if val is None:
            break
        lst.append(int(val))
    if aware_datetime:
        return datetime.datetime(*lst, tzinfo=datetime.UTC)
    return datetime.datetime(*lst)


def _date_to_string(d, aware_datetime):
    if aware_datetime:
        d = d.astimezone(datetime.UTC)
    return '%04d-%02d-%02dT%02d:%02d:%02dZ' % (
        d.year, d.month, d.day,
        d.hour, d.minute, d.second
    )

def _escape(text):
    m = _controlCharPat.search(text)
    if m is not None:
        raise ValueError("strings can't contain control characters; "
                         "use bytes instead")
    text = text.replace("\r\n", "\n")       # convert DOS line endings
    text = text.replace("\r", "\n")         # convert Mac line endings
    text = text.replace("&", "&amp;")       # escape '&'
    text = text.replace("<", "&lt;")        # escape '<'
    text = text.replace(">", "&gt;")        # escape '>'
    return text

class _PlistParser:
    def __init__(self, dict_type, aware_datetime=False):
        self.stack = []
        self.current_key = None
        self.root = None
        self._dict_type = dict_type
        self._aware_datetime = aware_datetime

    def parse(self, fileobj):
        self.parser = ParserCreate()
        self.parser.StartElementHandler = self.handle_begin_element
        self.parser.EndElementHandler = self.handle_end_element
        self.parser.CharacterDataHandler = self.handle_data
        self.parser.EntityDeclHandler = self.handle_entity_decl
        self.parser.ParseFile(fileobj)
        return self.root

    def handle_entity_decl(self, entity_name, is_parameter_entity, value, base, system_id, public_id, notation_name):
        # Reject plist files with entity declarations to avoid XML vulnerabilities in expat.
        # Regular plist files don't contain those declarations, and Apple's plutil tool does not
        # accept them either.
        raise InvalidFileException("XML entity declarations are not supported in plist files")

    def handle_begin_element(self, element, attrs):
        self.data = []
        handler = getattr(self, "begin_" + element, None)
        if handler is not None:
            handler(attrs)

    def handle_end_element(self, element):
        handler = getattr(self, "end_" + element, None)
        if handler is not None:
            handler()

    def handle_data(self, data):
        self.data.append(data)

    def add_object(self, value):
        if self.current_key is not None:
            if not isinstance(self.stack[-1], dict):
                raise ValueError("unexpected element at line %d" %
                                 self.parser.CurrentLineNumber)
            self.stack[-1][self.current_key] = value
            self.current_key = None
        elif not self.stack:
            # this is the root object
            self.root = value
        else:
            if not isinstance(self.stack[-1], list):
                raise ValueError("unexpected element at line %d" %
                                 self.parser.CurrentLineNumber)
            self.stack[-1].append(value)

    def get_data(self):
        data = ''.join(self.data)
        self.data = []
        return data

    # element handlers

    def begin_dict(self, attrs):
        d = self._dict_type()
        self.add_object(d)
        self.stack.append(d)

    def end_dict(self):
        if self.current_key:
            raise ValueError("missing value for key '%s' at line %d" %
                             (self.current_key,self.parser.CurrentLineNumber))
        self.stack.pop()

    def end_key(self):
        if self.current_key or not isinstance(self.stack[-1], dict):
            raise ValueError("unexpected key at line %d" %
                             self.parser.CurrentLineNumber)
        self.current_key = self.get_data()

    def begin_array(self, attrs):
        a = []
        self.add_object(a)
        self.stack.append(a)

    def end_array(self):
        self.stack.pop()

    def end_true(self):
        self.add_object(True)

    def end_false(self):
        self.add_object(False)

    def end_integer(self):
        raw = self.get_data()
        if raw.startswith('0x') or raw.startswith('0X'):
            self.add_object(int(raw, 16))
        else:
            self.add_object(int(raw))

    def end_real(self):
        self.add_object(float(self.get_data()))

    def end_string(self):
        self.add_object(self.get_data())

    def end_data(self):
        self.add_object(_decode_base64(self.get_data()))

    def end_date(self):
        self.add_object(_date_from_string(self.get_data(),
                                          aware_datetime=self._aware_datetime))


class _DumbXMLWriter:
    def __init__(self, file, indent_level=0, indent="\t"):
        self.file = file
        self.stack = []
        self._indent_level = indent_level
        self.indent = indent

    def begin_element(self, element):
        self.stack.append(element)
        self.writeln("<%s>" % element)
        self._indent_level += 1

    def end_element(self, element):
        assert self._indent_level > 0
        assert self.stack.pop() == element
        self._indent_level -= 1
        self.writeln("</%s>" % element)

    def simple_element(self, element, value=None):
        if value is not None:
            value = _escape(value)
            self.writeln("<%s>%s</%s>" % (element, value, element))

        else:
            self.writeln("<%s/>" % element)

    def writeln(self, line):
        if line:
            # plist has fixed encoding of utf-8

            # XXX: is this test needed?
            if isinstance(line, str):
                line = line.encode('utf-8')
            self.file.write(self._indent_level * self.indent)
            self.file.write(line)
        self.file.write(b'\n')


class _PlistWriter(_DumbXMLWriter):
    def __init__(
            self, file, indent_level=0, indent=b"\t", writeHeader=1,
            sort_keys=True, skipkeys=False, aware_datetime=False):

        if writeHeader:
            file.write(PLISTHEADER)
        _DumbXMLWriter.__init__(self, file, indent_level, indent)
        self._sort_keys = sort_keys
        self._skipkeys = skipkeys
        self._aware_datetime = aware_datetime

    def write(self, value):
        self.writeln("<plist version=\"1.0\">")
        self.write_value(value)
        self.writeln("</plist>")

    def write_value(self, value):
        if isinstance(value, str):
            self.simple_element("string", value)

        elif value is True:
            self.simple_element("true")

        elif value is False:
            self.simple_element("false")

        elif isinstance(value, int):
            if -1 << 63 <= value < 1 << 64:
                self.simple_element("integer", "%d" % value)
            else:
                raise OverflowError(value)

        elif isinstance(value, float):
            self.simple_element("real", repr(value))

        elif isinstance(value, dict):
            self.write_dict(value)

        elif isinstance(value, (bytes, bytearray)):
            self.write_bytes(value)

        elif isinstance(value, datetime.datetime):
            self.simple_element("date",
                                _date_to_string(value, self._aware_datetime))

        elif isinstance(value, (tuple, list)):
            self.write_array(value)

        else:
            raise TypeError("unsupported type: %s" % type(value))

    def write_bytes(self, data):
        self.begin_element("data")
        self._indent_level -= 1
        maxlinelength = max(
            16,
            76 - len((self.indent * self._indent_level).expandtabs()))

        for line in _encode_base64(data, maxlinelength).split(b"\n"):
            if line:
                self.writeln(line)
        self._indent_level += 1
        self.end_element("data")

    def write_dict(self, d):
        if d:
            self.begin_element("dict")
            if self._sort_keys:
                items = sorted(d.items())
            else:
                items = d.items()

            for key, value in items:
                if not isinstance(key, str):
                    if self._skipkeys:
                        continue
                    raise TypeError("keys must be strings")
                self.simple_element("key", key)
                self.write_value(value)
            self.end_element("dict")

        else:
            self.simple_element("dict")

    def write_array(self, array):
        if array:
            self.begin_element("array")
            for value in array:
                self.write_value(value)
            self.end_element("array")

        else:
            self.simple_element("array")


def _is_fmt_xml(header):
    prefixes = (b'<?xml', b'<plist')

    for pfx in prefixes:
        if header.startswith(pfx):
            return True

    # Also check for alternative XML encodings, this is slightly
    # overkill because the Apple tools (and plistlib) will not
    # generate files with these encodings.
    for bom, encoding in (
                (codecs.BOM_UTF8, "utf-8"),
                (codecs.BOM_UTF16_BE, "utf-16-be"),
                (codecs.BOM_UTF16_LE, "utf-16-le"),
                # expat does not support utf-32
                #(codecs.BOM_UTF32_BE, "utf-32-be"),
                #(codecs.BOM_UTF32_LE, "utf-32-le"),
            ):
        if not header.startswith(bom):
            continue

        for start in prefixes:
            prefix = bom + start.decode('ascii').encode(encoding)
            if header[:len(prefix)] == prefix:
                return True

    return False

#
# Binary Plist
#


class InvalidFileException (ValueError):
    def __init__(self, message="Invalid file"):
        ValueError.__init__(self, message)

_BINARY_FORMAT = {1: 'B', 2: 'H', 4: 'L', 8: 'Q'}

_undefined = object()

class _BinaryPlistParser:
    """
    Read or write a binary plist file, following the description of the binary
    format.  Raise InvalidFileException in case of error, otherwise return the
    root object.

    see also: http://opensource.apple.com/source/CF/CF-744.18/CFBinaryPList.c
    """
    def __init__(self, dict_type, aware_datetime=False):
        self._dict_type = dict_type
        self._aware_datime = aware_datetime

    def parse(self, fp):
        try:
            # The basic file format:
            # HEADER
            # object...
            # refid->offset...
            # TRAILER
            self._fp = fp
            self._fp.seek(-32, os.SEEK_END)
            trailer = self._fp.read(32)
            if len(trailer) != 32:
                raise InvalidFileException()
            (
                offset_size, self._ref_size, num_objects, top_object,
                offset_table_offset
            ) = struct.unpack('>6xBBQQQ', trailer)
            self._fp.seek(offset_table_offset)
            self._object_offsets = self._read_ints(num_objects, offset_size)
            self._objects = [_undefined] * num_objects
            return self._read_object(top_object)

        except (OSError, IndexError, struct.error, OverflowError,
                ValueError):
            raise InvalidFileException()

    def _get_size(self, tokenL):
        """ return the size of the next object."""
        if tokenL == 0xF:
            m = self._fp.read(1)[0] & 0x3
            s = 1 << m
            f = '>' + _BINARY_FORMAT[s]
            return struct.unpack(f, self._fp.read(s))[0]

        return tokenL

    def _read(self, size):
        cursize = min(size, _MIN_READ_BUF_SIZE)
        data = self._fp.read(cursize)
        while True:
            if len(data) != cursize:
                raise InvalidFileException
            if cursize == size:
                return data
            delta = min(cursize, size - cursize)
            data += self._fp.read(delta)
            cursize += delta

    def _read_ints(self, n, size):
        data = self._read(size * n)
        if size in _BINARY_FORMAT:
            return struct.unpack(f'>{n}{_BINARY_FORMAT[size]}', data)
        else:
            if not size:
                raise InvalidFileException()
            return tuple(int.from_bytes(data[i: i + size], 'big')
                         for i in range(0, size * n, size))

    def _read_refs(self, n):
        return self._read_ints(n, self._ref_size)

    def _read_object(self, ref):
        """
        read the object by reference.

        May recursively read sub-objects (content of an array/dict/set)
        """
        result = self._objects[ref]
        if result is not _undefined:
            return result

        offset = self._object_offsets[ref]
        self._fp.seek(offset)
        token = self._fp.read(1)[0]
        tokenH, tokenL = token & 0xF0, token & 0x0F

        if token == 0x00:
            result = None

        elif token == 0x08:
            result = False

        elif token == 0x09:
            result = True

        # The referenced source code also mentions URL (0x0c, 0x0d) and
        # UUID (0x0e), but neither can be generated using the Cocoa libraries.

        elif token == 0x0f:
            result = b''

        elif tokenH == 0x10:  # int
            result = int.from_bytes(self._fp.read(1 << tokenL),
                                    'big', signed=tokenL >= 3)

        elif token == 0x22: # real
            result = struct.unpack('>f', self._fp.read(4))[0]

        elif token == 0x23: # real
            result = struct.unpack('>d', self._fp.read(8))[0]

        elif token == 0x33:  # date
            f = struct.unpack('>d', self._fp.read(8))[0]
            # timestamp 0 of binary plists corresponds to 1/1/2001
            # (year of Mac OS X 10.0), instead of 1/1/1970.
            if self._aware_datime:
                epoch = datetime.datetime(2001, 1, 1, tzinfo=datetime.UTC)
            else:
                epoch = datetime.datetime(2001, 1, 1)
            result = epoch + datetime.timedelta(seconds=f)

        elif tokenH == 0x40:  # data
            s = self._get_size(tokenL)
            result = self._read(s)

        elif tokenH == 0x50:  # ascii string
            s = self._get_size(tokenL)
            data = self._read(s)
            result = data.decode('ascii')

        elif tokenH == 0x60:  # unicode string
            s = self._get_size(tokenL) * 2
            data = self._read(s)
            result = data.decode('utf-16be')

        elif tokenH == 0x80:  # UID
            # used by Key-Archiver plist files
            result = UID(int.from_bytes(self._fp.read(1 + tokenL), 'big'))

        elif tokenH == 0xA0:  # array
            s = self._get_size(tokenL)
            obj_refs = self._read_refs(s)
            result = []
            self._objects[ref] = result
            for x in obj_refs:
                result.append(self._read_object(x))

        # tokenH == 0xB0 is documented as 'ordset', but is not actually
        # implemented in the Apple reference code.

        # tokenH == 0xC0 is documented as 'set', but sets cannot be used in
        # plists.

        elif tokenH == 0xD0:  # dict
            s = self._get_size(tokenL)
            key_refs = self._read_refs(s)
            obj_refs = self._read_refs(s)
            result = self._dict_type()
            self._objects[ref] = result
            try:
                for k, o in zip(key_refs, obj_refs):
                    result[self._read_object(k)] = self._read_object(o)
            except TypeError:
                raise InvalidFileException()
        else:
            raise InvalidFileException()

        self._objects[ref] = result
        return result

def _count_to_size(count):
    if count < 1 << 8:
        return 1

    elif count < 1 << 16:
        return 2

    elif count < 1 << 32:
        return 4

    else:
        return 8

_scalars = (str, int, float, datetime.datetime, bytes)

class _BinaryPlistWriter (object):
    def __init__(self, fp, sort_keys, skipkeys, aware_datetime=False):
        self._fp = fp
        self._sort_keys = sort_keys
        self._skipkeys = skipkeys
        self._aware_datetime = aware_datetime

    def write(self, value):

        # Flattened object list:
        self._objlist = []

        # Mappings from object->objectid
        # First dict has (type(object), object) as the key,
        # second dict is used when object is not hashable and
        # has id(object) as the key.
        self._objtable = {}
        self._objidtable = {}

        # Create list of all objects in the plist
        self._flatten(value)

        # Size of object references in serialized containers
        # depends on the number of objects in the plist.
        num_objects = len(self._objlist)
        self._object_offsets = [0]*num_objects
        self._ref_size = _count_to_size(num_objects)

        self._ref_format = _BINARY_FORMAT[self._ref_size]

        # Write file header
        self._fp.write(b'bplist00')

        # Write object list
        for obj in self._objlist:
            self._write_object(obj)

        # Write refnum->object offset table
        top_object = self._getrefnum(value)
        offset_table_offset = self._fp.tell()
        offset_size = _count_to_size(offset_table_offset)
        offset_format = '>' + _BINARY_FORMAT[offset_size] * num_objects
        self._fp.write(struct.pack(offset_format, *self._object_offsets))

        # Write trailer
        sort_version = 0
        trailer = (
            sort_version, offset_size, self._ref_size, num_objects,
            top_object, offset_table_offset
        )
        self._fp.write(struct.pack('>5xBBBQQQ', *trailer))

    def _flatten(self, value):
        # First check if the object is in the object table, not used for
        # containers to ensure that two subcontainers with the same contents
        # will be serialized as distinct values.
        if isinstance(value, _scalars):
            if (type(value), value) in self._objtable:
                return

        elif id(value) in self._objidtable:
            return

        # Add to objectreference map
        refnum = len(self._objlist)
        self._objlist.append(value)
        if isinstance(value, _scalars):
            self._objtable[(type(value), value)] = refnum
        else:
            self._objidtable[id(value)] = refnum

        # And finally recurse into containers
        if isinstance(value, dict):
            keys = []
            values = []
            items = value.items()
            if self._sort_keys:
                items = sorted(items)

            for k, v in items:
                if not isinstance(k, str):
                    if self._skipkeys:
                        continue
                    raise TypeError("keys must be strings")
                keys.append(k)
                values.append(v)

            for o in itertools.chain(keys, values):
                self._flatten(o)

        elif isinstance(value, (list, tuple)):
            for o in value:
                self._flatten(o)

    def _getrefnum(self, value):
        if isinstance(value, _scalars):
            return self._objtable[(type(value), value)]
        else:
            return self._objidtable[id(value)]

    def _write_size(self, token, size):
        if size < 15:
            self._fp.write(struct.pack('>B', token | size))

        elif size < 1 << 8:
            self._fp.write(struct.pack('>BBB', token | 0xF, 0x10, size))

        elif size < 1 << 16:
            self._fp.write(struct.pack('>BBH', token | 0xF, 0x11, size))

        elif size < 1 << 32:
            self._fp.write(struct.pack('>BBL', token | 0xF, 0x12, size))

        else:
            self._fp.write(struct.pack('>BBQ', token | 0xF, 0x13, size))

    def _write_object(self, value):
        ref = self._getrefnum(value)
        self._object_offsets[ref] = self._fp.tell()
        if value is None:
            self._fp.write(b'\x00')

        elif value is False:
            self._fp.write(b'\x08')

        elif value is True:
            self._fp.write(b'\x09')

        elif isinstance(value, int):
            if value < 0:
                try:
                    self._fp.write(struct.pack('>Bq', 0x13, value))
                except struct.error:
                    raise OverflowError(value) from None
            elif value < 1 << 8:
                self._fp.write(struct.pack('>BB', 0x10, value))
            elif value < 1 << 16:
                self._fp.write(struct.pack('>BH', 0x11, value))
            elif value < 1 << 32:
                self._fp.write(struct.pack('>BL', 0x12, value))
            elif value < 1 << 63:
                self._fp.write(struct.pack('>BQ', 0x13, value))
            elif value < 1 << 64:
                self._fp.write(b'\x14' + value.to_bytes(16, 'big', signed=True))
            else:
                raise OverflowError(value)

        elif isinstance(value, float):
            self._fp.write(struct.pack('>Bd', 0x23, value))

        elif isinstance(value, datetime.datetime):
            if self._aware_datetime:
                dt = value.astimezone(datetime.UTC)
                offset = dt - datetime.datetime(2001, 1, 1, tzinfo=datetime.UTC)
                f = offset.total_seconds()
            else:
                f = (value - datetime.datetime(2001, 1, 1)).total_seconds()
            self._fp.write(struct.pack('>Bd', 0x33, f))

        elif isinstance(value, (bytes, bytearray)):
            self._write_size(0x40, len(value))
            self._fp.write(value)

        elif isinstance(value, str):
            try:
                t = value.encode('ascii')
                self._write_size(0x50, len(value))
            except UnicodeEncodeError:
                t = value.encode('utf-16be')
                self._write_size(0x60, len(t) // 2)

            self._fp.write(t)

        elif isinstance(value, UID):
            if value.data < 0:
                raise ValueError("UIDs must be positive")
            elif value.data < 1 << 8:
                self._fp.write(struct.pack('>BB', 0x80, value))
            elif value.data < 1 << 16:
                self._fp.write(struct.pack('>BH', 0x81, value))
            elif value.data < 1 << 32:
                self._fp.write(struct.pack('>BL', 0x83, value))
            elif value.data < 1 << 64:
                self._fp.write(struct.pack('>BQ', 0x87, value))
            else:
                raise OverflowError(value)

        elif isinstance(value, (list, tuple)):
            refs = [self._getrefnum(o) for o in value]
            s = len(refs)
            self._write_size(0xA0, s)
            self._fp.write(struct.pack('>' + self._ref_format * s, *refs))

        elif isinstance(value, dict):
            keyRefs, valRefs = [], []

            if self._sort_keys:
                rootItems = sorted(value.items())
            else:
                rootItems = value.items()

            for k, v in rootItems:
                if not isinstance(k, str):
                    if self._skipkeys:
                        continue
                    raise TypeError("keys must be strings")
                keyRefs.append(self._getrefnum(k))
                valRefs.append(self._getrefnum(v))

            s = len(keyRefs)
            self._write_size(0xD0, s)
            self._fp.write(struct.pack('>' + self._ref_format * s, *keyRefs))
            self._fp.write(struct.pack('>' + self._ref_format * s, *valRefs))

        else:
            raise TypeError(value)


def _is_fmt_binary(header):
    return header[:8] == b'bplist00'


#
# Generic bits
#

_FORMATS={
    FMT_XML: dict(
        detect=_is_fmt_xml,
        parser=_PlistParser,
        writer=_PlistWriter,
    ),
    FMT_BINARY: dict(
        detect=_is_fmt_binary,
        parser=_BinaryPlistParser,
        writer=_BinaryPlistWriter,
    )
}


def load(fp, *, fmt=None, dict_type=dict, aware_datetime=False):
    """Read a .plist file. 'fp' should be a readable and binary file object.
    Return the unpacked root object (which usually is a dictionary).
    """
    if fmt is None:
        header = fp.read(32)
        fp.seek(0)
        for info in _FORMATS.values():
            if info['detect'](header):
                P = info['parser']
                break

        else:
            raise InvalidFileException()

    else:
        P = _FORMATS[fmt]['parser']

    p = P(dict_type=dict_type, aware_datetime=aware_datetime)
    return p.parse(fp)


def loads(value, *, fmt=None, dict_type=dict, aware_datetime=False):
    """Read a .plist file from a bytes object.
    Return the unpacked root object (which usually is a dictionary).
    """
    if isinstance(value, str):
        if fmt == FMT_BINARY:
            raise TypeError("value must be bytes-like object when fmt is "
                            "FMT_BINARY")
        value = value.encode()
    fp = BytesIO(value)
    return load(fp, fmt=fmt, dict_type=dict_type, aware_datetime=aware_datetime)


def dump(value, fp, *, fmt=FMT_XML, sort_keys=True, skipkeys=False,
         aware_datetime=False):
    """Write 'value' to a .plist file. 'fp' should be a writable,
    binary file object.
    """
    if fmt not in _FORMATS:
        raise ValueError("Unsupported format: %r"%(fmt,))

    writer = _FORMATS[fmt]["writer"](fp, sort_keys=sort_keys, skipkeys=skipkeys,
                                     aware_datetime=aware_datetime)
    writer.write(value)


def dumps(value, *, fmt=FMT_XML, skipkeys=False, sort_keys=True,
          aware_datetime=False):
    """Return a bytes object with the contents for a .plist file.
    """
    fp = BytesIO()
    dump(value, fp, fmt=fmt, skipkeys=skipkeys, sort_keys=sort_keys,
         aware_datetime=aware_datetime)
    return fp.getvalue()

import warnings
warnings.warn(f"module {__name__!r} is deprecated",
              DeprecationWarning,
              stacklevel=2)

from re import _constants as _
globals().update({k: v for k, v in vars(_).items() if k[:2] != '__'})

# system configuration generated and used by the sysconfig module
build_time_vars = {
    'ABIFLAGS': '',
    'ABI_THREAD': '',
    'AC_APPLE_UNIVERSAL_BUILD': 0,
    'AIX_BUILDDATE': 0,
    'AIX_GENUINE_CPLUSPLUS': 0,
    'ALIGNOF_LONG': 8,
    'ALIGNOF_MAX_ALIGN_T': 16,
    'ALIGNOF_SIZE_T': 8,
    'ALT_SOABI': 0,
    'ANDROID_API_LEVEL': 0,
    'APP_STORE_COMPLIANCE_PATCH': '',
    'AR': 'x86_64-conda-linux-gnu-ar',
    'ARFLAGS': 'rcs',
    'BASECFLAGS': '-fno-strict-overflow -Wsign-compare',
    'BASECPPFLAGS': '-IObjects -IInclude -IPython',
    'BASEMODLIBS': '',
    'BINDIR': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/bin',
    'BINLIBDEST': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib/python3.14',
    'BLDLIBRARY': 'libpython3.14.a',
    'BLDSHARED': 'x86_64-conda-linux-gnu-gcc -pthread -shared -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib',
    'BOOTSTRAP_HEADERS': '\\',
    'BUILDEXE': '',
    'BUILDPYTHON': 'python',
    'BUILD_GNU_TYPE': 'x86_64-conda-linux-gnu',
    'BUILD_SCRIPTS_DIR': 'build/scripts-3.14',
    'BYTESTR_DEPS': '\\',
    'CC': 'x86_64-conda-linux-gnu-gcc -pthread',
    'CCSHARED': '-fPIC',
    'CFLAGS': '-fno-strict-overflow -Wsign-compare -DNDEBUG -O2 -Wall -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -fno-merge-constants -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include       -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -fno-merge-constants -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include      ',
    'CFLAGSFORSHARED': '',
    'CFLAGS_ALIASING': '-fno-strict-aliasing',
    'CFLAGS_CEVAL': '',
    'CODECS_COMMON_HEADERS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/multibytecodec.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/cjkcodecs.h',
    'COMPILEALL_OPTS': '-j0',
    'CONFIGFILES': 'configure configure.ac acconfig.h pyconfig.h.in Makefile.pre.in',
    'CONFIGURE_CFLAGS': '-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -fno-merge-constants -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include      ',
    'CONFIGURE_CFLAGS_NODIST': '-fno-semantic-interposition    -g -std=c11 -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wstrict-prototypes -Werror=implicit-function-declaration -fvisibility=hidden -D_Py_TIER2=3 -D_Py_JIT',
    'CONFIGURE_CPPFLAGS': '-DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include',
    'CONFIGURE_LDFLAGS': '-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib',
    'CONFIGURE_LDFLAGS_NODIST': '-fno-semantic-interposition    -g',
    'CONFIGURE_LDFLAGS_NOLTO': '-fno-lto',
    'CONFIG_ARGS': "'--prefix=/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default' '--build=x86_64-conda-linux-gnu' '--host=x86_64-conda-linux-gnu' '--enable-ipv6' '--with-ensurepip=no' '--with-tzpath=/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/share/zoneinfo:/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/share/tzinfo' '--with-computed-gotos' '--with-system-expat' '--enable-loadable-sqlite-extensions' '--with-tcltk-includes=-I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include' '--with-tcltk-libs=-L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -ltcl8.6 -ltk8.6' '--with-platlibdir=lib' '--enable-experimental-jit=yes-off' '--with-lto=full' '--enable-optimizations' '-oldincludedir=/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/_build_env/x86_64-conda-linux-gnu/sysroot/usr/include' '--disable-shared' 'PROFILE_TASK=-m test --pgo' 'build_alias=x86_64-conda-linux-gnu' 'host_alias=x86_64-conda-linux-gnu' 'PKG_CONFIG_PATH=/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib/pkgconfig' 'MACHDEP=linux' 'CC=x86_64-conda-linux-gnu-gcc' 'CFLAGS=-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -fno-merge-constants -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include      ' 'LDFLAGS=-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib' 'CPPFLAGS=-DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include' 'CPP=x86_64-conda-linux-gnu-cpp'",
    'CONFINCLUDEDIR': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include',
    'CONFINCLUDEPY': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include/python3.14',
    'COREPYTHONPATH': '',
    'COVERAGE_INFO': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/build-static/coverage.info',
    'COVERAGE_LCOV_OPTIONS': '--rc lcov_branch_coverage=1',
    'COVERAGE_REPORT': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/build-static/lcov-report',
    'COVERAGE_REPORT_OPTIONS': '--rc lcov_branch_coverage=1 --branch-coverage --title "CPython 3.14 LCOV report [commit $(shell )]"',
    'CPPFLAGS': '-IObjects -IInclude -IPython -I. -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include',
    'CXX': 'x86_64-conda-linux-gnu-c++ -pthread',
    'DESTDIRS': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib/python3.14 /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib/python3.14/lib-dynload',
    'DESTLIB': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib/python3.14',
    'DESTPATH': '',
    'DESTSHARED': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib/python3.14/lib-dynload',
    'DFLAGS': '',
    'DIRMODE': 755,
    'DIST': 'README.rst ChangeLog configure configure.ac acconfig.h pyconfig.h.in Makefile.pre.in Include Lib Misc Ext-dummy',
    'DISTDIRS': 'Include Lib Misc Ext-dummy',
    'DISTFILES': 'README.rst ChangeLog configure configure.ac acconfig.h pyconfig.h.in Makefile.pre.in',
    'DLINCLDIR': '.',
    'DLLLIBRARY': '',
    'DOUBLE_IS_ARM_MIXED_ENDIAN_IEEE754': 0,
    'DOUBLE_IS_BIG_ENDIAN_IEEE754': 0,
    'DOUBLE_IS_LITTLE_ENDIAN_IEEE754': 1,
    'DSYMUTIL': '',
    'DSYMUTIL_PATH': '',
    'DTRACE': '',
    'DTRACE_DEPS': '\\',
    'DTRACE_HEADERS': '',
    'DTRACE_OBJS': '',
    'DYNLOADFILE': 'dynload_shlib.o',
    'EMSCRIPTEN_DIR': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Platforms/emscripten',
    'ENABLE_IPV6': 1,
    'ENSUREPIP': 'no',
    'EXE': '',
    'EXEMODE': 755,
    'EXENAME': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/bin/python3.14',
    'EXPORTSFROM': '',
    'EXPORTSYMS': '',
    'EXTRATESTOPTS': '',
    'EXT_SUFFIX': '.cpython-314-x86_64-linux-gnu.so',
    'FILEMODE': 644,
    'FREEZE_MODULE': './_bootstrap_python /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Programs/_freeze_module.py',
    'FREEZE_MODULE_BOOTSTRAP': './Programs/_freeze_module',
    'FREEZE_MODULE_BOOTSTRAP_DEPS': 'Programs/_freeze_module',
    'FREEZE_MODULE_DEPS': '_bootstrap_python /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Programs/_freeze_module.py',
    'FROZEN_FILES_IN': '\\',
    'FROZEN_FILES_OUT': '\\',
    'GETPGRP_HAVE_ARG': 0,
    'GITBRANCH': '',
    'GITTAG': '',
    'GITVERSION': '',
    'GNULD': 'no',
    'HAVE_ACCEPT': 1,
    'HAVE_ACCEPT4': 1,
    'HAVE_ACOSH': 1,
    'HAVE_ADDRINFO': 1,
    'HAVE_ALARM': 1,
    'HAVE_ALIGNED_REQUIRED': 0,
    'HAVE_ALLOCA_H': 1,
    'HAVE_ALTZONE': 0,
    'HAVE_ASINH': 1,
    'HAVE_ASM_TYPES_H': 1,
    'HAVE_ATANH': 1,
    'HAVE_BACKTRACE': 1,
    'HAVE_BIND': 1,
    'HAVE_BIND_TEXTDOMAIN_CODESET': 1,
    'HAVE_BLUETOOTH_BLUETOOTH_H': 0,
    'HAVE_BLUETOOTH_H': 0,
    'HAVE_BROKEN_MBSTOWCS': 0,
    'HAVE_BROKEN_NICE': 0,
    'HAVE_BROKEN_PIPE_BUF': 0,
    'HAVE_BROKEN_POLL': 0,
    'HAVE_BROKEN_POSIX_SEMAPHORES': 0,
    'HAVE_BROKEN_PTHREAD_SIGMASK': 0,
    'HAVE_BROKEN_SEM_GETVALUE': 0,
    'HAVE_BROKEN_UNSETENV': 0,
    'HAVE_BUILTIN_ATOMIC': 1,
    'HAVE_BZLIB_H': 1,
    'HAVE_CHFLAGS': 0,
    'HAVE_CHMOD': 1,
    'HAVE_CHOWN': 1,
    'HAVE_CHROOT': 1,
    'HAVE_CLOCK': 1,
    'HAVE_CLOCK_GETRES': 1,
    'HAVE_CLOCK_GETTIME': 1,
    'HAVE_CLOCK_NANOSLEEP': 1,
    'HAVE_CLOCK_SETTIME': 1,
    'HAVE_CLOCK_T': 1,
    'HAVE_CLOSEFROM': 0,
    'HAVE_CLOSE_RANGE': 0,
    'HAVE_COMPUTED_GOTOS': 1,
    'HAVE_CONFSTR': 1,
    'HAVE_CONIO_H': 0,
    'HAVE_CONNECT': 1,
    'HAVE_COPY_FILE_RANGE': 0,
    'HAVE_CTERMID': 1,
    'HAVE_CTERMID_R': 0,
    'HAVE_CURSES_ESCDELAY': 1,
    'HAVE_CURSES_FILTER': 1,
    'HAVE_CURSES_GETMOUSE': 1,
    'HAVE_CURSES_H': 1,
    'HAVE_CURSES_HAS_KEY': 1,
    'HAVE_CURSES_IMMEDOK': 1,
    'HAVE_CURSES_IS_PAD': 1,
    'HAVE_CURSES_IS_TERM_RESIZED': 1,
    'HAVE_CURSES_RESIZETERM': 1,
    'HAVE_CURSES_RESIZE_TERM': 1,
    'HAVE_CURSES_SET_ESCDELAY': 1,
    'HAVE_CURSES_SET_TABSIZE': 1,
    'HAVE_CURSES_SYNCOK': 1,
    'HAVE_CURSES_TABSIZE': 1,
    'HAVE_CURSES_TYPEAHEAD': 1,
    'HAVE_CURSES_USE_ENV': 1,
    'HAVE_CURSES_WCHGAT': 1,
    'HAVE_DB_H': 0,
    'HAVE_DECL_RTLD_DEEPBIND': 1,
    'HAVE_DECL_RTLD_GLOBAL': 1,
    'HAVE_DECL_RTLD_LAZY': 1,
    'HAVE_DECL_RTLD_LOCAL': 1,
    'HAVE_DECL_RTLD_MEMBER': 0,
    'HAVE_DECL_RTLD_NODELETE': 1,
    'HAVE_DECL_RTLD_NOLOAD': 1,
    'HAVE_DECL_RTLD_NOW': 1,
    'HAVE_DECL_TZNAME': 0,
    'HAVE_DECL_UT_NAMESIZE': 1,
    'HAVE_DEVICE_MACROS': 1,
    'HAVE_DEV_PTC': 0,
    'HAVE_DEV_PTMX': 1,
    'HAVE_DIRECT_H': 0,
    'HAVE_DIRENT_D_TYPE': 1,
    'HAVE_DIRENT_H': 1,
    'HAVE_DIRFD': 1,
    'HAVE_DLADDR': 1,
    'HAVE_DLADDR1': 1,
    'HAVE_DLFCN_H': 1,
    'HAVE_DLOPEN': 1,
    'HAVE_DL_ITERATE_PHDR': 1,
    'HAVE_DUP': 1,
    'HAVE_DUP2': 1,
    'HAVE_DUP3': 1,
    'HAVE_DYLD_SHARED_CACHE_CONTAINS_PATH': 0,
    'HAVE_DYNAMIC_LOADING': 1,
    'HAVE_EDITLINE_READLINE_H': 0,
    'HAVE_ENDIAN_H': 1,
    'HAVE_EPOLL': 1,
    'HAVE_EPOLL_CREATE1': 1,
    'HAVE_ERF': 1,
    'HAVE_ERFC': 1,
    'HAVE_ERRNO_H': 1,
    'HAVE_EVENTFD': 1,
    'HAVE_EXECINFO_H': 1,
    'HAVE_EXECV': 1,
    'HAVE_EXPLICIT_BZERO': 0,
    'HAVE_EXPLICIT_MEMSET': 0,
    'HAVE_EXPM1': 1,
    'HAVE_FACCESSAT': 1,
    'HAVE_FCHDIR': 1,
    'HAVE_FCHMOD': 1,
    'HAVE_FCHMODAT': 1,
    'HAVE_FCHOWN': 1,
    'HAVE_FCHOWNAT': 1,
    'HAVE_FCNTL_H': 1,
    'HAVE_FDATASYNC': 1,
    'HAVE_FDOPENDIR': 1,
    'HAVE_FDWALK': 0,
    'HAVE_FEXECVE': 1,
    'HAVE_FFI_CLOSURE_ALLOC': 1,
    'HAVE_FFI_PREP_CIF_VAR': 1,
    'HAVE_FFI_PREP_CLOSURE_LOC': 1,
    'HAVE_FLOCK': 1,
    'HAVE_FORK': 1,
    'HAVE_FORK1': 0,
    'HAVE_FORKPTY': 1,
    'HAVE_FPATHCONF': 1,
    'HAVE_FSEEK64': 0,
    'HAVE_FSEEKO': 1,
    'HAVE_FSTATAT': 1,
    'HAVE_FSTATVFS': 1,
    'HAVE_FSYNC': 1,
    'HAVE_FTELL64': 0,
    'HAVE_FTELLO': 1,
    'HAVE_FTIME': 1,
    'HAVE_FTRUNCATE': 1,
    'HAVE_FUTIMENS': 1,
    'HAVE_FUTIMES': 1,
    'HAVE_FUTIMESAT': 1,
    'HAVE_GAI_STRERROR': 1,
    'HAVE_GCC_ASM_FOR_MC68881': 0,
    'HAVE_GCC_ASM_FOR_X64': 1,
    'HAVE_GCC_ASM_FOR_X87': 1,
    'HAVE_GCC_UINT128_T': 1,
    'HAVE_GDBM_DASH_NDBM_H': 0,
    'HAVE_GDBM_H': 0,
    'HAVE_GDBM_NDBM_H': 0,
    'HAVE_GETADDRINFO': 1,
    'HAVE_GETC_UNLOCKED': 1,
    'HAVE_GETEGID': 1,
    'HAVE_GETENTROPY': 0,
    'HAVE_GETEUID': 1,
    'HAVE_GETGID': 1,
    'HAVE_GETGRENT': 1,
    'HAVE_GETGRGID': 1,
    'HAVE_GETGRGID_R': 1,
    'HAVE_GETGRNAM_R': 1,
    'HAVE_GETGROUPLIST': 1,
    'HAVE_GETGROUPS': 1,
    'HAVE_GETHOSTBYADDR': 1,
    'HAVE_GETHOSTBYNAME': 1,
    'HAVE_GETHOSTBYNAME_R': 1,
    'HAVE_GETHOSTBYNAME_R_3_ARG': 0,
    'HAVE_GETHOSTBYNAME_R_5_ARG': 0,
    'HAVE_GETHOSTBYNAME_R_6_ARG': 1,
    'HAVE_GETHOSTNAME': 1,
    'HAVE_GETITIMER': 1,
    'HAVE_GETLOADAVG': 1,
    'HAVE_GETLOGIN': 1,
    'HAVE_GETLOGIN_R': 1,
    'HAVE_GETNAMEINFO': 1,
    'HAVE_GETPAGESIZE': 1,
    'HAVE_GETPEERNAME': 1,
    'HAVE_GETPGID': 1,
    'HAVE_GETPGRP': 1,
    'HAVE_GETPID': 1,
    'HAVE_GETPPID': 1,
    'HAVE_GETPRIORITY': 1,
    'HAVE_GETPROTOBYNAME': 1,
    'HAVE_GETPWENT': 1,
    'HAVE_GETPWNAM_R': 1,
    'HAVE_GETPWUID': 1,
    'HAVE_GETPWUID_R': 1,
    'HAVE_GETRANDOM': 0,
    'HAVE_GETRANDOM_SYSCALL': 1,
    'HAVE_GETRESGID': 1,
    'HAVE_GETRESUID': 1,
    'HAVE_GETRUSAGE': 1,
    'HAVE_GETSERVBYNAME': 1,
    'HAVE_GETSERVBYPORT': 1,
    'HAVE_GETSID': 1,
    'HAVE_GETSOCKNAME': 1,
    'HAVE_GETSPENT': 1,
    'HAVE_GETSPNAM': 1,
    'HAVE_GETUID': 1,
    'HAVE_GETWD': 1,
    'HAVE_GLIBC_MEMMOVE_BUG': 0,
    'HAVE_GRANTPT': 1,
    'HAVE_GRP_H': 1,
    'HAVE_HSTRERROR': 1,
    'HAVE_HTOLE64': 1,
    'HAVE_IF_NAMEINDEX': 1,
    'HAVE_INET_ATON': 1,
    'HAVE_INET_NTOA': 1,
    'HAVE_INET_PTON': 1,
    'HAVE_INITGROUPS': 1,
    'HAVE_INTTYPES_H': 1,
    'HAVE_IO_H': 0,
    'HAVE_IPA_PURE_CONST_BUG': 0,
    'HAVE_KILL': 1,
    'HAVE_KILLPG': 1,
    'HAVE_KQUEUE': 0,
    'HAVE_LANGINFO_H': 1,
    'HAVE_LARGEFILE_SUPPORT': 0,
    'HAVE_LCHFLAGS': 0,
    'HAVE_LCHMOD': 0,
    'HAVE_LCHOWN': 1,
    'HAVE_LIBDB': 0,
    'HAVE_LIBDL': 1,
    'HAVE_LIBDLD': 0,
    'HAVE_LIBIEEE': 0,
    'HAVE_LIBINTL_H': 1,
    'HAVE_LIBSENDFILE': 0,
    'HAVE_LIBSQLITE3': 1,
    'HAVE_LIBUTIL_H': 0,
    'HAVE_LINK': 1,
    'HAVE_LINKAT': 1,
    'HAVE_LINK_H': 1,
    'HAVE_LINUX_AUXVEC_H': 1,
    'HAVE_LINUX_CAN_BCM_H': 1,
    'HAVE_LINUX_CAN_H': 1,
    'HAVE_LINUX_CAN_J1939_H': 0,
    'HAVE_LINUX_CAN_RAW_FD_FRAMES': 1,
    'HAVE_LINUX_CAN_RAW_H': 1,
    'HAVE_LINUX_CAN_RAW_JOIN_FILTERS': 1,
    'HAVE_LINUX_FS_H': 1,
    'HAVE_LINUX_LIMITS_H': 1,
    'HAVE_LINUX_MEMFD_H': 1,
    'HAVE_LINUX_NETFILTER_IPV4_H': 1,
    'HAVE_LINUX_NETLINK_H': 1,
    'HAVE_LINUX_QRTR_H': 0,
    'HAVE_LINUX_RANDOM_H': 1,
    'HAVE_LINUX_SCHED_H': 1,
    'HAVE_LINUX_SOUNDCARD_H': 1,
    'HAVE_LINUX_TIPC_H': 1,
    'HAVE_LINUX_VM_SOCKETS_H': 1,
    'HAVE_LINUX_WAIT_H': 1,
    'HAVE_LISTEN': 1,
    'HAVE_LOCKF': 1,
    'HAVE_LOG1P': 1,
    'HAVE_LOG2': 1,
    'HAVE_LOGIN_TTY': 1,
    'HAVE_LONG_DOUBLE': 1,
    'HAVE_LSTAT': 1,
    'HAVE_LUTIMES': 1,
    'HAVE_LZMA_H': 0,
    'HAVE_MADVISE': 1,
    'HAVE_MAKEDEV': 1,
    'HAVE_MAXLOGNAME': 0,
    'HAVE_MBRTOWC': 1,
    'HAVE_MEMFD_CREATE': 0,
    'HAVE_MEMRCHR': 1,
    'HAVE_MINIX_CONFIG_H': 0,
    'HAVE_MKDIRAT': 1,
    'HAVE_MKFIFO': 1,
    'HAVE_MKFIFOAT': 1,
    'HAVE_MKNOD': 1,
    'HAVE_MKNODAT': 1,
    'HAVE_MKTIME': 1,
    'HAVE_MMAP': 1,
    'HAVE_MREMAP': 1,
    'HAVE_NANOSLEEP': 1,
    'HAVE_NCURSES': 0,
    'HAVE_NCURSESW': 1,
    'HAVE_NCURSESW_CURSES_H': 1,
    'HAVE_NCURSESW_NCURSES_H': 1,
    'HAVE_NCURSESW_PANEL_H': 1,
    'HAVE_NCURSES_CURSES_H': 1,
    'HAVE_NCURSES_H': 1,
    'HAVE_NCURSES_NCURSES_H': 1,
    'HAVE_NCURSES_PANEL_H': 1,
    'HAVE_NDBM_H': 0,
    'HAVE_NDIR_H': 0,
    'HAVE_NETCAN_CAN_H': 0,
    'HAVE_NETDB_H': 1,
    'HAVE_NETINET_IN_H': 1,
    'HAVE_NETLINK_NETLINK_H': 0,
    'HAVE_NETPACKET_PACKET_H': 1,
    'HAVE_NET_ETHERNET_H': 1,
    'HAVE_NET_IF_H': 1,
    'HAVE_NICE': 1,
    'HAVE_NON_UNICODE_WCHAR_T_REPRESENTATION': 0,
    'HAVE_OPENAT': 1,
    'HAVE_OPENDIR': 1,
    'HAVE_OPENPTY': 1,
    'HAVE_PANEL': 0,
    'HAVE_PANELW': 1,
    'HAVE_PANEL_H': 1,
    'HAVE_PATHCONF': 1,
    'HAVE_PAUSE': 1,
    'HAVE_PIPE': 1,
    'HAVE_PIPE2': 1,
    'HAVE_PLOCK': 0,
    'HAVE_POLL': 1,
    'HAVE_POLL_H': 1,
    'HAVE_POSIX_FADVISE': 1,
    'HAVE_POSIX_FALLOCATE': 1,
    'HAVE_POSIX_OPENPT': 1,
    'HAVE_POSIX_SPAWN': 1,
    'HAVE_POSIX_SPAWNP': 1,
    'HAVE_POSIX_SPAWN_FILE_ACTIONS_ADDCLOSEFROM_NP': 0,
    'HAVE_PREAD': 1,
    'HAVE_PREADV': 1,
    'HAVE_PREADV2': 0,
    'HAVE_PRLIMIT': 1,
    'HAVE_PROCESS_H': 0,
    'HAVE_PROCESS_VM_READV': 1,
    'HAVE_PROTOTYPES': 1,
    'HAVE_PTHREAD_CONDATTR_SETCLOCK': 1,
    'HAVE_PTHREAD_COND_TIMEDWAIT_RELATIVE_NP': 0,
    'HAVE_PTHREAD_DESTRUCTOR': 0,
    'HAVE_PTHREAD_GETATTR_NP': 1,
    'HAVE_PTHREAD_GETCPUCLOCKID': 1,
    'HAVE_PTHREAD_GETNAME_NP': 1,
    'HAVE_PTHREAD_GET_NAME_NP': 0,
    'HAVE_PTHREAD_H': 1,
    'HAVE_PTHREAD_INIT': 0,
    'HAVE_PTHREAD_KILL': 1,
    'HAVE_PTHREAD_SETNAME_NP': 1,
    'HAVE_PTHREAD_SET_NAME_NP': 0,
    'HAVE_PTHREAD_SIGMASK': 1,
    'HAVE_PTHREAD_STUBS': 0,
    'HAVE_PTSNAME': 1,
    'HAVE_PTSNAME_R': 1,
    'HAVE_PTY_H': 1,
    'HAVE_PWRITE': 1,
    'HAVE_PWRITEV': 1,
    'HAVE_PWRITEV2': 0,
    'HAVE_READLINE_READLINE_H': 0,
    'HAVE_READLINK': 1,
    'HAVE_READLINKAT': 1,
    'HAVE_READV': 1,
    'HAVE_REALPATH': 1,
    'HAVE_RECVFROM': 1,
    'HAVE_RENAMEAT': 1,
    'HAVE_RL_APPEND_HISTORY': 1,
    'HAVE_RL_CATCH_SIGNAL': 1,
    'HAVE_RL_COMPDISP_FUNC_T': 1,
    'HAVE_RL_COMPLETION_APPEND_CHARACTER': 1,
    'HAVE_RL_COMPLETION_DISPLAY_MATCHES_HOOK': 1,
    'HAVE_RL_COMPLETION_MATCHES': 1,
    'HAVE_RL_COMPLETION_SUPPRESS_APPEND': 1,
    'HAVE_RL_PRE_INPUT_HOOK': 1,
    'HAVE_RL_RESIZE_TERMINAL': 1,
    'HAVE_RTPSPAWN': 0,
    'HAVE_SCHED_GET_PRIORITY_MAX': 1,
    'HAVE_SCHED_H': 1,
    'HAVE_SCHED_RR_GET_INTERVAL': 1,
    'HAVE_SCHED_SETAFFINITY': 1,
    'HAVE_SCHED_SETPARAM': 1,
    'HAVE_SCHED_SETSCHEDULER': 1,
    'HAVE_SEM_CLOCKWAIT': 0,
    'HAVE_SEM_GETVALUE': 1,
    'HAVE_SEM_OPEN': 1,
    'HAVE_SEM_TIMEDWAIT': 1,
    'HAVE_SEM_UNLINK': 1,
    'HAVE_SENDFILE': 1,
    'HAVE_SENDTO': 1,
    'HAVE_SETEGID': 1,
    'HAVE_SETEUID': 1,
    'HAVE_SETGID': 1,
    'HAVE_SETGROUPS': 1,
    'HAVE_SETHOSTNAME': 1,
    'HAVE_SETITIMER': 1,
    'HAVE_SETJMP_H': 1,
    'HAVE_SETLOCALE': 1,
    'HAVE_SETNS': 1,
    'HAVE_SETPGID': 1,
    'HAVE_SETPGRP': 1,
    'HAVE_SETPRIORITY': 1,
    'HAVE_SETREGID': 1,
    'HAVE_SETRESGID': 1,
    'HAVE_SETRESUID': 1,
    'HAVE_SETREUID': 1,
    'HAVE_SETSID': 1,
    'HAVE_SETSOCKOPT': 1,
    'HAVE_SETUID': 1,
    'HAVE_SETVBUF': 1,
    'HAVE_SHADOW_H': 1,
    'HAVE_SHM_OPEN': 1,
    'HAVE_SHM_UNLINK': 1,
    'HAVE_SHUTDOWN': 1,
    'HAVE_SIGACTION': 1,
    'HAVE_SIGALTSTACK': 1,
    'HAVE_SIGFILLSET': 1,
    'HAVE_SIGINFO_T_SI_BAND': 1,
    'HAVE_SIGINTERRUPT': 1,
    'HAVE_SIGNAL_H': 1,
    'HAVE_SIGPENDING': 1,
    'HAVE_SIGRELSE': 1,
    'HAVE_SIGTIMEDWAIT': 1,
    'HAVE_SIGWAIT': 1,
    'HAVE_SIGWAITINFO': 1,
    'HAVE_SNPRINTF': 1,
    'HAVE_SOCKADDR_ALG': 1,
    'HAVE_SOCKADDR_SA_LEN': 0,
    'HAVE_SOCKADDR_STORAGE': 1,
    'HAVE_SOCKET': 1,
    'HAVE_SOCKETPAIR': 1,
    'HAVE_SOCKLEN_T': 1,
    'HAVE_SPAWN_H': 1,
    'HAVE_SPLICE': 1,
    'HAVE_SSIZE_T': 1,
    'HAVE_STATVFS': 1,
    'HAVE_STAT_TV_NSEC': 1,
    'HAVE_STAT_TV_NSEC2': 0,
    'HAVE_STDINT_H': 1,
    'HAVE_STDIO_H': 1,
    'HAVE_STDLIB_H': 1,
    'HAVE_STD_ATOMIC': 1,
    'HAVE_STRFTIME': 1,
    'HAVE_STRINGS_H': 1,
    'HAVE_STRING_H': 1,
    'HAVE_STRLCPY': 0,
    'HAVE_STROPTS_H': 0,
    'HAVE_STRSIGNAL': 1,
    'HAVE_STRUCT_PASSWD_PW_GECOS': 1,
    'HAVE_STRUCT_PASSWD_PW_PASSWD': 1,
    'HAVE_STRUCT_STAT_ST_BIRTHTIME': 0,
    'HAVE_STRUCT_STAT_ST_BLKSIZE': 1,
    'HAVE_STRUCT_STAT_ST_BLOCKS': 1,
    'HAVE_STRUCT_STAT_ST_FLAGS': 0,
    'HAVE_STRUCT_STAT_ST_GEN': 0,
    'HAVE_STRUCT_STAT_ST_RDEV': 1,
    'HAVE_STRUCT_TM_TM_ZONE': 1,
    'HAVE_SYMLINK': 1,
    'HAVE_SYMLINKAT': 1,
    'HAVE_SYNC': 1,
    'HAVE_SYSCONF': 1,
    'HAVE_SYSCTLBYNAME': 0,
    'HAVE_SYSEXITS_H': 1,
    'HAVE_SYSLOG_H': 1,
    'HAVE_SYSTEM': 1,
    'HAVE_SYS_AUDIOIO_H': 0,
    'HAVE_SYS_AUXV_H': 1,
    'HAVE_SYS_BSDTTY_H': 0,
    'HAVE_SYS_DEVPOLL_H': 0,
    'HAVE_SYS_DIR_H': 0,
    'HAVE_SYS_ENDIAN_H': 0,
    'HAVE_SYS_EPOLL_H': 1,
    'HAVE_SYS_EVENTFD_H': 1,
    'HAVE_SYS_EVENT_H': 0,
    'HAVE_SYS_FILE_H': 1,
    'HAVE_SYS_IOCTL_H': 1,
    'HAVE_SYS_KERN_CONTROL_H': 0,
    'HAVE_SYS_LOADAVG_H': 0,
    'HAVE_SYS_LOCK_H': 0,
    'HAVE_SYS_MEMFD_H': 0,
    'HAVE_SYS_MKDEV_H': 0,
    'HAVE_SYS_MMAN_H': 1,
    'HAVE_SYS_MODEM_H': 0,
    'HAVE_SYS_NDIR_H': 0,
    'HAVE_SYS_PARAM_H': 1,
    'HAVE_SYS_PIDFD_H': 0,
    'HAVE_SYS_POLL_H': 1,
    'HAVE_SYS_RANDOM_H': 0,
    'HAVE_SYS_RESOURCE_H': 1,
    'HAVE_SYS_SELECT_H': 1,
    'HAVE_SYS_SENDFILE_H': 1,
    'HAVE_SYS_SOCKET_H': 1,
    'HAVE_SYS_SOUNDCARD_H': 1,
    'HAVE_SYS_STATVFS_H': 1,
    'HAVE_SYS_STAT_H': 1,
    'HAVE_SYS_SYSCALL_H': 1,
    'HAVE_SYS_SYSCTL_H': 1,
    'HAVE_SYS_SYSMACROS_H': 1,
    'HAVE_SYS_SYS_DOMAIN_H': 0,
    'HAVE_SYS_TERMIO_H': 0,
    'HAVE_SYS_TIMERFD_H': 1,
    'HAVE_SYS_TIMES_H': 1,
    'HAVE_SYS_TIME_H': 1,
    'HAVE_SYS_TYPES_H': 1,
    'HAVE_SYS_UIO_H': 1,
    'HAVE_SYS_UN_H': 1,
    'HAVE_SYS_UTSNAME_H': 1,
    'HAVE_SYS_WAIT_H': 1,
    'HAVE_SYS_XATTR_H': 1,
    'HAVE_TCGETPGRP': 1,
    'HAVE_TCSETPGRP': 1,
    'HAVE_TEMPNAM': 1,
    'HAVE_TERMIOS_H': 1,
    'HAVE_TERM_H': 1,
    'HAVE_TIMEGM': 1,
    'HAVE_TIMERFD_CREATE': 1,
    'HAVE_TIMES': 1,
    'HAVE_TMPFILE': 1,
    'HAVE_TMPNAM': 1,
    'HAVE_TMPNAM_R': 1,
    'HAVE_TM_ZONE': 1,
    'HAVE_TRUNCATE': 1,
    'HAVE_TTYNAME_R': 1,
    'HAVE_TZNAME': 0,
    'HAVE_UMASK': 1,
    'HAVE_UNAME': 1,
    'HAVE_UNISTD_H': 1,
    'HAVE_UNLINKAT': 1,
    'HAVE_UNLOCKPT': 1,
    'HAVE_UNSHARE': 1,
    'HAVE_USABLE_WCHAR_T': 0,
    'HAVE_UTIL_H': 0,
    'HAVE_UTIMENSAT': 1,
    'HAVE_UTIMES': 1,
    'HAVE_UTIME_H': 1,
    'HAVE_UTMP_H': 1,
    'HAVE_UT_NAMESIZE': 1,
    'HAVE_UUID_CREATE': 0,
    'HAVE_UUID_ENC_BE': 0,
    'HAVE_UUID_GENERATE_TIME_SAFE': 1,
    'HAVE_UUID_GENERATE_TIME_SAFE_STABLE_MAC': 0,
    'HAVE_UUID_H': 1,
    'HAVE_UUID_UUID_H': 0,
    'HAVE_VFORK': 1,
    'HAVE_WAIT': 1,
    'HAVE_WAIT3': 1,
    'HAVE_WAIT4': 1,
    'HAVE_WAITID': 1,
    'HAVE_WAITPID': 1,
    'HAVE_WCHAR_H': 1,
    'HAVE_WCSCOLL': 1,
    'HAVE_WCSFTIME': 1,
    'HAVE_WCSXFRM': 1,
    'HAVE_WMEMCMP': 1,
    'HAVE_WORKING_TZSET': 1,
    'HAVE_WRITEV': 1,
    'HAVE_ZDICT_H': 0,
    'HAVE_ZLIB_COPY': 1,
    'HAVE_ZLIB_H': 0,
    'HAVE_ZSTD_H': 0,
    'HAVE__GETPTY': 0,
    'HAVE___UINT128_T': 1,
    'HOSTRUNNER': '',
    'HOST_GNU_TYPE': 'x86_64-conda-linux-gnu',
    'INCLDIRSTOMAKE': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include/python3.14 /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include/python3.14',
    'INCLUDEDIR': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include',
    'INCLUDEPY': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include/python3.14',
    'INSTALL': '/usr/bin/install -c',
    'INSTALL_DATA': '/usr/bin/install -c -m 644',
    'INSTALL_MIMALLOC': 'yes',
    'INSTALL_PROGRAM': '/usr/bin/install -c',
    'INSTALL_SCRIPT': '/usr/bin/install -c',
    'INSTALL_SHARED': '/usr/bin/install -c -m 755',
    'INSTSONAME': 'libpython3.14.a',
    'IO_H': 'Modules/_io/_iomodule.h',
    'IO_OBJS': '\\',
    'IPHONEOS_DEPLOYMENT_TARGET': '',
    'JIT_DEPS': '\\',
    'LDCXXSHARED': 'x86_64-conda-linux-gnu-c++ -pthread -shared -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib',
    'LDFLAGS': '-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib',
    'LDLIBRARY': 'libpython3.14.a',
    'LDLIBRARYDIR': '',
    'LDSHARED': 'x86_64-conda-linux-gnu-gcc -pthread -shared -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib',
    'LDVERSION': '3.14',
    'LIBC': '',
    'LIBDEST': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib/python3.14',
    'LIBDIR': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib',
    'LIBEXPAT_A': 'Modules/expat/libexpat.a',
    'LIBEXPAT_CFLAGS': '-fno-strict-overflow -Wsign-compare -DNDEBUG -O2 -Wall -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -fno-merge-constants -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include       -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -fno-merge-constants -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include       -fno-semantic-interposition    -g -std=c11 -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wstrict-prototypes -Werror=implicit-function-declaration -fvisibility=hidden -D_Py_TIER2=3 -D_Py_JIT   -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include/internal -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include/internal/mimalloc -IObjects -IInclude -IPython -I. -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -fPIC',
    'LIBEXPAT_HEADERS': '\\',
    'LIBEXPAT_OBJS': '\\',
    'LIBHACL_BLAKE2_HEADERS': '\\',
    'LIBHACL_BLAKE2_LIB_SHARED': '\\',
    'LIBHACL_BLAKE2_LIB_STATIC': 'Modules/_hacl/libHacl_Hash_BLAKE2.a',
    'LIBHACL_BLAKE2_OBJS': '\\',
    'LIBHACL_BLAKE2_SIMD128_CFLAGS': '-msse -msse2 -msse3 -msse4.1 -msse4.2 -DHACL_CAN_COMPILE_VEC128',
    'LIBHACL_BLAKE2_SIMD128_OBJS': 'Modules/_hacl/Hacl_Hash_Blake2s_Simd128.o',
    'LIBHACL_BLAKE2_SIMD256_CFLAGS': '-mavx2 -DHACL_CAN_COMPILE_VEC256',
    'LIBHACL_BLAKE2_SIMD256_OBJS': 'Modules/_hacl/Hacl_Hash_Blake2b_Simd256.o',
    'LIBHACL_CFLAGS': '-I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/_hacl -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/_hacl/include -D_BSD_SOURCE -D_DEFAULT_SOURCE -DLINUX_NO_EXPLICIT_BZERO -fno-strict-overflow -Wsign-compare -DNDEBUG -O2 -Wall -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -fno-merge-constants -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include       -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -fno-merge-constants -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include       -fno-semantic-interposition    -g -std=c11 -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wstrict-prototypes -Werror=implicit-function-declaration -fvisibility=hidden -D_Py_TIER2=3 -D_Py_JIT   -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include/internal -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include/internal/mimalloc -IObjects -IInclude -IPython -I. -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -fPIC',
    'LIBHACL_HEADERS': '\\',
    'LIBHACL_HMAC_HEADERS': '\\',
    'LIBHACL_HMAC_LIB_SHARED': '\\',
    'LIBHACL_HMAC_LIB_STATIC': 'Modules/_hacl/libHacl_HMAC.a',
    'LIBHACL_HMAC_OBJS': '\\',
    'LIBHACL_LDFLAGS': '',
    'LIBHACL_MD5_HEADERS': '\\',
    'LIBHACL_MD5_LIB_SHARED': '\\',
    'LIBHACL_MD5_LIB_STATIC': 'Modules/_hacl/libHacl_Hash_MD5.a',
    'LIBHACL_MD5_OBJS': '\\',
    'LIBHACL_SHA1_HEADERS': '\\',
    'LIBHACL_SHA1_LIB_SHARED': '\\',
    'LIBHACL_SHA1_LIB_STATIC': 'Modules/_hacl/libHacl_Hash_SHA1.a',
    'LIBHACL_SHA1_OBJS': '\\',
    'LIBHACL_SHA2_HEADERS': '\\',
    'LIBHACL_SHA2_LIB_SHARED': '\\',
    'LIBHACL_SHA2_LIB_STATIC': 'Modules/_hacl/libHacl_Hash_SHA2.a',
    'LIBHACL_SHA2_OBJS': '\\',
    'LIBHACL_SHA3_HEADERS': '\\',
    'LIBHACL_SHA3_LIB_SHARED': '\\',
    'LIBHACL_SHA3_LIB_STATIC': 'Modules/_hacl/libHacl_Hash_SHA3.a',
    'LIBHACL_SHA3_OBJS': '\\',
    'LIBM': '-lm',
    'LIBMPDEC_A': 'Modules/_decimal/libmpdec/libmpdec.a',
    'LIBMPDEC_CFLAGS': '-I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -fno-strict-overflow -Wsign-compare -DNDEBUG -O2 -Wall -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -fno-merge-constants -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include       -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -fno-merge-constants -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include       -fno-semantic-interposition    -g -std=c11 -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wstrict-prototypes -Werror=implicit-function-declaration -fvisibility=hidden -D_Py_TIER2=3 -D_Py_JIT   -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include/internal -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include/internal/mimalloc -IObjects -IInclude -IPython -I. -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -fPIC',
    'LIBMPDEC_HEADERS': '\\',
    'LIBMPDEC_OBJS': '\\',
    'LIBOBJDIR': 'Python/',
    'LIBOBJS': '',
    'LIBPC': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib/pkgconfig',
    'LIBPL': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib/python3.14/config-3.14-x86_64-linux-gnu',
    'LIBPYTHON': '',
    'LIBRARY': 'libpython3.14.a',
    'LIBRARY_DEPS': 'libpython3.14.a',
    'LIBRARY_OBJS': '\\',
    'LIBRARY_OBJS_OMIT_FROZEN': '\\',
    'LIBS': '-lpthread -ldl  -lutil',
    'LIBSUBDIRS': 'asyncio \\',
    'LINKCC': 'x86_64-conda-linux-gnu-gcc -pthread',
    'LINKFORSHARED': '-Xlinker -export-dynamic',
    'LINK_PYTHON_DEPS': 'libpython3.14.a',
    'LINK_PYTHON_OBJS': '\\',
    'LIPO_32BIT_FLAGS': '',
    'LIPO_INTEL64_FLAGS': '',
    'LLVM_PROF_ERR': 'no',
    'LLVM_PROF_FILE': '',
    'LLVM_PROF_MERGER': 'true',
    'LN': 'ln',
    'LOCALMODLIBS': '-lm',
    'MACHDEP': 'linux',
    'MACHDEP_OBJS': '',
    'MACHDESTLIB': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib/python3.14',
    'MACOSX_DEPLOYMENT_TARGET': '',
    'MAJOR_IN_MKDEV': 0,
    'MAJOR_IN_SYSMACROS': 1,
    'MAKESETUP': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/makesetup',
    'MANDIR': '/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/share/man',
    'MIMALLOC_HEADERS': '\\',
    'MKDIR_P': '/usr/bin/mkdir -p',
    'MODBUILT_NAMES': 'array  _asyncio  _bisect  _csv  _heapq  _json  _lsprof  _pickle  _queue  _random  _remote_debugging  _struct  _interpreters  _interpchannels  _interpqueues  _zoneinfo  math  cmath  _statistics  _decimal  binascii  _bz2  _lzma  _zstd  zlib  readline  _md5  _sha1  _sha2  _sha3  _blake2  _hmac  pyexpat  _elementtree  _codecs_cn  _codecs_hk  _codecs_iso2022  _codecs_jp  _codecs_kr  _codecs_tw  _multibytecodec  unicodedata  fcntl  grp  mmap  _posixsubprocess  resource  select  _socket  syslog  termios  _posixshmem  _multiprocessing  _ctypes  _curses  _curses_panel  _sqlite3  _ssl  _hashlib  _uuid  _tkinter  xxsubtype  _xxtestfuzz  _testbuffer  _testinternalcapi  _testcapi  _testlimitedcapi  _testclinic  _testclinic_limited  _testimportmultiple  _testmultiphase  _testsinglephase  _ctypes_test  xxlimited  xxlimited_35  atexit  faulthandler  posix  _signal  _tracemalloc  _suggestions  _datetime  _codecs  _collections  errno  _io  itertools  _sre  _sysconfig  _thread  time  _types  _typing  _weakref  _abc  _functools  _locale  _opcode  _operator  _stat  _symtable  pwd',
    'MODDISABLED_NAMES': '',
    'MODLIBS': '-lm',
    'MODOBJS': 'Modules/atexitmodule.o  Modules/faulthandler.o  Modules/posixmodule.o  Modules/signalmodule.o  Modules/_tracemalloc.o  Modules/_suggestions.o  Modules/_datetimemodule.o  Modules/_codecsmodule.o  Modules/_collectionsmodule.o  Modules/errnomodule.o  Modules/_io/_iomodule.o Modules/_io/iobase.o Modules/_io/fileio.o Modules/_io/bytesio.o Modules/_io/bufferedio.o Modules/_io/textio.o Modules/_io/stringio.o  Modules/itertoolsmodule.o  Modules/_sre/sre.o  Modules/_sysconfig.o  Modules/_threadmodule.o  Modules/timemodule.o  Modules/_typesmodule.o  Modules/_typingmodule.o  Modules/_weakref.o  Modules/_abc.o  Modules/_functoolsmodule.o  Modules/_localemodule.o  Modules/_opcode.o  Modules/_operator.o  Modules/_stat.o  Modules/symtablemodule.o  Modules/pwdmodule.o',
    'MODSHARED_NAMES': 'array _asyncio _bisect _csv _heapq _json _lsprof _pickle _queue _random _remote_debugging _struct _interpreters _interpchannels _interpqueues _zoneinfo math cmath _statistics _decimal binascii _bz2 _lzma _zstd zlib readline _md5 _sha1 _sha2 _sha3 _blake2 _hmac pyexpat _elementtree _codecs_cn _codecs_hk _codecs_iso2022 _codecs_jp _codecs_kr _codecs_tw _multibytecodec unicodedata fcntl grp mmap _posixsubprocess resource select _socket syslog termios _posixshmem _multiprocessing _ctypes _curses _curses_panel _sqlite3 _ssl _hashlib _uuid _tkinter xxsubtype _xxtestfuzz _testbuffer _testinternalcapi _testcapi _testlimitedcapi _testclinic _testclinic_limited _testimportmultiple _testmultiphase _testsinglephase _ctypes_test xxlimited xxlimited_35',
    'MODULE_ARRAY_STATE': 'yes',
    'MODULE_ATEXIT_LDFLAGS': '',
    'MODULE_BINASCII_CFLAGS': '-DUSE_ZLIB_CRC32 -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include',
    'MODULE_BINASCII_LDFLAGS': '-L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -lz',
    'MODULE_BINASCII_STATE': 'yes',
    'MODULE_CMATH_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/_math.h',
    'MODULE_CMATH_LDFLAGS': '-lm',
    'MODULE_CMATH_STATE': 'yes',
    'MODULE_DEPS_SHARED': 'Modules/config.c',
    'MODULE_DEPS_STATIC': 'Modules/config.c',
    'MODULE_ERRNO_LDFLAGS': '',
    'MODULE_FAULTHANDLER_LDFLAGS': '',
    'MODULE_FCNTL_LDFLAGS': '',
    'MODULE_FCNTL_STATE': 'yes',
    'MODULE_GRP_STATE': 'yes',
    'MODULE_ITERTOOLS_LDFLAGS': '',
    'MODULE_LDFLAGS_SHARED': '$(if ,libpython3.14.a)',
    'MODULE_MATH_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/_math.h',
    'MODULE_MATH_LDFLAGS': '-lm',
    'MODULE_MATH_STATE': 'yes',
    'MODULE_MMAP_STATE': 'yes',
    'MODULE_OBJS': '\\',
    'MODULE_POSIX_LDFLAGS': '',
    'MODULE_PWD_LDFLAGS': '',
    'MODULE_PWD_STATE': 'yes',
    'MODULE_PYEXPAT_CFLAGS': '',
    'MODULE_PYEXPAT_DEPS': '',
    'MODULE_PYEXPAT_LDFLAGS': '-lexpat',
    'MODULE_PYEXPAT_STATE': 'yes',
    'MODULE_READLINE_CFLAGS': '-D_GNU_SOURCE -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include/ncurses -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include',
    'MODULE_READLINE_LDFLAGS': '-L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -lreadline',
    'MODULE_READLINE_STATE': 'yes',
    'MODULE_RESOURCE_STATE': 'yes',
    'MODULE_SELECT_STATE': 'yes',
    'MODULE_SYSLOG_STATE': 'yes',
    'MODULE_TERMIOS_STATE': 'yes',
    'MODULE_TIME_LDFLAGS': '',
    'MODULE_TIME_STATE': 'yes',
    'MODULE_UNICODEDATA_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/unicodedata_db.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/unicodename_db.h',
    'MODULE_UNICODEDATA_STATE': 'yes',
    'MODULE_XXLIMITED_35_STATE': 'yes',
    'MODULE_XXLIMITED_STATE': 'yes',
    'MODULE_XXSUBTYPE_STATE': 'yes',
    'MODULE_ZLIB_CFLAGS': '-I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include',
    'MODULE_ZLIB_LDFLAGS': '-L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -lz',
    'MODULE_ZLIB_STATE': 'yes',
    'MODULE__ABC_LDFLAGS': '',
    'MODULE__ASYNCIO_STATE': 'yes',
    'MODULE__BISECT_STATE': 'yes',
    'MODULE__BLAKE2_CFLAGS': '-I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/_hacl -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/_hacl/include -D_BSD_SOURCE -D_DEFAULT_SOURCE -DLINUX_NO_EXPLICIT_BZERO -fno-strict-overflow -Wsign-compare -DNDEBUG -O2 -Wall -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -fno-merge-constants -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include       -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -fno-merge-constants -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include       -fno-semantic-interposition    -g -std=c11 -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wstrict-prototypes -Werror=implicit-function-declaration -fvisibility=hidden -D_Py_TIER2=3 -D_Py_JIT   -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include/internal -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include/internal/mimalloc -IObjects -IInclude -IPython -I. -I/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -fPIC',
    'MODULE__BLAKE2_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/hashlib.h \\ \\',
    'MODULE__BLAKE2_LDEPS': '\\',
    'MODULE__BLAKE2_LDFLAGS': '\\',
    'MODULE__BLAKE2_STATE': 'yes',
    'MODULE__BZ2_CFLAGS': '',
    'MODULE__BZ2_LDFLAGS': '-lbz2',
    'MODULE__BZ2_STATE': 'yes',
    'MODULE__CODECS_CN_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/mappings_cn.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/multibytecodec.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/cjkcodecs.h',
    'MODULE__CODECS_CN_STATE': 'yes',
    'MODULE__CODECS_HK_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/mappings_hk.h  /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/multibytecodec.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/cjkcodecs.h',
    'MODULE__CODECS_HK_STATE': 'yes',
    'MODULE__CODECS_ISO2022_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/mappings_jisx0213_pair.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/alg_jisx0201.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/emu_jisx0213_2000.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/multibytecodec.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/cjkcodecs.h',
    'MODULE__CODECS_ISO2022_STATE': 'yes',
    'MODULE__CODECS_JP_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/mappings_jisx0213_pair.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/alg_jisx0201.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/emu_jisx0213_2000.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/mappings_jp.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/multibytecodec.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/cjkcodecs.h',
    'MODULE__CODECS_JP_STATE': 'yes',
    'MODULE__CODECS_KR_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/mappings_kr.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/multibytecodec.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/cjkcodecs.h',
    'MODULE__CODECS_KR_STATE': 'yes',
    'MODULE__CODECS_LDFLAGS': '',
    'MODULE__CODECS_TW_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/mappings_tw.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/multibytecodec.h /home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/cjkcodecs/cjkcodecs.h',
    'MODULE__CODECS_TW_STATE': 'yes',
    'MODULE__COLLECTIONS_LDFLAGS': '',
    'MODULE__CSV_STATE': 'yes',
    'MODULE__CTYPES_CFLAGS': '-fno-strict-overflow -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include',
    'MODULE__CTYPES_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/_ctypes/ctypes.h',
    'MODULE__CTYPES_LDFLAGS': '-L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -lffi -ldl',
    'MODULE__CTYPES_MALLOC_CLOSURE': '',
    'MODULE__CTYPES_STATE': 'yes',
    'MODULE__CTYPES_TEST_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/_ctypes/_ctypes_test_generated.c.h',
    'MODULE__CTYPES_TEST_LDFLAGS': '-lm',
    'MODULE__CTYPES_TEST_STATE': 'yes',
    'MODULE__CURSES_CFLAGS': '-D_GNU_SOURCE -DNCURSES_WIDECHAR -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include/ncursesw -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include',
    'MODULE__CURSES_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include/py_curses.h',
    'MODULE__CURSES_LDFLAGS': '-L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-O2 -Wl,--sort-common -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -lncursesw -ltinfow',
    'MODULE__CURSES_PANEL_CFLAGS': '-D_GNU_SOURCE -DNCURSES_WIDECHAR -D_GNU_SOURCE -DNCURSES_WIDECHAR -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include/ncursesw -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include/ncursesw -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include -D_GNU_SOURCE -DNCURSES_WIDECHAR -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include/ncursesw -I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include',
    'MODULE__CURSES_PANEL_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include/py_curses.h',
    'MODULE__CURSES_PANEL_LDFLAGS': '-L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-O2 -Wl,--sort-common -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -lpanelw -L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-O2 -Wl,--sort-common -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -Wl,-rpath-link,/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -lncursesw -ltinfow',
    'MODULE__CURSES_PANEL_STATE': 'yes',
    'MODULE__CURSES_STATE': 'yes',
    'MODULE__DATETIME_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Include/datetime.h',
    'MODULE__DATETIME_LDFLAGS': '-lm',
    'MODULE__DATETIME_STATE': 'yes',
    'MODULE__DBM_STATE': 'missing',
    'MODULE__DECIMAL_CFLAGS': '-I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include',
    'MODULE__DECIMAL_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/_decimal/docstrings.h',
    'MODULE__DECIMAL_LDFLAGS': '-L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib -lmpdec -lm',
    'MODULE__DECIMAL_STATE': 'yes',
    'MODULE__ELEMENTTREE_CFLAGS': '',
    'MODULE__ELEMENTTREE_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/pyexpat.c',
    'MODULE__ELEMENTTREE_STATE': 'yes',
    'MODULE__FUNCTOOLS_LDFLAGS': '',
    'MODULE__GDBM_STATE': 'missing',
    'MODULE__HASHLIB_CFLAGS': '-I/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/include',
    'MODULE__HASHLIB_DEPS': '/home/conda/feedstock_root/build_artifacts/python-split_1787152744847/work/Modules/hashlib.h',
    'MODULE__HASHLIB_LDFLAGS': '-L/home/octoopt/workspace/projects/lakoreai/tokenzip/.pixi/envs/default/lib   -lcrypto',
    'MODULE__HASHLIB_STATE': 'yes',
    'MODULE__HEAPQ_STATE': 'yes',
    'MODULE__HMAC_CFLAGS': '-I/home/conda/feedstock_root/build_artifacts/python-spli