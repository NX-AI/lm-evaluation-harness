import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
import sys

def check_correctness(check_program, timeout=3):
    """
    Evaluates the functional correctness of a completion.
    """
    manager = multiprocessing.Manager()
    try:
        result = manager.list()

        p = multiprocessing.Process(target=unsafe_execute, args=(check_program, result, timeout))
        p.start()
        p.join(timeout=timeout + 1)
        
        if p.is_alive():
            p.kill()
            p.join()

        if not result:
            result.append("timed out")

        # In CRUXEval, "passed" is appended if exec() finishes without error.
        return result[0] == "passed"
    finally:
        manager.shutdown()


def unsafe_execute(check_program, result, timeout):
    with create_tempdir():
        # Import/Disable functionalities that can make destructive changes
        import os
        import shutil
        
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        reliability_guard()

        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except Exception as e:
            # Capturing the error is useful for debugging, though we return False
            result.append(f"failed: {e}")

        # Restore cleanup functions
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # Use standard SIGALRM
    if hasattr(signal, "setitimer"):
        prev = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev)
    else:
        # Fallback for non-Unix systems (though execution is risky on Windows)
        yield


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""
    def read(self, *args, **kwargs): raise OSError
    def readline(self, *args, **kwargs): raise OSError
    def readlines(self, *args, **kwargs): raise OSError
    def readable(self, *args, **kwargs): return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes=None):
    """
    Disables destructive functions.
    WARNING: Not a full security sandbox. Run in a container if possible.
    """
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()
    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Disable os functions
    for attr in ['kill', 'system', 'putenv', 'remove', 'removedirs', 'rmdir', 
                 'fchdir', 'setuid', 'fork', 'forkpty', 'killpg', 'rename', 
                 'renames', 'truncate', 'replace', 'unlink', 'fchmod', 'fchown', 
                 'chmod', 'chown', 'chroot', 'lchflags', 'lchmod', 'lchown', 
                 'getcwd', 'chdir']:
        if hasattr(os, attr):
            setattr(os, attr, None)

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None 

    import builtins
    builtins.help = None
    
    # Disable common modules that might cause issues
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None