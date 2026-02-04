# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reward scoring function for HumanEval code generation benchmark.

HumanEval tests are assertion-based, so we execute the generated code
along with the test cases and check if all assertions pass.
"""

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import traceback


def _unsafe_execute(code: str, timeout: float, result_queue):
    """Execute code in a sandboxed environment."""
    with _create_tempdir():
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        _reliability_guard()

        try:
            exec_globals = {}
            with _swallow_io():
                with _time_limit(timeout):
                    exec(compile(code, "<string>", "exec"), exec_globals)
            result_queue.put("passed")
        except TimeoutException:
            result_queue.put("timed out")
        except BaseException as e:
            result_queue.put(f"failed: {type(e).__name__}: {e}")

        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def check_correctness(
    completion: str,
    prompt: str,
    test: str,
    entry_point: str,
    timeout: float = 5.0,
) -> dict:
    """
    Check correctness of generated code by running test cases.

    Args:
        completion: The generated code (should complete the function)
        prompt: The function signature + docstring
        test: The test cases (assert statements)
        entry_point: The function name
        timeout: Execution timeout in seconds

    Returns:
        dict with 'passed' (bool) and 'result' (str) keys
    """
    # Extract code from markdown if present
    if "```python" in completion:
        completion = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]

    # Build the full code to execute
    # The completion should include the function, we prepend the prompt if needed
    if entry_point not in completion:
        # Model didn't include the function signature, prepend it
        full_code = prompt + completion
    else:
        # Model included the full function
        full_code = completion

    # Add test cases
    check_program = full_code + "\n" + test

    # Add the check() call if the test defines it
    if f"check({entry_point})" not in test and "def check(" in test:
        check_program += f"\ncheck({entry_point})"

    # Execute in a separate process for safety
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    p = multiprocessing.Process(
        target=_unsafe_execute,
        args=(check_program, timeout, result_queue),
    )
    p.start()
    p.join(timeout=timeout + 1)

    if p.is_alive():
        p.kill()
        p.join()
        result = "timed out"
    elif result_queue.empty():
        result = "failed: no result"
    else:
        result = result_queue.get()

    passed = result == "passed"
    return {"passed": passed, "result": result}


def compute_score(solution_str: str, ground_truth: dict) -> float:
    """
    Compute reward score for HumanEval.

    Args:
        solution_str: The model's generated solution
        ground_truth: Dict containing 'prompt', 'test', 'entry_point', 'task_id'

    Returns:
        1.0 if all tests pass, 0.0 otherwise
    """
    try:
        result = check_correctness(
            completion=solution_str,
            prompt=ground_truth["prompt"],
            test=ground_truth["test"],
            entry_point=ground_truth["entry_point"],
            timeout=5.0,
        )
        return 1.0 if result["passed"] else 0.0
    except Exception as e:
        traceback.print_exc()
        return 0.0


# Helper classes and functions for sandboxed execution
# Adapted from OpenAI's human-eval repository

class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds: float):
    if platform.system() == "Windows":
        # Windows doesn't support SIGALRM
        yield
        return

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _swallow_io():
    stream = _WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            yield


@contextlib.contextmanager
def _create_tempdir():
    import tempfile
    with tempfile.TemporaryDirectory() as dirname:
        with _chdir(dirname):
            yield dirname


@contextlib.contextmanager
def _chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as e:
        raise e
    finally:
        os.chdir(cwd)


class _WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


def _reliability_guard(maximum_memory_bytes: int = None):
    """
    Disables various destructive functions to prevent the generated code
    from interfering with the test environment.
    """
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None

    __builtins__["help"] = None

    import sys
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
