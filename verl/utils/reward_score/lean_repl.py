import subprocess
import json
import os
import signal
import queue
import time
import threading
from typing import Optional, Dict, Any, Tuple

# Default configuration
_LEAN_DIR = os.getenv("LEAN_DIR", "/data2/haikang/projects/cloned/kimina-lean-server/workspace/math_learning")
# Restart worker after this many requests (since noSnapshot doesn't return env)
_DEFAULT_RESTART_EVERY = 256


def clean_solution_str(solution_str: str) -> str:
    """Remove markdown fences / language tags and surrounding whitespace."""
    solution_str = solution_str.strip()
    
    # for chat format
    if "```lean4\n" in solution_str:
        solution_str = solution_str[solution_str.rfind("```lean4\n") + len("```lean4\n"):]
        solution_str = solution_str[solution_str.find("theorem", 0):]
        solution_str = solution_str[solution_str.find(":= by", 0)+len(":= by"):]
        
        if solution_str.endswith("```"):
            solution_str = solution_str[:-3]
        return solution_str.rstrip()
    
    # for str format
    else:
        # Remove markdown code blocks if present
        print(f"🟥 ```lean4\n not found in solution_str: {solution_str}")
        
        if solution_str.startswith("```lean"):
            solution_str = solution_str[7:]
        elif solution_str.startswith("```"):
            solution_str = solution_str[3:]

        if solution_str.endswith("```"):
            solution_str = solution_str[:-3]

        return solution_str.rstrip()


class LeanREPLWorker:
    """
    Persistent Lean REPL worker with thread-safe lifecycle management.

    Mirrors the robust behavior of the standalone verification script:
    - Starts `lake exe repl` and loads imports once.
    - Communicates via JSON over stdin/stdout with multi-line responses.
    - Enforces per-request timeouts.
    - Tracks request count and lets the pool recycle when needed.

    The public `verify` method returns a compact dict with:
      { "score": float, "acc": bool, "info": str }
    suitable for reward-model usage.
    """

    def __init__(
        self,
        lean_dir: str,
        worker_id: int,
        restart_every: int = _DEFAULT_RESTART_EVERY,
        verbose_profile: bool = False,
    ):
        self.lean_dir = lean_dir
        self.worker_id = worker_id
        self.proc: Optional[subprocess.Popen] = None
        # Base environment after imports are loaded
        self.base_env: Optional[int] = None
        self.request_count: int = 0  # Track number of verification requests
        self.restart_every = restart_every
        self.verbose_profile = verbose_profile
        self._lifecycle_lock = threading.Lock()  # Serialize start/kill operations

    def start(self):
        """Starts the 'repl' process and loads imports. BLOCKING (takes ~10s). Thread-safe."""
        with self._lifecycle_lock:
            # Kill existing process and get its PID
            killed_pid = self._kill_unlocked()
            
            # Wait for the process to actually die (verify PID is gone)
            if killed_pid is not None:
                if not self._wait_for_process_death(killed_pid, timeout=5.0):
                    print(f"⚠️ Worker {self.worker_id} (PID {killed_pid}) did not terminate within timeout, but proceeding anyway...")
                else:
                    print(f"✅ Worker {self.worker_id} (PID {killed_pid}) fully terminated")
            
            # Double-check: ensure no process object exists before starting new one
            if self.proc is not None:
                raise RuntimeError(f"Worker {self.worker_id}: Process object still exists after kill - this should not happen!")
            
            cmd = ["lake", "exe", "repl"]
            print(f"🟢 Starting REPL worker {self.worker_id}")
            # Create process in new process group so we can kill entire process tree
            self.proc = subprocess.Popen(
                cmd,
                cwd=self.lean_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True  # Create new process group
            )

        header = (
            "import Mathlib\n"
            "import Aesop\n"
            "set_option maxHeartbeats 0\n"
            "open BigOperators Real Nat Topology Rat"
        )
        try:
            # Use a safe timeout for imports (e.g. 180s)
            res, _ = self._send_command({"cmd": header}, timeout=180.0)
            
            if "env" not in res:
                raise RuntimeError(f"Worker {self.worker_id}: Failed to load imports. Response: {res}")
            
            self.base_env = res["env"]
            self.request_count = 0  # Reset request counter on restart
            print(f"✅ Worker {self.worker_id} ready (env={self.base_env})")
            
        except Exception as e:
            print(f"🟥 Worker {self.worker_id} start error: {e}")
            with self._lifecycle_lock:
                self._kill_unlocked()
            raise

    def verify(self, code: str) -> Dict[str, Any]:
        """
        Verifies Lean code starting from the pre-loaded base environment.

        Returns a reward-style dict:
            { "score": 1.0/-1.0, "acc": bool, "info": str }
        """
        start_time = time.perf_counter()

        # Failsafe: if worker is dead (e.g. killed externally), mark for recycling
        # Don't restart here - let the pool handle it to avoid concurrent starts
        if self.proc is None:
            return {"score": -1.0, "acc": False, "info": "worker_not_ready"}

        # Fast fail on textual "sorry"
        if "sorry" in code:
            return {"score": -1.0, "acc": False, "info": "contains_sorry (text scan)"}

        req = {"cmd": code, "env": self.base_env, "noSnapshot": True}
        
        try:
            # 3-minute timeout per proof
            res, _ = self._send_command(req, timeout=180.0)
            
            # Increment request counter (noSnapshot doesn't return env, so we track manually)
            self.request_count += 1
                
        except TimeoutError:
            # TIMEOUT HANDLER: Kill process and mark for recycling
            # Don't restart here - let the pool handle it to prevent concurrent processes
            with self._lifecycle_lock:
                self._kill_unlocked()
            self.request_count = 0  # Reset counter on restart
            return {"score": -1.0, "acc": False, "info": "execution_timeout"}

        except Exception as e:
            # Mark as dead so Pool knows to recycle it
            with self._lifecycle_lock:
                self._kill_unlocked()
            self.request_count = 0  # Reset counter on crash
            return {"score": -1.0, "acc": False, "info": f"repl_crash: {str(e)}"}

        # Parse diagnostics
        messages = res.get("messages", [])
        errors = [m for m in messages if m.get("severity") == "error"]
        if errors:
            err_msg = errors[0].get("data", "unknown error").replace("\n", " ")
            return {"score": -1.0, "acc": False, "info": f"error: {err_msg}"}

        # Guard against "sorry" in diagnostic messages.
        for m in messages:
            if "sorry" in m.get("data", ""):
                return {"score": -1.0, "acc": False, "info": "contains_sorry"}

        return {"score": 1.0, "acc": True, "info": "verified"}

    def _send_command(self, cmd_dict: Dict, timeout: float = 180.0) -> Tuple[Dict, str]:
        """
        Sends command and reads response with a timeout using threading.
        Returns (parsed_result, raw_output) tuple.
        """
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise EOFError("REPL process is not running")

        json_line = json.dumps(cmd_dict) + "\n\n"
        
        try:
            self.proc.stdin.write(json_line)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise EOFError("REPL closed connection unexpectedly (Broken Pipe)")

        # Use a queue to communicate between reader thread and main thread
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def read_response():
            try:
                lines = []
                while True: 
                    line = self.proc.stdout.readline()
                    if not line: 
                        exception_queue.put(
                            EOFError("REPL closed connection unexpectedly")
                        )
                        return

                    # Blank line separates messages
                    if not line.strip():
                        if not lines:
                            # Skip leading blank lines
                            continue
                        else:
                            break

                    lines.append(line)

                full_output = "".join(lines)

                def parse_fallback(raw: str):
                    """
                    Try to parse line-by-line (NDJSON-style) and return the
                    last successfully parsed JSON object. This is robust to
                    Lean emitting multiple JSON objects back-to-back.
                    """
                    last_obj = None
                    for ln in raw.splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            last_obj = json.loads(ln)
                        except json.JSONDecodeError:
                            continue
                    if last_obj is not None:
                        return last_obj
                    raise json.JSONDecodeError("fallback parse failed", raw, 0)

                try:
                    result = json.loads(full_output)
                except json.JSONDecodeError: 
                    try:
                        result = parse_fallback(full_output)
                        self.base_env = 0
                    except json.JSONDecodeError as e:
                        print(
                            f"🟥 Failed to parse JSON from REPL. Raw output:\n{full_output}"
                        )
                        exception_queue.put(e)
                        return
                
                result_queue.put((result, full_output))
            except Exception as e:
                exception_queue.put(e)
        
        # Make thread non-daemon and store reference
        reader_thread = threading.Thread(target=read_response, daemon=False)
        reader_thread.start()
        
        # Store thread reference so we can join it later
        self._active_reader_thread = reader_thread
        
        # Wait for result with timeout
        try:
            result, raw_output = result_queue.get(timeout=timeout)
            return result, raw_output
        except queue.Empty:
            # Timeout occurred
            raise TimeoutError("REPL timed out")
        finally:
            # Check if there was an exception in the reader thread
            if not exception_queue.empty():
                raise exception_queue.get()

    def kill(self):
        """Properly terminate the REPL process and clean up resources. Thread-safe. Returns PID if process was killed, None otherwise."""
        with self._lifecycle_lock:
            return self._kill_unlocked()
    
    def _kill_unlocked(self):
        """Internal kill implementation (assumes lock is already held). Returns PID if process was killed, None otherwise."""
        if self.proc is None:
            return None  # Already killed/not started
        
        proc_to_kill = self.proc
        pid = proc_to_kill.pid  # Store PID for verification
        pgid = None  # Process group ID
        
        try:
            # Try to get process group ID to kill entire process tree
            try:
                pgid = os.getpgid(pid)
            except (OSError, ProcessLookupError):
                pass  # Process might not exist or not have a group
            
            self.proc = None  # Set to None to mark as dead (but PID might still be alive)
        except Exception:
            self.proc = None
            return None
        
        try: 
            # Close stdin first to signal the process to exit gracefully
            if proc_to_kill.stdin:
                try:
                    proc_to_kill.stdin.close()
                except:
                    pass  # Might already be closed
            
            # Try graceful termination first (SIGTERM) - kill process group if available
            try:
                if pgid is not None:
                    try:
                        os.killpg(pgid, signal.SIGTERM)
                    except (OSError, ProcessLookupError):
                        # Fallback to killing just the process
                        proc_to_kill.terminate()
                else:
                    proc_to_kill.terminate()
                
                # Wait up to 2 seconds for graceful shutdown
                try:
                    proc_to_kill.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass  # Will force kill below
            except (OSError, ProcessLookupError):
                # Process might already be dead
                pass
            
            # Force kill if process still exists
            try:
                # Check if process still exists
                os.kill(pid, 0)
                # Process still exists, force kill it and its group
                if pgid is not None:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass
                try:
                    proc_to_kill.kill()
                except (OSError, ProcessLookupError):
                    pass
                
                # Wait for process to die (with timeout)
                try:
                    proc_to_kill.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass  # Process might be stuck, but we tried
            except (OSError, ProcessLookupError):
                # Process doesn't exist - good
                pass
            
            # Close remaining pipes
            try:
                if proc_to_kill.stdout:
                    proc_to_kill.stdout.close()
                if proc_to_kill.stderr:
                    proc_to_kill.stderr.close()
            except:
                pass
            
            # Return PID so caller can verify it's actually dead
            return pid
                        
        except Exception as e:
            print(f"[Pool Kill Error] Worker {self.worker_id}: {e}")
            # Force kill as last resort - try everything
            try:
                if pgid is not None:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except:
                        pass
                try:
                    os.kill(pid, signal.SIGKILL)
                except:
                    pass
                try:
                    proc_to_kill.kill()
                except:
                    pass
                try:
                    proc_to_kill.wait(timeout=1.0)
                except:
                    pass
                # Close pipes
                try:
                    if proc_to_kill.stdout:
                        proc_to_kill.stdout.close()
                    if proc_to_kill.stderr:
                        proc_to_kill.stderr.close()
                except:
                    pass
            except:
                pass
            return pid
    
    def _wait_for_process_death(self, pid: int, timeout: float = 5.0) -> bool:
        """Wait for a process to actually die by checking if PID exists. Returns True if process is dead, False if timeout."""
        start_time = time.time()
        check_interval = 0.1
        
        # First, try to get process group if it still exists
        pgid = None
        try:
            pgid = os.getpgid(pid)
        except (OSError, ProcessLookupError):
            pass  # Process or group doesn't exist
        
        while time.time() - start_time < timeout:
            try:
                os.kill(pid, 0)  # Check if process exists (doesn't kill, just checks)
                # Process still exists
                time.sleep(check_interval)
            except OSError:
                # Process doesn't exist - it's dead!
                return True
        
        # Timeout - process might still be around, try aggressive kill
        try:
            # Final aggressive kill attempt - kill process group if available
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
            try:
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            
            # Wait a bit and check again
            time.sleep(0.5)
            try:
                os.kill(pid, 0)  # Check if process still exists
                # Still exists after SIGKILL - this is bad but we've done all we can
                print(f"⚠️ Worker {self.worker_id} (PID {pid}) still exists after aggressive kill")
                return False
            except OSError:
                # Process is now dead
                return True
        except Exception as e:
            print(f"⚠️ Error during final kill attempt for PID {pid}: {e}")
            # Assume it's dead if we can't check
            return True

    def needs_recycle(self) -> bool:
        """
        Check if worker needs recycling based on request count or dead process.
        Since noSnapshot=True doesn't return env IDs, we track requests manually.
        Also recycle if the process is dead (None) or not actually running.
        """
        if self.proc is None:
            return True
        
        # Verify process is actually alive
        try:
            os.kill(self.proc.pid, 0)  # Check if process exists (doesn't kill, just checks)
        except OSError:
            # Process doesn't exist - mark as dead and needs recycle
            self.proc = None
            return True
        
        return self.request_count >= self.restart_every


class LeanREPLPool:
    """
    Thread-safe pool of Lean REPL workers.

    Workers are started (and restarted) in background threads so that the
    main training loop only blocks when acquiring workers from the queue.
    """

    def __init__(
        self,
        num_workers: int,
        lean_dir: str,
        restart_every: int = _DEFAULT_RESTART_EVERY,
        verbose_profile: bool = False,
    ):
        self.queue: "queue.Queue[LeanREPLWorker]" = queue.Queue()
        self.workers = []
        self.lean_dir = lean_dir
        self.num_workers = num_workers
        self._recycling_locks = {}  # Track which workers are currently being recycled
        
        print(f"[Pool] Launching {num_workers} workers (Background startup)...")
        for i in range(num_workers):
            w = LeanREPLWorker(
                lean_dir,
                i,
                restart_every=restart_every,
                verbose_profile=verbose_profile,
            )
            self.workers.append(w)
            self._recycling_locks[w.worker_id] = threading.Lock()
            threading.Thread(
                target=self._recycle_worker, args=(w,), daemon=True
            ).start()

    def get_worker(self) -> "LeanREPLWorker":
        return self.queue.get(block=True)

    def return_worker(self, worker: "LeanREPLWorker"):
        """
        Returns worker to pool. 
        Checks the request count to decide if background recycling is needed.
        """
        if worker.needs_recycle():
            # Trigger background restart (recycle_worker will handle lock to prevent duplicates)
            threading.Thread(
                target=self._recycle_worker, args=(worker,), daemon=True
            ).start()
        else:
            # Healthy worker, return to queue immediately
            self.queue.put(worker)

    def _recycle_worker(self, worker: "LeanREPLWorker"):
        """Recycle a worker, ensuring only one recycling operation per worker at a time."""
        recycling_lock = self._recycling_locks.get(worker.worker_id)
        if not recycling_lock:
            # Fallback if lock missing
            recycling_lock = threading.Lock()
            self._recycling_locks[worker.worker_id] = recycling_lock
        
        # Ensure only one recycling thread per worker
        if not recycling_lock.acquire(blocking=False):
            # Another thread is already recycling this worker
            return
        
        try:
            print(f"♻️ [Pool] Recycling worker {worker.worker_id}")
            
            # Verify process count before recycling
            active_count = self.count_active_processes()
            if active_count >= self.num_workers:
                print(f"⚠️ [Pool] {active_count} processes active (max: {self.num_workers}), waiting before recycling worker {worker.worker_id}...")
                # Wait a bit to let some processes finish
                time.sleep(1.0)
            
            worker.start()  # Blocks for ~10s, kills old process first
            
            # Verify we didn't exceed process limit
            active_count = self.count_active_processes()
            if active_count > self.num_workers:
                print(f"⚠️ [Pool] WARNING: {active_count} processes active (should be <= {self.num_workers})")
            
            self.queue.put(worker)
            print(f"✅ [Pool] Worker {worker.worker_id} recycled successfully")
        except Exception as e:
            print(f"[Pool] Failed to recycle worker {worker.worker_id}: {e}")
            # Don't retry immediately to avoid rapid retry loops
            time.sleep(5)
        finally:
            recycling_lock.release()

    def close(self):
        """Kill all REPL workers and clean up resources."""
        print(f"[Pool] Shutting down {len(self.workers)} REPL workers...")
        
        # Kill all workers and collect their PIDs
        killed_pids = []
        for w in self.workers:
            killed_pid = w.kill()
            if killed_pid is not None:
                killed_pids.append(killed_pid)
        
        # Wait for all processes to actually die
        if killed_pids:
            print(f"[Pool] Waiting for {len(killed_pids)} processes to terminate...")
            for pid in killed_pids:
                # Wait up to 3 seconds per process
                max_wait = 3.0
                start_time = time.time()
                while time.time() - start_time < max_wait:
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        time.sleep(0.1)
                    except OSError:
                        # Process is dead
                        break
                else:
                    # Final kill attempt
                    try:
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except:
                            pass
        
        # Give background threads a moment to finish
        time.sleep(0.5)
        
        print(f"[Pool] Closed {len(self.workers)} REPL workers")
        
    def count_active_processes(self):
        """Count how many REPL processes are actually running."""
        count = 0
        for w in self.workers:
            if w.proc is not None:
                try:
                    # Check if process is actually alive
                    os.kill(w.proc.pid, 0)
                    count += 1
                except OSError:
                    # Process doesn't exist
                    w.proc = None
        return count


# Global reference for the pool to persist across verify calls
_GLOBAL_POOL: Optional[LeanREPLPool] = None


def get_or_init_pool(num_workers: int, lean_dir: str) -> LeanREPLPool:
    """
    Lazily create a global LeanREPLPool.

    This keeps the REPL workers alive across reward-manager calls so that
    we do not pay import overhead repeatedly.
    """
    global _GLOBAL_POOL
    if _GLOBAL_POOL is None:
        _GLOBAL_POOL = LeanREPLPool(num_workers, lean_dir)
    return _GLOBAL_POOL


def compute_score_repl(
    solution_str: str,
    pool: LeanREPLPool,
    ground_truth: Optional[str] = None,
    extra_info: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Bridge function to verification.

    Acquires a worker from the pool, verifies the composed Lean code,
    and returns the worker to the pool.
    """
    if extra_info is None:
        extra_info = {}

    # Statement (if provided) should *not* include imports, since imports
    # are already handled in the worker header.
    clean_stmt = extra_info.get("statement", "") or ""

    clean_sol = clean_solution_str(solution_str)
    
    theorem_code = clean_stmt + "\n" + clean_sol if clean_stmt else clean_sol
    
    print(f"🟢 cleaned theorem_code: {theorem_code}")

    worker = pool.get_worker()
    try:
        result = worker.verify(theorem_code)
        return result
    finally:
        pool.return_worker(worker)
