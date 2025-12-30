import subprocess
import os
import re
import tempfile
from typing import Optional

# Default configuration
_LEAN_DIR = os.getenv("LEAN_DIR", "/data2/haikang/projects/lean-prover/lean-project")
_TIMEOUT = int(os.getenv("LEAN_TIMEOUT", "300"))

def clean_solution_str(solution_str: str) -> str:
    """Remove trailing triple backticks if present."""
    solution_str = solution_str.strip()
    if solution_str.endswith("```"):
        solution_str = solution_str[:-3].strip()
    if solution_str.startswith("```") and solution_str.endswith("```"):
        lines = solution_str.split("\n")
        if len(lines) > 2 and lines[0].startswith("```"):
            solution_str = "\n".join(lines[1:-1])
    return solution_str

def combine_statement_and_solution(statement_with_header: str, solution_str: str) -> str:
    solution_str = clean_solution_str(solution_str)
    if statement_with_header:
        return statement_with_header + "\n" + solution_str
    return solution_str

def execute_lean_file(lean_code: str, lean_dir: str, timeout: int = 300) -> dict:
    """Execute a Lean file locally using lake env lean."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False, dir=lean_dir) as f:
        f.write(lean_code)
        temp_file = f.name
    
    try:
        # We rely on 'lake env lean' to handle imports correctly
        process = subprocess.run(
            ["lake", "env", "lean", temp_file],
            cwd=lean_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Check for explicit "sorry" which invalidates the proof
        by_sorry_pattern = r'by\s*\n*\s*sorry'
        no_sorry = not re.search(by_sorry_pattern, lean_code)
        
        return {
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "timeout": False,
            "no_sorry": no_sorry
        }
        
    except subprocess.TimeoutExpired as e:
        return {
            "returncode": -1,
            "stdout": e.stdout.decode('utf-8') if e.stdout else "",
            "stderr": e.stderr.decode('utf-8') if e.stderr else "Execution timeout",
            "timeout": True,
            "no_sorry": False
        }
    finally:
        try:
            os.unlink(temp_file)
        except OSError:
            pass

def compute_score_local(
    solution_str: str,
    ground_truth: Optional[str] = None,
    extra_info: Optional[dict] = None,
) -> dict:
    """
    Compute reward score by running Lean locally.
    Designed to be thread-safe for ThreadPoolExecutor.
    """
    if extra_info is None:
        extra_info = {}
    
    statement_with_header = extra_info.get("statement_with_header", "")
    lean_code = combine_statement_and_solution(statement_with_header, solution_str)
    
    lean_dir = extra_info.get("lean_dir", _LEAN_DIR)
    timeout = extra_info.get("timeout", _TIMEOUT)
    
    try:
        result = execute_lean_file(lean_code, lean_dir, timeout=timeout)
        
        verified = (result["returncode"] == 0 and 
                    result.get("no_sorry", False) and 
                    not result.get("timeout", False))
        
        if verified:
            info = "verified"
        else:
            info_parts = []
            if result.get("timeout"):
                info_parts.append("timeout")
            if not result.get("no_sorry", False):
                info_parts.append("contains_sorry")
            if result["returncode"] != 0:
                stderr = result.get("stderr", "").strip()
                first_line = stderr.split("\n")[0][:100] if stderr else f"code_{result['returncode']}"
                info_parts.append(f"error: {first_line}")
            info = ", ".join(info_parts) if info_parts else "verification_failed"
        
        return {
            "score": 1.0 if verified else -1.0,
            "acc": verified,
            "info": info
        }
        
    except Exception as e:
        return {"score": -1.0, "acc": False, "info": f"exception: {str(e)}"}