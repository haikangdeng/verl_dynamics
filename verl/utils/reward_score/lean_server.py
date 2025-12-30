import os

# Import KiminaPool dynamically
try:
    from KiminaPool import KiminaPool
    KIMINA_AVAILABLE = True
except ImportError:
    KIMINA_AVAILABLE = False
    KiminaPool = None

# Cache for KiminaPool instances to reuse connections
_kimina_pools = {}

def get_kimina_pool(base_url, timeout, num_proc=64, batch_size=64):
    """Singleton-like accessor for KiminaPool."""
    if not KIMINA_AVAILABLE:
        raise ImportError("KiminaPool is not available. Please install it.")
    
    key = (base_url, timeout, num_proc, batch_size)
    if key not in _kimina_pools:
        _kimina_pools[key] = KiminaPool(
            base_url=base_url, 
            timeout=timeout, 
            num_proc=num_proc, 
            batch_size=batch_size
        )
    return _kimina_pools[key]

def clean_solution_str(solution_str: str) -> str:
    solution_str = solution_str.strip()
    if solution_str.endswith("```"):
        solution_str = solution_str[:-3].strip()
    if solution_str.startswith("```") and solution_str.endswith("```"):
        lines = solution_str.split("\n")
        if len(lines) > 2 and lines[0].startswith("```"):
            solution_str = "\n".join(lines[1:-1])
    return solution_str

def combine_statement_and_solution(statement: str, solution: str) -> str:
    solution = clean_solution_str(solution)
    return (statement + "\n" + solution) if statement else solution

def classify_failure(result: dict) -> str:
    """Analyze server response to determine failure type."""
    if result.get("error"):
        err = str(result["error"]).lower()
        if "timeout" in err: return "client_timeout"
        if "connection" in err: return "client_network"
        return "client_error"

    msgs = result.get("response", {}).get("messages", [])
    for m in msgs:
        if m.get("severity") == "error":
            data = (m.get("data") or "").strip()
            # Common Lean errors
            if "unknown constant" in data: return "compile_unknown_constant"
            if "unresolved notation" in data: return "compile_unresolved_notation"
            if "type mismatch" in data: return "compile_type_mismatch"
            if "unsolved goals" in data: return "compile_unsolved_goals"
            
            first_line = data.split("\n")[0][:100]
            return f"compile_error: {first_line}"
    
    return "other_reject"

def verify_batch_items(pool, items: list) -> dict:
    """
    Wrapper to call pool.verify_simple_batch.
    items: list of dicts {'id': str, 'code': str}
    Returns: dict mapping id -> result object
    """
    return pool.verify_simple_batch(items)