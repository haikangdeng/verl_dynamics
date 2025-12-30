#!/usr/bin/env python3
"""
Local verification script for model outputs using lake env lean.
Executes Lean code locally instead of sending to Kimina server.
"""
import subprocess
import os
import json
import time
import re
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
import itertools


def safe_div(a: float, b: float) -> float:
    """Safe division that returns 0 if denominator is 0."""
    return (a / b) if b else 0.0


def classify_failure(result: dict) -> str:
    """
    Classify the failure reason for error analysis.
    """
    if result.get("timeout"):
        return "execution_timeout"
    
    if result["returncode"] != 0:
        stderr = result.get("stderr", "").lower()
        if "unknown constant" in stderr or "unknown identifier" in stderr:
            return "compile_unknown_constant"
        if "type mismatch" in stderr:
            return "compile_type_mismatch"
        if "unsolved goals" in stderr:
            return "compile_unsolved_goals"
        if "invalid" in stderr:
            return "compile_invalid_syntax"
        if "unexpected" in stderr:
            return "compile_unexpected_token"
        return "compile_error"
    
    if not result.get("no_sorry"):
        return "contains_sorry"
    
    return "other_reject"


def execute_lean_file(lean_code: str, lean_dir: str, timeout: int = 300) -> dict:
    """
    Execute a Lean file locally and return the result.
    
    Args:
        lean_code: The Lean code to execute
        lean_dir: Directory where lake/lean is set up
        timeout: Execution timeout in seconds
    
    Returns:
        dict with: returncode, stdout, stderr, timeout, no_sorry
    """
    # Create a temporary lean file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False, dir=lean_dir) as f:
        f.write(lean_code)
        temp_file = f.name
    
    try:
        # Execute using lake env lean
        process = subprocess.run(
            ["lake", "env", "lean", temp_file],
            cwd=lean_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Check if code contains "sorry"
        by_sorry_pattern = r'by\s*\n*\s*sorry'
        no_sorry = not re.search(by_sorry_pattern, lean_code)
        
        result = {
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "timeout": False,
            "no_sorry": no_sorry
        }
        
    except subprocess.TimeoutExpired as e:
        result = {
            "returncode": -1,
            "stdout": e.stdout.decode('utf-8') if e.stdout else "",
            "stderr": e.stderr.decode('utf-8') if e.stderr else "Execution timeout",
            "timeout": True,
            "no_sorry": False
        }
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except:
            pass
    
    return result


def is_verified(result: dict) -> bool:
    """Check if a proof is successfully verified."""
    return (result["returncode"] == 0 and 
            result.get("no_sorry", False) and 
            not result.get("timeout", False))


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def main():
    ap = argparse.ArgumentParser(description="Verify model outputs using local Lean execution")
    ap.add_argument("--preds", required=True, help="Path to model output JSON file")
    ap.add_argument("--lean_dir", default="/data2/haikang/projects/lean-prover/lean-project",
                    help="Directory with lake/lean setup")
    ap.add_argument("--timeout", type=int, default=300, help="Execution timeout in seconds")
    ap.add_argument("--num_workers", type=int, default=64, 
                    help="Number of parallel workers for execution (default: 64)")
    ap.add_argument("--no_save_files", action="store_true", 
                    help="Disable saving error and report files")
    args = ap.parse_args()

    # Setup paths
    preds_path = Path(args.preds)
    if not preds_path.exists():
        print(f"[ERROR] Prediction file not found: {args.preds}")
        return
    
    preds_dir = preds_path.parent
    preds_stem = preds_path.stem
    
    if args.no_save_files:
        save_errors = None
        save_correctness = None
        report_path = None
    else:
        save_errors = str(preds_dir / f"{preds_stem}_local_errors.json")
        save_correctness = str(preds_dir / f"{preds_stem}_local_correctness.json")
        report_path = str(preds_dir / f"{preds_stem}_local_report.txt")

    # Initialize report file
    def print_and_log(*args, **kwargs):
        print(*args, **kwargs)
        if report_path:
            with open(report_path, 'a', encoding='utf-8') as f:
                print(*args, file=f, **kwargs)

    if report_path:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Local Lean Execution Report\n")
            f.write(f"Generated for: {args.preds}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

    # Load predictions
    print_and_log(f"[INFO] Loading predictions from: {args.preds}")
    with open(args.preds, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    
    print_and_log(f"[INFO] Loaded {len(preds)} problems")
    
    # Check if lean_dir exists
    if not os.path.isdir(args.lean_dir):
        print_and_log(f"[ERROR] Lean directory not found: {args.lean_dir}")
        return
    
    print_and_log(f"[INFO] Using Lean directory: {args.lean_dir}")
    print_and_log(f"[INFO] Using {args.num_workers} parallel workers")
    
    # Execute all proofs
    print_and_log(f"[INFO] Starting local execution...")
    start_time = time.time()
    
    results = {}
    problem_results = defaultdict(list)
    all_results_details = []  # All results (both pass and fail)
    correctness_results = []  # Correctness status for each proof
    error_counter = Counter()
    
    # Thread-safe locks for shared data structures
    results_lock = threading.Lock()
    error_lock = threading.Lock()
    
    # Prepare all tasks
    tasks = []
    for i, pred_entry in enumerate(preds):
        proofs = pred_entry.get("proofs", [])
        for j, proof_attempt in enumerate(proofs):
            code = proof_attempt.get("cleaned_proof", "")
            if not code:
                print_and_log(f"[WARNING] Empty cleaned_proof at problem {i}, proof {j}")
                continue
            tasks.append((i, j, code, pred_entry.get("problem", "")))
    
    total_items = len(tasks)
    print_and_log(f"[INFO] Executing {total_items} proofs with {args.num_workers} workers...")
    
    pbar = tqdm(total=total_items, desc="Executing", 
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}]")
    
    def process_task(task):
        i, j, code, problem = task
        result = execute_lean_file(code, args.lean_dir, timeout=args.timeout)
        return (i, j, code, problem, result)
    
    # Execute tasks in parallel
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}
        
        for future in as_completed(futures):
            try:
                i, j, code, problem, result = future.result()
                
                item_id = f"{i}:{j}"
                with results_lock:
                    results[item_id] = result
                    problem_results[i].append((j, result))
                
                verified = is_verified(result)
                
                # Record correctness
                with error_lock:
                    if save_correctness:
                        correctness_results.append({
                            "problem_idx": i,
                            "proof_idx": j,
                            "verified": verified
                        })
                    
                    # Record all results (both pass and fail) in error.json
                    if save_errors:
                        if verified:
                            result_detail = {
                                "problem_idx": i,
                                "proof_idx": j,
                                "problem": problem,
                                "proof": code,
                                "verified": True,
                                "returncode": result["returncode"],
                                "stdout": result["stdout"]
                            }
                        else:
                            error_type = classify_failure(result)
                            error_counter[error_type] += 1
                            result_detail = {
                                "problem_idx": i,
                                "proof_idx": j,
                                "problem": problem,
                                "proof": code,
                                "verified": False,
                                "error_type": error_type,
                                "returncode": result["returncode"],
                                "stdout": result["stdout"],
                                "stderr": result["stderr"],
                                "timeout": result.get("timeout", False)
                            }
                        all_results_details.append(result_detail)
                    elif not verified:
                        # Still count errors even if not saving
                        error_type = classify_failure(result)
                        error_counter[error_type] += 1
                
            except Exception as e:
                print_and_log(f"[ERROR] Exception processing task: {e}")
            
            pbar.update(1)
    
    pbar.close()
    total_time = time.time() - start_time
    print_and_log(f"[INFO] Execution completed in {total_time:.2f} seconds")
    
    # Calculate statistics
    num_problems = len(preds)
    num_responses = len(results)
    num_verified_responses = sum(1 for r in results.values() if is_verified(r))
    
    # Problem-level statistics
    problems_solved = 0
    problems_with_responses = 0
    proof_lengths = []
    responses_per_problem = []
    
    for i in range(num_problems):
        if i not in problem_results:
            responses_per_problem.append(0)
            continue
        
        problem_proofs = problem_results[i]
        responses_per_problem.append(len(problem_proofs))
        problems_with_responses += 1
        
        any_verified = any(is_verified(result) for _, result in problem_proofs)
        if any_verified:
            problems_solved += 1
        
        for j, result in problem_proofs:
            code = preds[i]["proofs"][j].get("cleaned_proof", "")
            proof_lengths.append(len(code))
    
    # Pass@k calculations - compute for all k=2^n <= max samples
    max_samples = max((len(problem_results[i]) for i in problem_results), default=0)
    k_values = []
    k = 1
    while k <= max_samples:
        k_values.append(k)
        k *= 2
    
    # Collect num_samples and num_correct for all problems
    num_samples_list = []
    num_correct_list = []
    
    for i in range(num_problems):
        if i in problem_results:
            problem_proofs = problem_results[i]
            n_samples = len(problem_proofs)
            n_correct = sum(1 for _, result in problem_proofs if is_verified(result))
            num_samples_list.append(n_samples)
            num_correct_list.append(n_correct)
    
    # Calculate pass@k for each k value using the new estimator
    pass_at_k_dict = {}
    for k in k_values:
        if num_samples_list:
            pass_at_k_values = estimate_pass_at_k(num_samples_list, num_correct_list, k)
            pass_at_k_dict[k] = pass_at_k_values.tolist()
    
    # Print report
    print_and_log("\n" + "=" * 70)
    print_and_log("VERIFICATION STATISTICS (Local Execution)")
    print_and_log("=" * 70)
    
    print_and_log(f"\n[Dataset Statistics]")
    print_and_log(f"  Total problems:                {num_problems}")
    print_and_log(f"  Problems with responses:       {problems_with_responses}")
    print_and_log(f"  Total proof attempts:          {num_responses}")
    print_and_log(f"  Avg proofs per problem:        {safe_div(num_responses, num_problems):.2f}")
    
    print_and_log(f"\n[Verification Results]")
    print_and_log(f"  Verified proof attempts:       {num_verified_responses}")
    print_and_log(f"  Failed proof attempts:         {num_responses - num_verified_responses}")
    print_and_log(f"  Sample-level accuracy:         {safe_div(num_verified_responses, num_responses):.4f}")
    
    print_and_log(f"\n[Problem-level Results]")
    print_and_log(f"  Problems solved (any@k):       {problems_solved}")
    print_and_log(f"  Problem solve rate:            {safe_div(problems_solved, num_problems):.4f}")
    
    if k_values and pass_at_k_dict:
        print_and_log(f"\n[Pass@k Metrics]")
        for k in k_values:
            if k in pass_at_k_dict and pass_at_k_dict[k]:
                avg_pass = np.mean(pass_at_k_dict[k])
                print_and_log(f"  Pass@{k:<2}:                        {avg_pass:.4f}")
    
    if proof_lengths:
        print_and_log(f"\n[Proof Length Statistics]")
        print_and_log(f"  Average proof length:          {sum(proof_lengths)/len(proof_lengths):.1f} chars")
        print_and_log(f"  Min proof length:              {min(proof_lengths)}")
        print_and_log(f"  Max proof length:              {max(proof_lengths)}")
    
    print_and_log(f"\n[Performance]")
    print_and_log(f"  Total execution time:          {total_time:.2f} seconds")
    print_and_log(f"  Avg time per proof:            {safe_div(total_time, num_responses):.3f} seconds")
    print_and_log(f"  Throughput:                    {safe_div(num_responses, total_time):.2f} proofs/sec")
    
    if error_counter:
        print_and_log(f"\n[Error Analysis]")
        print_and_log(f"  Total errors:                  {sum(error_counter.values())}")
        print_and_log(f"  Error type breakdown:")
        for error_type, count in error_counter.most_common():
            print_and_log(f"    {error_type:30s}: {count:5d} ({safe_div(count, sum(error_counter.values())):.2%})")
    
    print_and_log("\n" + "=" * 70)
    
    # Sort and save all results (both pass and fail)
    if save_errors and all_results_details:
        # Sort by problem_idx, then proof_idx
        all_results_details.sort(key=lambda x: (x["problem_idx"], x["proof_idx"]))
        print_and_log(f"\n[INFO] Saving {len(all_results_details)} results (pass and fail) to: {save_errors}")
        with open(save_errors, 'w', encoding='utf-8') as f:
            json.dump(all_results_details, f, indent=2, ensure_ascii=False)
        print_and_log(f"[INFO] Results saved successfully")
    
    # Sort and save correctness results
    if save_correctness and correctness_results:
        # Sort by problem_idx, then proof_idx
        correctness_results.sort(key=lambda x: (x["problem_idx"], x["proof_idx"]))
        print_and_log(f"\n[INFO] Saving {len(correctness_results)} correctness results to: {save_correctness}")
        with open(save_correctness, 'w', encoding='utf-8') as f:
            json.dump(correctness_results, f, indent=2, ensure_ascii=False)
        print_and_log(f"[INFO] Correctness results saved successfully")
    
    if report_path:
        print_and_log(f"\n[INFO] Report saved to: {report_path}")


if __name__ == "__main__":
    main()