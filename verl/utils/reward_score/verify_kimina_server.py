#!/usr/bin/env python3
"""
Simple verification script for model outputs (e.g., minif2f-test).
Sends cleaned_proof directly to Kimina server without wrappers.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter
from KiminaPool import KiminaPool
from tqdm import tqdm
import numpy as np
import itertools


def safe_div(a: float, b: float) -> float:
    """Safe division that returns 0 if denominator is 0."""
    return (a / b) if b else 0.0


def classify_failure(entry: dict) -> str:
    """
    Classify the failure reason for error analysis.
    Priority: client error > compile error > other.
    """
    if entry.get("error"):
        err = str(entry["error"]).lower()
        if "timeout" in err:
            return "client_timeout"
        if "connection" in err or "network" in err:
            return "client_network"
        return "client_error"

    msgs = entry.get("response", {}).get("messages", [])
    for m in msgs:
        if m.get("severity") == "error":
            data = (m.get("data") or "").strip()
            if "unknown constant" in data:
                return "compile_unknown_constant"
            if "unresolved notation" in data:
                return "compile_unresolved_notation"
            if "type mismatch" in data:
                return "compile_type_mismatch"
            if "invalid field" in data:
                return "compile_invalid_field"
            if "unexpected token" in data:
                return "compile_unexpected_token"
            if "unsolved goals" in data:
                return "compile_unsolved_goals"
            return "compile_error"

    return "other_reject"


def is_verified(entry: dict) -> bool:
    """Check if a proof is successfully verified."""
    if entry.get("error") is not None:
        return False
    msgs = entry.get("response", {}).get("messages", [])
    # Any error message means failure
    if any(m.get("severity") == "error" for m in msgs):
        return False
    return True

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
    ap = argparse.ArgumentParser(description="Verify model outputs using Kimina server")
    ap.add_argument("--preds", required=True, help="Path to model output JSON file")
    ap.add_argument("--kimina_host", default="http://localhost:8001", help="Kimina server URL")
    ap.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds")
    ap.add_argument("--num_proc", type=int, default=64, help="Number of parallel processes")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for verification")
    ap.add_argument("--no_save_files", action="store_true", help="Disable saving error and report files")
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
        save_errors = str(preds_dir / f"{preds_stem}_errors.json")
        save_correctness = str(preds_dir / f"{preds_stem}_correctness.json")
        report_path = str(preds_dir / f"{preds_stem}_report.txt")

    # Initialize report file
    def print_and_log(*args, **kwargs):
        print(*args, **kwargs)
        if report_path:
            with open(report_path, 'a', encoding='utf-8') as f:
                print(*args, file=f, **kwargs)

    if report_path:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Kimina Verification Report\n")
            f.write(f"Generated for: {args.preds}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

    # Load predictions
    print_and_log(f"[INFO] Loading predictions from: {args.preds}")
    with open(args.preds, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    
    print_and_log(f"[INFO] Loaded {len(preds)} problems")
    
    # Initialize Kimina pool
    kimina = KiminaPool(args.kimina_host, timeout=args.timeout, 
                       num_proc=args.num_proc, batch_size=args.batch_size)
    
    # Prepare verification items
    items = []
    for i, pred_entry in enumerate(preds):
        proofs = pred_entry.get("proofs", [])
        for j, proof_attempt in enumerate(proofs):
            code = proof_attempt.get("cleaned_proof", "")
            if not code:
                print_and_log(f"[WARNING] Empty cleaned_proof at problem {i}, proof {j}")
                continue
            items.append({
                "id": f"{i}:{j}",
                "code": code,
                "problem_idx": i,
                "proof_idx": j
            })
    
    print_and_log(f"[INFO] Prepared {len(items)} proof attempts for verification")
    
    # Verify all proofs
    print_and_log(f"[INFO] Starting verification...")
    start_time = time.time()
    
    pbar = tqdm(total=len(items), desc="Verifying", 
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}]")
    
    # Use simple batch verification
    results = kimina.verify_simple_batch(items, 
                                         on_progress=lambda n: pbar.update(n))
    pbar.close()
    
    total_time = time.time() - start_time
    print_and_log(f"[INFO] Verification completed in {total_time:.2f} seconds")
    
    # Process results
    problem_results = defaultdict(list)  # problem_idx -> list of (proof_idx, result)
    all_results_details = []  # All results (both pass and fail)
    correctness_results = []  # Correctness status for each proof
    error_counter = Counter()
    
    for item_id, result in results.items():
        i, j = map(int, item_id.split(':'))
        problem_results[i].append((j, result))
        
        verified = is_verified(result)
        
        # Record correctness
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
                    "problem": preds[i].get("problem", ""),
                    "proof": preds[i]["proofs"][j].get("cleaned_proof", ""),
                    "verified": True,
                    "verification_response": result.get("response", {})
                }
            else:
                error_type = classify_failure(result)
                error_counter[error_type] += 1
                result_detail = {
                    "problem_idx": i,
                    "proof_idx": j,
                    "problem": preds[i].get("problem", ""),
                    "proof": preds[i]["proofs"][j].get("cleaned_proof", ""),
                    "verified": False,
                    "error_type": error_type,
                    "error_response": result.get("error"),
                    "verification_response": result.get("response", {})
                }
            all_results_details.append(result_detail)
        elif not verified:
            # Still count errors even if not saving
            error_type = classify_failure(result)
            error_counter[error_type] += 1
    
    # Calculate statistics
    num_problems = len(preds)
    num_responses = len(items)
    num_verified_responses = sum(1 for _, r in results.items() if is_verified(r))
    
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
        
        # Check if any proof works
        any_verified = any(is_verified(result) for _, result in problem_proofs)
        if any_verified:
            problems_solved += 1
        
        # Calculate proof lengths
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
    print_and_log("VERIFICATION STATISTICS")
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
    print_and_log(f"  Total verification time:       {total_time:.2f} seconds")
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

