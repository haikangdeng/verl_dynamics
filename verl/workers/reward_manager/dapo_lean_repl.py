from collections import defaultdict
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Import from the local lean_repl file
from verl.utils.reward_score.lean_repl import LeanREPLPool, compute_score_repl, _LEAN_DIR

@register("dapo_lean_repl")
class DapoLeanREPLRewardManager(AbstractRewardManager):
    """
    Process Reward Manager for Lean 4.
    Creates a new pool of 'repl' subprocesses for each step and cleans them up afterwards.
    This ensures proper cleanup even when training errors occur.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None, 
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        num_workers=None
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len
        self.overlong_buffer_cfg = overlong_buffer_cfg
        
        # Default to a reasonable number of workers (e.g., 8)
        self.num_workers = num_workers or 16
        self.lean_dir = _LEAN_DIR

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]
        
        # Create a new pool for this step
        pool = LeanREPLPool(self.num_workers, self.lean_dir)
        
        try:
            verification_start = time.time()
            print(f"⌛️ [REPL verification_start] Batch size: {len(data)}")

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_extra_info = defaultdict(list)
            already_print_data_sources = {}

            # 1. Batch Decode
            prompt_ids = data.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_response_lengths = data.batch["attention_mask"][:, prompt_len:].sum(dim=1)
            
            decode_ids = [
                data.batch["responses"][i, :valid_response_lengths[i]] 
                for i in range(len(data))
            ]
            response_strs = self.tokenizer.batch_decode(decode_ids, skip_special_tokens=True)
            eos = self.tokenizer.eos_token
            response_strs = [s[:-len(eos)] if s.endswith(eos) else s for s in response_strs]

            # 2. Prepare Tasks
            tasks = []
            for i in range(len(data)):
                data_item = data[i]
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                rollout_scores = data_item.non_tensor_batch.get("reward_scores", {})
                extra_info["rollout_reward_scores"] = rollout_scores
                
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

                # We pass the pool to the function via partial
                tasks.append(partial(
                    compute_score_repl,
                    solution_str=response_strs[i],
                    pool=pool,
                    ground_truth=ground_truth,
                    extra_info=extra_info
                ))

            # 3. Parallel Execution
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(lambda f: f(), tasks))
                
            verification_time = time.time() - verification_start

            # 4. Scatter Results & Logging
            batch_size = len(results)
            throughput = batch_size / verification_time if verification_time > 0 else 0
            print(f"⌛️ [REPL stats] Total: {verification_time:.2f}s | Throughput: {throughput:.2f} it/s")
            
            for i, result in enumerate(results):
                score = result["score"]
                reward_extra_info["acc"].append(result["acc"])
                reward_extra_info["info"].append(result["info"])
                
                # Length Penalty Logic
                reward = score
                if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                    valid_len = valid_response_lengths[i]
                    expected = self.max_resp_len - self.overlong_buffer_cfg.len
                    exceed = valid_len - expected
                    penalty = min(-float(exceed) / self.overlong_buffer_cfg.len * self.overlong_buffer_cfg.penalty_factor, 0)
                    reward += penalty

                target_idx = max(0, int(valid_response_lengths[i]) - 1)
                reward_tensor[i, target_idx] = reward

                # Debug Printing
                data_source = data[i].non_tensor_batch[self.reward_fn_key]
                if already_print_data_sources.get(data_source, 0) < self.num_examine:
                    already_print_data_sources[data_source] = already_print_data_sources.get(data_source, 0) + 1
                    print(f"[REPL Check] Score: {score} | Info: {result['info']}")

            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            return reward_tensor
        finally:
            # Always clean up the pool, even if an error occurs
            print(f"🧹 [REPL] Cleaning up pool with {self.num_workers} workers")
            pool.close()