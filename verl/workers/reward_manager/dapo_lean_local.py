from collections import defaultdict
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Import from the utility path
from verl.utils.reward_score.lean_local import compute_score_local

@register("dapo_lean_local")
class DAPOLocalRewardManager(AbstractRewardManager):
    """
    Parallelized Reward Manager for Local Lean Verification.
    Uses ThreadPoolExecutor to run multiple 'lake env lean' subprocesses concurrently.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None, # Ignored, we use compute_score_local
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
        self.num_workers = num_workers or 32

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]
        
        verification_start = time.time()
        print(f"⌛️ [verification_start]")
        print(f'✅ len(data) = {len(data)}')

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

            tasks.append(partial(
                compute_score_local,
                solution_str=response_strs[i],
                ground_truth=ground_truth,
                extra_info=extra_info
            ))

        # 3. Parallel Execution
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(lambda f: f(), tasks))
        verification_time = time.time() - verification_start

        # 4. Scatter Results
        batch_size = len(results)
        avg_time_per_item = verification_time / batch_size if batch_size > 0 else 0
        throughput = batch_size / verification_time if verification_time > 0 else 0
        print(f"⌛️ [verification_timing] batch_size={batch_size}, total_time={verification_time:.2f}s, avg_per_item={avg_time_per_item:.2f}s, throughput={throughput:.2f} items/s")
        
        for i, result in enumerate(results):
            score = result["score"]
            reward_extra_info["acc"].append(result["acc"])
            reward_extra_info["info"].append(result["info"])
            reward_extra_info["verification_time"].append(verification_time)
            reward_extra_info["avg_time_per_item"].append(avg_time_per_item)
            
            # Overlong Penalty
            reward = score
            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                valid_len = valid_response_lengths[i]
                expected = self.max_resp_len - self.overlong_buffer_cfg.len
                exceed = valid_len - expected
                penalty = min(-float(exceed) / self.overlong_buffer_cfg.len * self.overlong_buffer_cfg.penalty_factor, 0)
                reward += penalty
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(penalty)

            target_idx = max(0, int(valid_response_lengths[i]) - 1)
            reward_tensor[i, target_idx] = reward

            # Logging
            data_source = data[i].non_tensor_batch[self.reward_fn_key]
            if already_print_data_sources.get(data_source, 0) < self.num_examine:
                already_print_data_sources[data_source] = already_print_data_sources.get(data_source, 0) + 1
                print(f"[prompt] ...") 
                print(f"[score] {score} | {result['info']}")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor