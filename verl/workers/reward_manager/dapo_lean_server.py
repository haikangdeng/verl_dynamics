from collections import defaultdict
import torch
import os
import logging
import time
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Import from the utility path
from verl.utils.reward_score.lean_server import (
    get_kimina_pool, 
    combine_statement_and_solution, 
    classify_failure,
    verify_batch_items
)

@register("dapo_lean_server")
class DAPOServerRewardManager(AbstractRewardManager):
    """
    Batched Reward Manager for Remote Kimina Server.
    Aggregates the entire batch into a single HTTP request for maximum throughput.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        # Server Config
        kimina_host=None,
        kimina_timeout=300,
        kimina_num_proc=64
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len
        self.overlong_buffer_cfg = overlong_buffer_cfg
        
        self.host = kimina_host or os.getenv("KIMINA_HOST", "http://localhost:8001")
        self.timeout = int(kimina_timeout)
        self.num_proc = int(kimina_num_proc)

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
             return data.batch["rm_scores"]

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

        # 2. Build Batch for Server
        verification_items = []
        id_map = [] 

        for i in range(len(data)):
            extra_info = data.batch["non_tensor_batch"][i].get("extra_info", {})
            header = extra_info.get("statement_with_header", "")
            
            full_code = combine_statement_and_solution(header, response_strs[i])
            uid = str(i)
            verification_items.append({"id": uid, "code": full_code})
            id_map.append(uid)

        # 3. Execute Single Server Call
        verification_start = time.time()
        try:
            pool = get_kimina_pool(self.host, self.timeout, num_proc=self.num_proc)
            batch_results = verify_batch_items(pool, verification_items)
        except Exception as e:
            logging.error(f"Kimina Batch Failed: {e}")
            batch_results = {}
        verification_time = time.time() - verification_start

        # 4. Scatter Results
        batch_size = len(verification_items)
        avg_time_per_item = verification_time / batch_size if batch_size > 0 else 0
        throughput = batch_size / verification_time if verification_time > 0 else 0
        print(f"⌛️ [verification_timing] batch_size={batch_size}, total_time={verification_time:.2f}s, avg_per_item={avg_time_per_item:.2f}s, throughput={throughput:.2f} items/s")
        
        for i in range(len(data)):
            uid = id_map[i]
            result = batch_results.get(uid)
            
            verified = False
            info = "no_result"
            
            if result:
                if result.get("error"):
                    verified = False
                    info = classify_failure(result)
                else:
                    msgs = result.get("response", {}).get("messages", [])
                    has_error = any(m.get("severity") == "error" for m in msgs)
                    verified = not has_error
                    info = "verified" if verified else classify_failure(result)

            score = 1.0 if verified else -1.0
            
            reward_extra_info["acc"].append(verified)
            reward_extra_info["info"].append(info)
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
                print(f"[score] {score} ({info})")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor