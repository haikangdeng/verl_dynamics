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


def compute_score(solution_str: str, ground_truth: str = None) -> dict:
    """Compute the reward score based on format only (presence of \\boxed{}).

    This function rewards responses that contain the \\boxed{} format,
    regardless of whether the answer is correct or not.

    Args:
        solution_str: The solution string to check
        ground_truth: The ground truth answer (not used, kept for API compatibility)

    Returns:
        Dictionary with score and has_format
    """
    # Check if the response contains \boxed{}
    has_format = "\\boxed{" in solution_str
    reward = 1.0 if has_format else -1.0

    return {
        "score": reward,
        "acc": has_format,
    }

