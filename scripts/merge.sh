# python3 -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /data2/haikang/projects/verl_dynamics/checkpoints/format_gsm8k/Qwen3-1.7B-Base_1020_102036/global_step_100/actor \
#     --target_dir /data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen3-1.7B-Base/format_gsm8k_100

# python3 -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /data2/haikang/projects/verl_dynamics/checkpoints/gsm8k/format2correct_Qwen3-1.7B-Base_1020_160908/global_step_155/actor \
#     --target_dir /data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen3-1.7B-Base/format2correct_gsm8k_155

# python3 -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /data2/haikang/projects/cloned/verl/checkpoints/DAPO-Official/DAPO-Qwen2.5-Math-7B/BASELINE_d/global_step_300/actor \
#     --target_dir /data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen2.5-Math-7B/BASELINE_d_300

# python3 -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /data2/haikang/projects/cloned/verl/checkpoints/DAPO-Official/Format-only-DAPO-Qwen2.5-Math-7B/BASELINE_d/global_step_50/actor \
#     --target_dir /data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen2.5-Math-7B/Format-only-BASELINE_d_50

# python3 -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /data2/haikang/projects/cloned/verl/checkpoints/DAPO-Official/Format-only-DAPO-Qwen2.5-Math-7B/BASELINE_d/global_step_20/actor \
#     --target_dir /data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen2.5-Math-7B/Format-only-BASELINE_d_20

# python3 -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /data2/haikang/projects/verl_dynamics/checkpoints/dapo/correct_dapo_aime_Qwen3-8B-Base_1107_123414/global_step_35/actor \
#     --target_dir /data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen3-8B-Base/correct_dapo_aime_35

# python3 -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /data2/haikang/projects/verl_dynamics/checkpoints/dapo/correct_dapo_aime_Qwen3-8B-Base_1107_123414/global_step_10/actor \
#     --target_dir /data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen3-8B-Base/correct_dapo_aime_10

# python3 -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /data2/haikang/projects/verl_dynamics/checkpoints/dapo/correct_dapo_aime_Qwen3-8B-Base_1107_123414/global_step_20/actor \
#     --target_dir /data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen3-8B-Base/correct_dapo_aime_20

# python3 -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /data2/haikang/projects/verl_dynamics/checkpoints/dapo/correct_dapo_aime_Qwen3-8B-Base_1107_123414/global_step_30/actor \
#     --target_dir /data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen3-8B-Base/correct_dapo_aime_30

# python3 -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /data2/haikang/projects/verl_dynamics/checkpoints/dapo/correct_dapo_aime_Qwen3-8B-Base_1107_123414/global_step_40/actor \
#     --target_dir /data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen3-8B-Base/correct_dapo_aime_40

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /data2/haikang/projects/verl_dynamics/checkpoints/dapo/correct_dapo_aime_Qwen3-1.7B-Base_1113_142821/global_step_20/actor \
    --target_dir /data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen3-1.7B-Base/correct_dapo_aime_20_test