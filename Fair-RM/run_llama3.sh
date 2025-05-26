ts=`date +%Y_%m_%d_%H_%M_%S`

mode=bt #bt fr fc
alpha=0.1
beta=-1
gamma=0.5
desc=llama3_8b_sft

output_path=./checkpoints/RM/${desc}/output_train_$ts
mkdir -p $output_path


model_name=RLHFlow/LLaMA3-SFT
train_set_path=./data/hh_combine_base_train.jsonl

current_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGPATH=$current_dir/output
mkdir -p $LOGPATH

echo -e "[now] ${ts}\nsave to file ${output_path}"

accelerate launch llama3_8B_rm.py \
                    --model_name $model_name \
                    --max_length 2048 \
                    --train_set_path $train_set_path \
                    --per_device_train_batch_size 32 \
                    --gradient_accumulation_steps 2 \
                    --learning_rate 2e-6 \
                    --num_train_epochs 1 \
                    --output_path $output_path \
                    --mode $mode \
                    --alpha $alpha \
                    --beta $beta \
                    --gamma $gamma \
                    --deepspeed ./deepspeed_configs/deepspeed_3.json \
                    2>&1 | tee $LOGPATH/training_${desc}_$ts.log 

