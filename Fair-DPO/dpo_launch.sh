ts=`date +%Y_%m_%d_%H_%M_%S`
MASTER_PORT=29500
mode=dpo #fr fc
alpha=0.1
tau=-1
gamma=0.5

exp_name=llama3-sft
# exp_name=llama3-sft-fair-reg-a${alpha}-t${tau}
# exp_name=llama3-sft-fair-coe-g${gamma}-t${tau}

output_path=./checkpoints/DPO/${exp_name}/output_train_$ts
mkdir -p $output_path

MODEL_PATH=RLHFlow/LLaMA3-SFT
CKPT=RLHFlow/LLaMA3-SFT
batch_size=8

current_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGPATH=$current_dir/outputs
mkdir -p $LOGPATH

accelerate launch \
   --config_file accelerate_config/fsdp_8gpu.yaml \
   --main_process_port $MASTER_PORT \
   launch.py \
   loss=dpo \
   model=llama \
   datasets=[ultrabin,shp] \
   exp_name=$exp_name \
   +mode=$mode \
   +alpha=$alpha \
   +tau=$tau \
   +gamma=$gamma \
   ++cache_dir=$output_path \
   ++model.name_or_path=$MODEL_PATH \
   ++model.load_from=$CKPT \
   ++lr=5e-6 \
   ++loss.beta=0.1 \
   ++model.batch_size=$batch_size \
   ++model.gradient_accumulation_steps=4 \
   ++model.eval_batch_size=8 \
   2>&1 | tee $LOGPATH/training_${exp_name}_$ts.log
