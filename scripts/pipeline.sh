####################################################################
set -v
set -e

#! MODEL_ARGS
PRETRAINED_MODEL=../pretrain_models/m2m_base
N_PROMPT_TOKENS=0 
MODEL_TYPE=M2M # mBART
FIX_PARAM=0 #! 1 means fix, 0 means not fix 

#! DATA ARGS
export HF_DATASETS_CACHE="../cache_dir/datasets"
DATA_DIR=XXX #! data_dir


TRAIN_DATASET_PATH=${DATA_DIR}/train.json

VALID_DATASET_PATH=${DATA_DIR}/valid.json
TEST_DATASET_PATH=${DATA_DIR}/test.json
DATA_MODE=en-zh #! en-zh-fr

MAX_SOURCE_LENGTH=40
MAX_TARGET_LENGTH=20
GENERATION_MAX_LENGTH=20


TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16


#! TRAINING ARGS
EPOCHS=10 
SEED=42 
NUM_BEAMS=5 
WARMUP_RATIO=0.1 
METRIC_FOR_BEST_MODEL=loss 

GRADIENT_ACCUMULATE=1
PATIENCE=5


for LR in 1e-2 2e-2
do
    #! WANDB
    # export WANDB_DISABLED=true #! whether to use wandb or not
    WANDB_PROJECT_NAME=XXX #! set wandb project name
    ENTITY=XXX #! set wandb entity name
    WANDB_RUN_NAME=data_${DATA_MODE}-${MODEL_TYPE}-train-batch_${TRAIN_BATCH_SIZE}-lr_${LR}-epochs_${EPOCHS}-beams_${NUM_BEAMS}-prompt-num_${N_PROMPT_TOKENS}-fix-param_${FIX_PARAM}
    export WANDB_PROJECT=${WANDB_PROJECT_NAME}
    export WANDB_ENTITY=${ENTITY}


    #!OUTPUT
    OUTPUT_MODEL_DIR=../checkpoints/${WANDB_RUN_NAME}

    mkdir -p ${OUTPUT_MODEL_DIR}

    DEVICE=0 
    export CUDA_VISIBLE_DEVICES=${DEVICE}


    python -u ../src/train.py \
        --model_name_or_path ${PRETRAINED_MODEL} \
        --do_train \
        --do_eval \
        --output_dir ${OUTPUT_MODEL_DIR} \
        --train_file ${TRAIN_DATASET_PATH} \
        --validation_file ${VALID_DATASET_PATH} \
        --max_source_length ${MAX_SOURCE_LENGTH} \
        --max_target_length ${MAX_TARGET_LENGTH} \
        --generation_max_length ${GENERATION_MAX_LENGTH} \
        --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
        --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATE} \
        --learning_rate ${LR} \
        --num_train_epochs ${EPOCHS} \
        --n_prompt_tokens ${N_PROMPT_TOKENS} \
        --fix_param ${FIX_PARAM} \
        --warmup_ratio ${WARMUP_RATIO} \
        --seed ${SEED} \
        --metric_for_best_model ${METRIC_FOR_BEST_MODEL} \
        --num_beams ${NUM_BEAMS} \
        --patience ${PATIENCE} \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --overwrite_output_dir \
        --pad_to_max_length \
        --save_total_limit 2 \
        --report_to wandb \
        --run_name ${WANDB_RUN_NAME} \


    export CUDA_VISIBLE_DEVICES=0

   
    OUTPUT_PREFIX=test
    python ../src/eval.py \
        --model_type ${MODEL_TYPE} \
        --eval_batch_size ${EVAL_BATCH_SIZE} \
        --n_prompt_tokens ${N_PROMPT_TOKENS} \
        --beams ${NUM_BEAMS} \
        --model_path ${OUTPUT_MODEL_DIR} \
        --test_dataset_path ${TEST_DATASET_PATH} \
        --output_prefix ${OUTPUT_PREFIX}

done

