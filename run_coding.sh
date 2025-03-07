#!/bin/bash

# 设置路径
TRAIN_FILE="/root/autodl-fs/autodl-fs/bert/data/merged_train_50.csv"
VALIDATION_FILE="/root/autodl-fs/autodl-fs/bert/data/merged_val_50.csv"
TEST_FILE="/root/autodl-fs/autodl-fs/bert/data/merged_test_50.csv"

# 设置描述特征嵌入和超曲率嵌入的路径
DESC_EMBEDDINGS="/root/autodl-fs/autodl-fs/bert/data/biobert_embeddings.pkl"
HYPER_EMBEDDINGS="/root/autodl-fs/autodl-fs/bert/data/hyperbolic_embeddings.pkl"

# ICD9映射文件路径
ICD9_MAPPING_FILE="/root/autodl-fs/autodl-fs/bert/data/icd9_mapping.pkl"

# 输出目录
OUTPUT_DIR="/root/autodl-fs/autodl-fs/bert/results"

# 预训练模型的名称或路径
MODEL_NAME="/root/autodl-fs/autodl-fs/bert/biobert_finetuned/best_model"

# 设置训练超参数
BATCH_SIZE=16
EPOCHS=100  # 训练周期数
MAX_SEQ_LENGTH=256  # 最大序列长度
LEARNING_RATE=0.0002  # 学习率
Q_LEARNING_RATE=0.0002  # Q-learning 的学习率
GRAD_ACCUM_STEPS=4 # 设置梯度累积步数为 4

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 使用Python脚本进行训练、验证和预测
python3 /root/autodl-fs/autodl-fs/bert/run_coding.py \
  --model_name_or_path $MODEL_NAME \
  --train_file $TRAIN_FILE \
  --validation_file $VALIDATION_FILE \
  --test_file $TEST_FILE \
  --description_embeddings $DESC_EMBEDDINGS \
  --hyperbolic_embeddings $HYPER_EMBEDDINGS \
  --icd9_mapping_file $ICD9_MAPPING_FILE \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs $EPOCHS \
  --max_seq_length $MAX_SEQ_LENGTH \
  --learning_rate $LEARNING_RATE \
  --q_learning_rate $Q_LEARNING_RATE \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_output_dir \
  --document_pooling_strategy flat \
  --lazy_loading \
  --dropout_rate 0.5
