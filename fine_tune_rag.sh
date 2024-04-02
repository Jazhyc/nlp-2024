# Example given for fine-tuning RAG on a custom dataset

python examples/rag/finetune_rag.py \
    --data_dir data/gold \
    --output_dir outputs \
    --model_name_or_path bart \
    --model_type rag_sequence \
    --fp16 \
    --gpus 1 \
    --profile \
    --do_train \
    --do_predict \
    --n_val -1 \
    --train_batch_size 8 \
    --eval_batch_size 1 \
    --max_source_length 128 \
    --max_target_length 25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 3e-05 \
    --num_train_epochs 100 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1 \