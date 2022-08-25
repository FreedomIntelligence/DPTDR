
python -m torch.distributed.launch --nproc_per_node 8 -m dptdr.driver.train \
  --output_dir ./retriever_model_s2.lr7e-3.nepochs60.psl32.prefix \
  --overwrite_output_dir \
  --model_type roberta \
  --model_name_or_path ./uscl-nq-roberta-large \
  --q_max_len 32 \
  --p_max_len 128 \
  --prefix \
  --pre_seq_len 32 \
  --hidden_dropout_prob 0.1 \
  --per_device_train_batch_size 16 \
  --learning_rate 7e-3 \
  --num_train_epochs 60 \
  --grad_cache \
  --gc_q_chunk_size 8 \
  --gc_p_chunk_size 120 \
  --save_steps 6000 \
  --train_dir ./nq/robertal/train-all \
  --train_n_passages 2 \
  --negatives_x_device \
  --fp16 \
  --untie_encoder \
  --positive_passage_no_shuffle 2>&1 | tee -a log.retriever_model_s2.lr7e-3.nepochs60.psl32.prefix

