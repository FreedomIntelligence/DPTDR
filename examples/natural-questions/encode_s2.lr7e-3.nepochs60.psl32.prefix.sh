MODEL_NAME="retriever_model_s2.lr7e-3.nepochs60.psl32.prefix"

python -m torch.distributed.launch --nproc_per_node 8 -m dptdr.driver.mencode \
  --output_dir ./retriever_model \
  --model_type roberta \
  --model_name_or_path $MODEL_NAME \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path ./nq/robertal/corpus \
  --encoded_save_path mencoding/corpus.$MODEL_NAME

python -m torch.distributed.launch --nproc_per_node 2 -m dptdr.driver.mencode \
  --output_dir ./retriever_model \
  --model_type roberta \
  --model_name_or_path $MODEL_NAME \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --encode_in_path ./nq/robertal/query/dev.query.json \
  --encoded_save_path mencoding/query.$MODEL_NAME/dev.query

python -m torch.distributed.launch --nproc_per_node 2 -m dptdr.driver.mencode \
  --output_dir ./retriever_model \
  --model_type roberta \
  --model_name_or_path $MODEL_NAME \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --encode_in_path ./nq/robertal/query/test.query.json \
  --encoded_save_path mencoding/query.$MODEL_NAME/test.query
