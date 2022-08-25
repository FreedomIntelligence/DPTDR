MODEL_NAME="retriever_model_s2.lr7e-3.nepochs10.psl128.prefix"

python -m dptdr.faiss_retriever \
--query_reps mencoding/query.$MODEL_NAME/dev.query \
--passage_reps mencoding/corpus.$MODEL_NAME/'*.pt' \
--depth 1000 \
--batch_size -1 \
--save_text \
--save_ranking_to dev.rank.$MODEL_NAME.tsv

python -m dptdr.faiss_retriever \
--query_reps mencoding/query.$MODEL_NAME/test.query \
--passage_reps mencoding/corpus.$MODEL_NAME/'*.pt' \
--depth 1000 \
--batch_size -1 \
--save_text \
--save_ranking_to test.rank.$MODEL_NAME.tsv

