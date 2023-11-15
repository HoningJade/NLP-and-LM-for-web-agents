TOKENIZERS_PARALLELISM=false \
MODEL_NAME=meta-llama/Llama-2-70b-hf \
MODEL_CLASS=AutoModelForCausalLM \
DEPLOYMENT_FRAMEWORK=hf_accelerate \
DTYPE=fp16 \
MAX_INPUT_LENGTH=4096 \
MAX_BATCH_SIZE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
gunicorn -t 0 -w 1 -b 0.0.0.0:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'