# 11711-webarena


## Reproducing WebArena Baseline Results
Follow the [official WebArena instructions](https://github.com/web-arena-x/webarena#quick-walkthrough) in setting up the environment.

Then, to reproduce the GPT-3.5 baseline, run:

```
python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
  --test_start_idx 0 \
  --test_end_idx 812 \
  --model gpt-3.5-turbo \
  --result_dir results/
```


## Examine and Visualize GPT 3.5 Baseline Result
```python logfile_to_csv.py```


## Launch LLaMA-2-70B Server

The default arguments are to host LLaMA-2-70B at half precision (fp16) on a server with 4 GPUs. To do so, run:

```
bash launch_llama70b_server.sh
```

After the server is running, we can make POST requests to it, as per the instructions in [lti-llm-deployment](https://github.com/neulab/lti-llm-deployment/tree/main#example-api-usage).