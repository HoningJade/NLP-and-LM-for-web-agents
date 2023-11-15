#!/bin/bash

mkdir -p ./.auth
python browser_env/auto_login.py

python run.py --instruction_path agent/prompts/jsons/p_cot_id_actree_2s_no_na.json --model gpt-3.5-turbo --result_dir ../webarena/reproduce_results

