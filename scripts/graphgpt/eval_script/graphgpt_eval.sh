# to fill in the following path to extract projector for the second tuning stage!
output_model=
datapath=
graph_data_path=
res_path=
start_id=
end_id=
num_gpus=

python -m pygfm.baseline_models.graphgpt.eval.run_graphgpt --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}