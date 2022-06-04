
WANDB_LOG_MODEL=true python -m experiments.main \
	--config config/bitfit.yaml \
	--device cuda \
	--model_name "roberta-base" \
	--run_name "bitfit-lr-sched" \
	--run_dir "runs/bitfit/bitfit-lr-sched" \
	--n_runs 10

