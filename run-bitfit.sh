
WANDB_LOG_MODEL=true python -m experiments.main \
	--config config/bitfit.yaml \
	--device cuda \
	--model_name "roberta-base" \
	--run_name "bitfit-final" \
	--run_dir "runs/bitfit/bitfit-final" \
	--n_runs 10

