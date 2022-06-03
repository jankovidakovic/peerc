WANDB_LOG_MODEL=true python -m experiments.main \
	--config config/baseline-roberta.yaml \
	--device cuda \
	--model_name "roberta-base" \
	--run_name "roberta-baseline-again" \
	--run_dir "runs/roberta-baseline/roberta-baseline-again" \
	--n_runs 4

