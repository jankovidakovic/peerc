WANDB_LOG_MODEL=true python -m experiments.main \
	--config config/adapter-bottleneck.yaml \
	--device cuda \
	--model_name "roberta-base" \
	--run_name "adapter-bottleneck-final" \
	--run_dir "runs/adapters/bottleneck-final" \
	--n_runs 10

