WANDB_LOG_MODEL=true python -m experiments.main \
	--config config/adapter-test.yaml \
	--device cuda \
	--model_name "roberta-base" \
	--run_name "roberta-test" \
	--run_dir "runs/test/roberta" \
	--n_runs 2

