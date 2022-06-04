WANDB_LOG_MODEL=true python -m baselines.lstm.main \
	--config config/baseline-lstm.yaml \
	--device cuda \
	--save_metrics \
	--run_name "baseline-lstm" \
	--run_dir "runs/lstm/baseline-lstm" \
	--n_runs 3

