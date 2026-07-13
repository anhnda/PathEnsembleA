python e1_batch_tabular.py --dataset breast_cancer --tau_diag --diag_n 30 --diag_oracle \
  --tau_sweep 0.01 0.1 1 10 100 --rivals --insdel_mode marginal

python e1_batch_image.py benchmark_50 --N 500 --tau_diag --diag_n 30 \
  --diag_lo 0.5 --diag_hi 64 --sigma_sweep 2 4 8 16 --rivals

python e1_batch_nlp.py --model distilbert --dataset sst2 --limit 50 --tau_diag --diag_n 30 --rivals

python e1_batch_image.py benchmark_50 --N 500 --sigma_sweep 2 4 8 16 --rivals --ig2_steps 30 --me_steps 100

python e1_batch_tabular.py --dataset breast_cancer --tau_sweep 0.01 0.1 1 10 100 --rivals --ig2_steps 40 --insdel_mode marginal

python e1_batch_nlp.py --model distilbert --dataset sst2 --limit 50 --tau_sweep 0.1 1 10 100 --rivals --ig2_steps 40 --me_steps 100