#python e1_batch_image.py benchmark_50 --N 500 --tau_diag --diag_n 30 --diag_lo 0.5 --diag_hi 64 --rivals --tau_star
python e1_batch_image.py benchmark_50 --N 500 --tau_diag --diag_n 30 --diag_lo 0.5 --diag_hi 64 --rivals --tau_star --eg_real
python e1_batch_nlp.py --model distilbert --dataset sst2 --limit 50 --tau_diag --diag_n 30 --rivals --tau_star
python e1_batch_tabular.py --dataset wine --tau_diag --diag_n 30 --diag_oracle --tau_star --rivals --tau_sweep 0.01 0.1 1 10 100 > log_1x.txt 
