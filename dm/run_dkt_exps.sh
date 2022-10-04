#!/bin/bash
#SBATCH -p ceashpc
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cwang25@albany.edu
#SBATCH -o sbatch.out
#SBATCH -e sbatch.error
#SBATCH --time=11-0:1 # The job should take 0 days, 0 hours, 1 minutes

# Now, run the python script for MasteryGrids
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 32 --hidden_dim 8
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 32 --hidden_dim 16
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 32 --hidden_dim 32
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 32 --hidden_dim 64
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 32 --hidden_dim 128

#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 64 --hidden_dim 8
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 64 --hidden_dim 16
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 64 --hidden_dim 32
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 64 --hidden_dim 64
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 64 --hidden_dim 128

#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 128 --hidden_dim 8
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 128 --hidden_dim 16
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 128 --hidden_dim 32
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 128 --hidden_dim 64
python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 50 --batch_size 128 --hidden_dim 128

#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 32 --hidden_dim 8
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 32 --hidden_dim 16
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 32 --hidden_dim 32
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 32 --hidden_dim 64
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 32 --hidden_dim 128

#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 64 --hidden_dim 8
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 64 --hidden_dim 16
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 64 --hidden_dim 32
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 64 --hidden_dim 64
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 64 --hidden_dim 128

#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 128 --hidden_dim 8
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 128 --hidden_dim 16
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 128 --hidden_dim 32
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 128 --hidden_dim 64
#python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/MOCHI/dm/run_dkt_exps.py --seq_len 200 --batch_size 128 --hidden_dim 128
