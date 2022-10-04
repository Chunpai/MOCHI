import argparse
from dm.exps import check_progress, hyperparameters_tuning, test_5folds


def generate_configs(model, data, max_seq_len, batch_size, hidden_dim):
    config_dict_list = []
    if model == "DKT" and data in ["MasteryGrids"]:
        stride = max_seq_len
        for lr in [0.05, 0.01, 0.003]:
            config = {
                "agent": "{}Agent".format(model),
                "mode": "test",
                "metric": "auc",
                "cuda": False,
                "seed": 1024,
                "data_name": data,
                "min_seq_len": 2,
                "max_seq_len": max_seq_len,
                "max_subseq_len": None,
                "stride": stride,
                "validation_split": 0.2,
                "shuffle": True,
                "num_workers": 8,
                "batch_size": batch_size,
                "test_batch_size": 1024,
                "rnn_type": "LSTM",
                "hidden_dim": hidden_dim,
                "num_layers": 1,
                "nonlinearity": "tanh",
                "optimizer": "adam",
                "learning_rate": lr,
                "epsilon": 0.1,
                "max_grad_norm": 10.0,
                "max_epoch": 50,
                "log_interval": 10,
                "validate_every": 1,
                "save_checkpoint": False,
                "checkpoint_file": "checkpoint.pth.tar"
            }
            config_dict_list.append(config)
    return config_dict_list


if __name__ == '__main__':
    model = "DKT"
    data = "MasteryGrids"

    # mode = "hyperparameters"
    # mode = "check_progress"
    mode = "5folds"

    arg_parser = argparse.ArgumentParser(description="DKT Experiments")
    arg_parser.add_argument('-sl', '--seq_len', type=int, default=50)
    arg_parser.add_argument('-bs', '--batch_size', type=int, default=512)
    arg_parser.add_argument('-hd', '--hidden_dim', type=int, default=100)
    args = arg_parser.parse_args()

    args_list = ["max_seq_len", "batch_size", "hidden_dim", "learning_rate"]
    exp_name = "exp_sl_{}_bs_{}_hd_{}_lr_{}"

    progress_dict, best_config = check_progress(model, data, args_list)
    if mode == "hyperparameters":
        config_list = generate_configs(model, data, args.seq_len, args.batch_size, args.hidden_dim)
        hyperparameters_tuning(model, data, args_list, exp_name, config_list, progress_dict)
    elif mode == "5folds":
        print(best_config)
        best_config["save_checkpoint"] = True
        test_5folds(model, data, best_config)
    else:
        print(mode)
