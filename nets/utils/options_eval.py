
import os
import time 
import argparse    
import torch  

def get_options(args=None):

    parser = argparse.ArgumentParser(
        description="Attention based model for solving the VRP with Reinforcement Learning")

    parser.add_argument('--mode', default='eval', help="Modo de uso 'train' or 'eval'")
    
    # Data
    parser.add_argument('--problem', default='vrp', help="The problem to solve, default 'vrp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--pomo', type=int, default = 128, help="")

    parser.add_argument('--dist_type', default='euclidian', help="")

    # pretrened model
    parser.add_argument('--model', default='', help="")
    parser.add_argument('--models', default=None, nargs='+', help="")
    parser.add_argument('--models_size', type=int, default=None, nargs='+', help="")
    parser.add_argument('--strategy', type=str, default="N", help='N - Normal, CF - ClusterFirst')

    # Params
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
        
    # Device
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    # Validate
    parser.add_argument("--dataset", default=None, help="Filename of the dataset(s) to evaluate")
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--decode_strategy', default="sampling", type=str,
                        help='Sampling (sample) or Greedy (greedy)')
    
    # Misc
    parser.add_argument('--log_step', type=int, default=100, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', action='store_true', help='Resume from previous checkpoint file')
   
    parser.add_argument('--results_dir', default='results', help="Name of results directory")

    parser.add_argument('--no_gurobi', action='store_true', help='Disable gurobi solver')
    parser.add_argument('--use_lora', action='store_true', help='')
    parser.add_argument('--swapopt', action='store_true', help='')

    parser.add_argument('--dropout', type=float, default=0.1, help='')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda

    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )

    assert opts.model is not None or (opts.models is not None and opts.models_size is not None), "Indique o modelo a ser usado!!!"
    if opts.models is not None:
        assert opts.models_size is not None, "Indique a quantidade de nós dos dataset do modelo"
        assert len(opts.models) == len(opts.models_size), ""

    print(opts.models)
    print(opts.models_size)

    return opts
