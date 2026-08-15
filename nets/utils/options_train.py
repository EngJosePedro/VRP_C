
import os
import time 
import argparse    
import torch  

def get_options(args=None):

    parser = argparse.ArgumentParser(
        description="Attention based model for solving the VRP with Reinforcement Learning")

    parser.add_argument('--mode', default='train', help="Modo de uso 'train' or 'eval'")
    
    # Data
    parser.add_argument('--problem', default='vrp', help="The problem to solve, default 'vrp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--epoch_epsodes', type=int, default=-1, help='Number of batchs of instances per epoch during training')
    parser.add_argument('--epoch_size', type=int, default=1280000, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--pomo', type=int, default = 128, help="")

    parser.add_argument('--dist_type', default='euclidian', help="")

    # pretrened model
    parser.add_argument('--model', default='', help="")

    # Model
    parser.add_argument('--embedding_dim', type=int, default=256, help='Dimension of input embedding')
    parser.add_argument('--head_num', type=int, default=16, help='Dimension of attention heads')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the graph encoder')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Dimension of input embedding')
        
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    
    # BASELINE
    parser.add_argument('--baseline_type', default='rollout', help="baseline method")

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--freeze_model_epoch', type=int, default=-1, help="")
    
    parser.add_argument('--lr_critic', type=float, default=5e-5, help="Set the learning rate for the critic network")
    
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    
    # Params
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
        
    # Device
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    

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
    parser.add_argument('--use_checkpoint', action='store_true', help='')
    parser.add_argument('--sync', action='store_true', help='')
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

    if opts.epoch_size % opts.batch_size != 0:
        opts.epoch_size += (opts.batch_size - opts.epoch_size % opts.batch_size)
        print(f"Epoch size must be integer multiple of batch size!, epoch size updated for {opts.epoch_size}")

    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    
    if opts.use_lora: assert opts.use_lora and opts.model != "", "Adicione um modelo inicial!!!"

    return opts
