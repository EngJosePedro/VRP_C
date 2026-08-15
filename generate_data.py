import argparse
import os
import numpy as np


#from utils.tools import load_problem
from Problems import load_problem
from utils.data_utils import save_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_size', type=int, nargs='+', default=[20],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("--distr", type=str, default='uniform', help="")

    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")

    parser.add_argument("--problem", type=str, default='vrp',
                        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
                             " or 'all' to generate all")
    
    
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()


    Problem = load_problem(opts.problem)
    
    for size in opts.graph_size:
        DS = Problem.make_dataset(n_cust = size, num_samples = opts.dataset_size, seed = opts.seed, like_kool = True)#, seed = opts.seed

        dataset = DS.get_zip_data()

        datadir = os.path.join(opts.data_dir, Problem.NAME)
        print(datadir, Problem.NAME, opts.problem)
        os.makedirs(datadir, exist_ok=True)
        
        filename = os.path.join(datadir, "{}_{}_{}_seed{}.pkl".format(
                            Problem.NAME,
                            size, opts.name, opts.seed))
        
        save_dataset(dataset, filename)
        
        
