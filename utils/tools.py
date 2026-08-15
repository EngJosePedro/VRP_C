
import os
import json
import torch
from torch.nn import DataParallel

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


# --- setups
def setup_save_dir(save_dir, opts):
    os.makedirs(save_dir, exist_ok=True)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)



def load_problem(name):
    from Problems import VRP
    problem = {
        "vrp": VRP
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args

def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_model(path, model_class, epoch=None):
    #from nets.AttentionModel import AttentionModel
    #from nets.AttentionModelNode import AttentionModel as AttentionModelNode   

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))

    problem = load_problem(args['problem'])

    #model_class = AttentionModel
    
    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        Problem = problem,
        pomo = args["pomo"], 
        embedding_dim = args["embedding_dim"],
        num_heads = args["head_num"],
    )
    #model = model_class(
    #    problem = problem,
    #    graph_encoder_layer_num = args["n_graph_encode_layers"],
    #    graph_embedding_dim = args["embedding_dim"],
    #    graph_qkv_dim = args["qkv_dim"],
    #    graph_head_num = args["head_num"],
    #    graph_ff_hidden_dim = args["ff_hidden_dim"],
    #    graph_logit_clipping = args["tanh_clipping"],
    #)
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args

def torch_load_cpu2(load_path):
    return torch.load(
        load_path,
        map_location="cpu",
        weights_only=True
    )
def load_model2(path, model_class, epoch=None):
    
    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))

    problem = load_problem(args['problem'])

    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        Problem = problem,
        opts = args
    )
    
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu2(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model_state_dict', {})})

    return model, args

def torch_load_cpu2(load_path):
    return torch.load(
        load_path,
        map_location="cpu",
        weights_only=True
    )

from types import SimpleNamespace
def load_args2(filename):
    with open(filename, 'r') as f:
        args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    return args

def load_model2(path, model_class, epoch=None):
    
    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args2(os.path.join(path, 'args.json'))

    problem = load_problem(args.problem)

    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        Problem = problem,
        opts = args,
    )
    
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu2(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model_state_dict', {})})
    
    return model, args

def load_model3(path, actor_class, critic_class, epoch=None):

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)

    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if filename.endswith(".pt")
            )
        model_filename = os.path.join(path, f'epoch-{epoch}.pt')

    else:
        raise ValueError(f"{path} is not valid")

    args = load_args(os.path.join(path, 'args.json'))
    problem = load_problem(args['problem'])

    # ===== instancia modelos =====
    actor = actor_class(
        Problem=problem,
        pomo=args["pomo"],
        embedding_dim=args["embedding_dim"],
        num_heads=args["head_num"],
    )

    critic = critic_class(
                embedding_dim=args["embedding_dim"],
                hidden_dim=args["hidden_dim"],
                n_heads=args["head_num"],
            )
    
    # ===== load checkpoint =====
    load_data = torch_load_cpu2(model_filename)

    # ===== load actor =====
    actor.load_state_dict(load_data["model_state_dict"])

    # ===== load critic =====
    try:
        critic.load_state_dict(load_data["critic_state_dict"])
    except:
        pass

    return actor, critic, args

from types import SimpleNamespace
def load_args_Fer(filename):
    with open(filename, 'r') as f:
        args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    
    return args

def load_modelFer(path, model_class, base_model, epoch=None):
    
    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args_Fer(os.path.join(path, 'args.json'))
    
    #problem = load_problem(args['problem'])

    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(args, base_model = base_model)
    #    Problem = problem,
    #    pomo = args["pomo"], 
    #    embedding_dim = args["embedding_dim"],
    #    num_heads = args["head_num"],
    #    apply_gate = args["apply_gate_information"],
    #)
    
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu2(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model_state_dict', {})})

    return model, args