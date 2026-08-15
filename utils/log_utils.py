
def log_values(cost, grad_norms, epoch, batch_id):
    
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('-epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))


def model_resume(model):
    print("--------------------------")
    for name, param in model.named_parameters():
        print(name, param.shape, param.numel())

    print("----------------------------")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable: {trainable:,}")
    print(f"Total: {total:,}")
    print("----------------------------")