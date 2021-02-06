def count_parameters(model):
    """ パラメータ数を取得する．
    """
    n = 0
    for param in model.parameters():
        if param.requires_grad:
            n += param.numel()
    return n
