"""Utils module"""

def number_parameters(model, aggregate=True, only_trainable=True): #TODO: check
    """
    Return the number of parameters of a given `model`.
    
    :param torch.nn.Module model: the torch module to inspect.
    :param bool aggregate: if True the return values is an integer corresponding
        to the total amount of parameters of whole model. If False, it returns a
        dictionary whose keys are the names of layers and the values the
        corresponding number of parameters. Default is True.
    :param bool trainable: if True, only trainable parameters are count,
        otherwise no. Default is True.
    :return: the number of parameters of the model
    :rtype: dict or int
    """
    tmp = {}
    for name, parameter in model.named_parameters():
        if only_trainable and not parameter.requires_grad:
            continue

        tmp[name] = parameter.numel()

    if aggregate:
        tmp = sum(tmp.values())

    return tmp
