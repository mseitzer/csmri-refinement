"""Hacky utils"""

def get_to_cuda(cuda):
    def to_cuda(tensor):
        return tensor.cuda() if cuda else tensor
    return to_cuda

def get_params(model):
    return [w for w in model.parameters()]


