import sys
if './' not in sys.path:
	sys.path.append('./')

import torch

from utils.share import *
from models.util import create_model


def init_local(sd_weights_path, config_path, output_path):
    pretrained_weights = torch.load(sd_weights_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']
    model = create_model(config_path=config_path)
    scratch_dict = model.state_dict()
    target_dict = {}
    for sk in scratch_dict.keys():
        if sk.replace('local_adapter.', 'model.diffusion_model.') in pretrained_weights.keys():
            target_dict[sk] = pretrained_weights[sk.replace('local_adapter.', 'model.diffusion_model.')].clone()
        else:
            target_dict[sk] = scratch_dict[sk].clone()
            print('new params: {}'.format(sk))
    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)
    print('Done.')

def init_global(sd_weights_path, config_path, output_path):
    pretrained_weights = torch.load(sd_weights_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']
    model = create_model(config_path=config_path)
    scratch_dict = model.state_dict()
    target_dict = {}
    for sk in scratch_dict.keys():
        if sk in pretrained_weights.keys():
            target_dict[sk] = pretrained_weights[sk].clone()
        else:
            target_dict[sk] = scratch_dict[sk].clone()
            print('new params: {}'.format(sk))
    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)
    print('Done.')

def integrate(local_weights, global_weights, config_path, output_path):
    local_weights = torch.load(local_weights)
    if 'state_dict' in local_weights:
        local_weights = local_weights['state_dict']
    global_weights = torch.load(global_weights)
    if 'state_dict' in global_weights:
        global_weights = global_weights['state_dict']
    model = create_model(config_path=config_path)
    scratch_dict = model.state_dict()
    target_dict = {}
    for sk in scratch_dict.keys():
        if sk in local_weights and sk in global_weights:
            assert local_weights[sk].equal(global_weights[sk])
            target_dict[sk] = local_weights[sk].clone()
        elif 'local_adapter' in sk:
            assert sk in local_weights.keys()
            target_dict[sk] = local_weights[sk].clone()
        elif 'global_adapter' in sk:
            assert sk in global_weights.keys()
            target_dict[sk] = global_weights[sk].clone()
        else:
            raise ValueError()
    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)
    print('Done.')


if __name__ == '__main__':
    mode = sys.argv[1]
    assert mode in ['init_local', 'init_global', 'integrate']
    if mode == 'init_local':
        assert len(sys.argv) == 5
        init_local(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == 'init_global':
        assert len(sys.argv) == 5
        init_global(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        assert len(sys.argv) == 6
        integrate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])