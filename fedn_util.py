import collections
import io
import os

import torch
from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def load_weights_into_model(weights, model):
    inpath = helper.get_tmp_path()
    with open(inpath, "wb") as fh:
        fh.write(weights.getbuffer())
    weights = helper.load(inpath)
    os.unlink(inpath)
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)




def extract_weights_from_model(model):
    # Convert from pytorch weights format numpy array
    updated_weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    outpath = helper.get_tmp_path()
    helper.save(updated_weights, outpath)
    with open(outpath, "rb") as fr:
        out_model = io.BytesIO(fr.read())
    os.unlink(outpath)

    return out_model