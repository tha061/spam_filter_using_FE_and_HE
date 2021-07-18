"""
This is to generate an pairing group curve instance.

"""
import os
import sys
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import setup

from core import (
    discretelogarithm,
    make_keys,
    scheme,
)
import os
import json

from charm.toolbox.pairinggroup import PairingGroup
from charm.core.engine.util import objectToBytes, bytesToObject

from charm.toolbox.pairinggroup import PairingGroup
from core.utils import exp, fast_exp_const_time, is_array, Serializable

group = PairingGroup("MNT224")
# print(type(group))
scheme = scheme.ML_DGP()
print(scheme)
instance = scheme.create_(group)

path = 'objects/instantiations/'
out_filename = 'mnt224_th'
instance.export(path, out_filename)


# # decode the instance
# inst = '/home/ubuntu/tham_project/corey/project/reading-in-the-dark/mnist/objects/instantiations/MNT159.inst'
print("Decode .inst:")
inst = 'objects/instantiations/mnt224_th.inst'
call = Serializable()
res = call.fromFile(inst)




