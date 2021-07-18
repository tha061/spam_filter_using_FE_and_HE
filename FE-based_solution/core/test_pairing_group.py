
import os
import sys
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import os

from utils import exp, fast_exp_const_time, is_array, Serializable


inst = './mnist/objects/instantiations/MNT159.inst'
#model = 'objects/ml_models/SpamModel_2491.mlm'
vector_length =  10
call = Serializable()
# print("Investigate a key file in 'objects/msk/common_10.msk")
# source_file = './mnist/objects/msk/common_10.msk'
# res = call.fromFile(source_file)
print("Investigate file MNT159.inst:")
res = call.fromFile(inst)

print(res)



