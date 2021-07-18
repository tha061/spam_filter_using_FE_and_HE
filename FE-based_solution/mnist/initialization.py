"""
This should be the first file you run. Generates your master secret key and the
corresponding public key, and fills the database with precomputations.

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

# inst = 'objects/instantiations/MNT159.inst'
inst = 'objects/instantiations/mnt224_th.inst'
#model = 'objects/ml_models/SpamModel_2491.mlm'
vector_length =  2000

if not os.path.exists('objects/msk'):
    os.makedirs('objects/msk')
if not os.path.exists('objects/pk'):
    os.makedirs('objects/pk')
# if not os.path.exists('objects/msk/common_{}.msk'.format(vector_length)):
if not os.path.exists('objects/msk/common_mnt224_th_{}.msk'.format(vector_length)):
    print('Generating keys.')
    make_keys.make_keys(
        vector_length,
        inst=inst,
        # name='common',
        name='common_mnt224_th',
        path='objects',
    )
    print('Done!\n')
else:
    print('Keys were already generated.\n')


print('Precomputing discrete logarithm.')

scheme = scheme.ML_DGP(inst)
dlog = discretelogarithm.PreCompBabyStepGiantStep(
    scheme.group,
    scheme.gt,
    minimum=-1.7e+11,
    maximum=2.7e+11,
    step=1 << 13,
)
dlog.precomp()
print('Done!\n')
