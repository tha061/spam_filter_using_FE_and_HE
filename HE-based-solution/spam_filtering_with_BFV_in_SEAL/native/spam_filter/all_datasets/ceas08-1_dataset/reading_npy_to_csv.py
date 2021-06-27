import numpy as np 
from numpy import asarray
from numpy import savetxt

file_name = "vec_4000/sequential_hidden_3_MatMul_ReadVariableOp_transpose.npy"

arr = np.load(file_name)
arr_tranpose = arr.transpose()
# print(len(arr))
rows = len(arr_tranpose)
cols = len(arr_tranpose[0])


print(rows)
print(cols)

savetxt("vec_4000/vec_4000_weight_3.csv", arr, delimiter=',')

