import numpy as np 
from numpy import asarray
from numpy import savetxt

file_name = "enron_dataset/vec_5000/sequential_output_layer_MatMul_ReadVariableOp_transpose.npy"
# file_name = "enron_dataset/vec_5000/sequential_hidden_1_MatMul_ReadVariableOp_transpose.npy"

arr = np.load(file_name)
arr_tranpose = arr.transpose()
# print(len(arr))
rows = len(arr_tranpose)
cols = len(arr_tranpose[0])


print(rows)
print(cols)


savetxt("enron_dataset/vec_5000/vec_5000_weight_4.csv", arr, delimiter=',')

