#Bài tập 1: Numpy cơ bản
#câu 1
import numpy as np
arr = np.arange(0, 10, 1)
print(arr)

#Câu 2
arr = np.ones((3,3)) > 0
print(arr)
arr1 = np.ones((3,3), dtype = bool)
print(arr1)
arr2 = np.full((3,3), fill_value = True, dtype = bool)
print(arr2)

#Cau 3
import numpy as np
arr = np.arange(0,10)
print(arr[arr%2 == 1])

#Câu 4
import numpy as np
arr = np.arange(0,10)
arr[arr%2 == 1] = -1
print(arr)

#Cau 5
import numpy as np
arr = np.arange(10)
arr_2d = arr.reshape(2,-1)
print(arr_2d)

#câu 6
import numpy as np
arr1 = np.arange(10).reshape(2,-1)
arr2= np.repeat(1,10).reshape(2,-1)
c = np.concatenate([arr1,arr2], axis = 0)
print("Result: \n", c)

#cau 7
import numpy as np
arr1 = np.arange(10).reshape(2,-1)
arr2 = np.repeat(1,10).reshape(2,-1)
c = np.concatenate([arr1, arr2], axis = 1)
print("C = " , c)

#Câu 8
import numpy as np
arr = np.array([1, 2, 3])
print(np.repeat(arr,3))
print(np.tile(arr,3))

#Cau 9
import numpy as np
a = np.array ([2,6,1,9,10,3,27])
index = np.nonzero((a>=5) & (a<=10))
print("result", a[index])

#Cau 10
import numpy as np
def maxi(x,y):
    if x >= y:
        return x
    else:
        return y

a= np.array([5,7,9,8,6,4,5])
b = np.array([6,3,4,8,9,7,1])
pair_max = np.vectorize(maxi, otypes = [float])
print(pair_max(a,b))

#cau 11
import numpy as np
a= np.array([5,7,9,8,6,4,5])
b = np.array([6,3,4,8,9,7,1])
print("result", np.where(a < b, b, a))