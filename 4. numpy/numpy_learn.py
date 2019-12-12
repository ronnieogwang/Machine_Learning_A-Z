'''Learning numby'''
#Arrays vs LIsts
#x_array = [1 2 3] , has no commas, its for numerical data
#x_list = [1,2,3]

#arrays are matrices so can be multidimensional i.e 3 x 3
#arrays allows vector wise addition i.e [1 2 3] +[1 2 3] = [2 4 6] 

#using lists
list_two = list(range(1,4))
list_three = list(range(1,4))
list_sum = []

for i in range(3):
    list_two[i] = list_two[i]**2
    list_three[i] = list_three[i]**3
    list_sum.append(list_two[i] + list_three[i]) 

print(list_sum)

#using numpy
import numpy as np
array_two = np.arange(1,4)**2
array_three = np.arange(1,4)**3
array_sum = array_two + array_three

#more operations at https://www.tutorialspoint.com/numpy/numpy_arithmetic_operations.html

#multidimensional 2D or 3D
x = np.arange(3) #1x 3
y = np.arange(3)
z = np.arange(3)
w = np.array([x,y,z])
np.shape(w)# 3x3 

#linespace(1,10.50) grabs 50 evenly spaced samples between 1 and 10

#indexing rows and columns is from 0
w[0,2]
w.dtype

#reshape array, te new array must be compartible with the data items in the original array
p = np.arange(9) #1x9
p.reshape(3,3)   #3x3

q = np.arange(18) #1x9
q.reshape(2,3,3)   #2 matrices @ 3 x 3
q.reshape(3,2,3)   #3 matrices @ 2x3


#slicing multidimensional arrays
#q(3,2,3), to index elements of the second block
q = np.arange(18).reshape(3,2,3) 
print(q[1, 0:2, 0:3]) #slices the second block

#conditional slicing
q = np.arange(18).reshape(3,2,3) 
greater5 = q>5
print(q[greater5])

#flatteing an array
q = np.arange(18).reshape(3,2,3)
print(q)
print(q.flatten())
print(q.ravel())
#Note: ravel and flatten do the same thing-flattening. But a change made to the ravelled array
#affects the original array, while flatten just returns another copy.

#stacking arrays
#1st axis, run vertically along the rows
#second axis ru horizontally along columns
x = np.arange(4).reshape(2,2)
y = np.arange(4,8).reshape(2,2)







