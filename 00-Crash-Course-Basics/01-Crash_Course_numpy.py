import numpy as np
my_list = [1,2,3]
print(type(np.array(my_list)))

print(np.array(my_list))
print(np.arange(0,10,2))
print(np.zeros((3,5)))
print(np.ones((3,5)))
print(np.linspace(0,11,10))
print(np.random.randint(0,10))
print(np.random.randint(0,100,(3,3)))
np.random.seed(101)
print(np.random.randint(0,100,10))