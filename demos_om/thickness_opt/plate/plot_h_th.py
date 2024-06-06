import numpy as np
import matplotlib.pyplot as plt

'''
# Distributed load
w_int0: 0.0136487
w_int1: 0.01315443
w_int2: 0.012946
w_int3: 0.01296533

# Line load
w_int0: 0.16976146
w_int1: 0.16217173
w_int2: 0.16194494
w_int3: 0.16196668
'''

num_tests = 4
f_files = []
for i in range(num_tests):
    f_files += [np.load("./h_th_profile"+str(i)+".npz"),]

plt.figure()
plt.plot(f_files[0]['x'], f_files[0]['h'], label="panel constant")
for i in range(1, num_tests):
    plt.plot(f_files[i]['x'], f_files[i]['h'], label="p="+str(i))

plt.legend()
plt.xlabel("x")
plt.ylabel("h")
plt.show()