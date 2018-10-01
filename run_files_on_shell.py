from subprocess import call
import numpy as np

load_tau = np.loadtxt('tau.txt', dtype=float)

start = 0
for i in range(start,start+100):
    # command = "python reinforce.py --mass " + str(load_tau[i,0]) + " --length " + str(load_tau[i,1])
    command = "qsub -l short -t 1 run.sh " + str(load_tau[i, 0]) + " " +str(load_tau[i, 1])
    # command = "pwd"
    l = command.split(" ")
    # print(l)
    call(l)

# call(["python", "reinforce.py", "--mass", "2.", "--length", "8."])