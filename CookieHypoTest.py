#imports of external packages to use in our code
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import math
import numpy as np
from random import random, uniform
from math import log


# import our Random class from python/Random.py file
sys.path.append(".")
from MySort import MySort



    # function returns a random double (0 to infty) according to an exponential distribution
def Exponential(x,rate):

    
    f1 = rate * np.exp(-rate * x);
    return f1


    

# main function for our CookieAnalysis Python code
if __name__ == "__main__":
   
    haveInput = [False, False]

    InputFile = [None, None]

    alpha = 0.05

    for i in range(1,len(sys.argv)):
        if sys.argv[i] == '-h' or sys.argv[i] == '--help':
            continue


        if sys.argv[i] == '-input0':
            InputFile[0] = sys.argv[i+1]
            haveInput[0] = True

        if sys.argv[i] == '-input1':
            InputFile[1] = sys.argv[i+1]
            haveInput[1] = True

        if sys.argv[i] == '-alpha':
            alpha = float(sys.argv[i + 1])

    
    if '-h' in sys.argv or '--help' in sys.argv or not haveInput:
        print ("Usage: %s [options] [input file]" % sys.argv[0])
        print ("  options:")
        print ("   --help(-h)          print options")
        print
        sys.exit(1)
    
    Nmeas = 0
    rate = []
    time= []
    need_rate = True
     # loop over all hypotheses (only 2)
    for h in range(2):
        
        need_rate = True
        this_hyp = []
        
        with open(InputFile[h]) as ifile:
            
            # parse each line
            for line in ifile:
                
                # first line is the rate parameter
                if need_rate:
                    need_rate = False
                    rate.append(float(line))
                    continue
            
                # each line is a different experiment
                lineVals = line.split()
                Nmeas = len(lineVals)
                
                this_exp = []
                
                # need to go through all measurements to convert them from string to float
                for m in range(Nmeas):
                    this_exp.append(float(lineVals[m]))
                this_hyp.append(this_exp)

        time.append(this_hyp)


    LLR = []
    
    # loop over all hypotheses
    for h in range(2):
        
        this_hyp = []

        Nexp = len(time[h])

        # loop over all experiments
        for e in range(Nexp):
            Nmeas = len(time[h][e])

            LogLikeRatio = 0.

            # loop over all measurements to calculate the LLR
            for m in range(Nmeas):
    
                # LLR is a sum; one contributes positive, other negative
                LogLikeRatio += np.log( Exponential( time[h][e][m], rate[1] ) ) # LLR for H1

                LogLikeRatio -= np.log( Exponential( time[h][e][m], rate[0] ) ) # LLR for H0

            this_hyp.append(LogLikeRatio)

        LLR.append(this_hyp)



        



    Sorter = MySort()

    LLR[0] =  np.array(Sorter.DefaultSort(LLR[0]))
    LLR[1] =  np.array(Sorter.DefaultSort(LLR[1]))


    N0 = len(LLR[0])

    print("N0:"+str(N0))
    N1 = len (LLR[1])
    print("N1:"+str(N1))

    array0 = LLR[0]
    array1 = LLR[1]
    hmin = min(array0[0], array1[0])


    hmax = max(array0[N0-1], array1[N1-1])

    
    t = "{} measurements / experiment with rates $\lambda_0 = {:.2f}$, $\lambda_1 = {:.2f}$ times / sec".format(Nmeas, 2, 4)
    #make Npass figure
    plt.figure()
    ax = plt.axes()

    plt.hist(array0,10,density=True, color='r',alpha=0.5,label='$P(\lambda | Input0)$')
    plt.hist(array1, 10, density=True, color='b', alpha=0.5,label='$P(\lambda | Input1)$')
        
    lambda_crit = LLR[0][min(int((1-alpha)*N0), N0-1)]
    first_leftover = np.where( LLR[1] > lambda_crit )[0][0]
    beta = first_leftover/N1

    plt.axvline(lambda_crit, color='b')
    plt.text(lambda_crit, ax.get_ylim()[1] * 0.8, '$\\alpha = {:.3f}$'.format(alpha))
    plt.plot([],[], '', label='$\\alpha = {:.3f}$'.format(alpha))
    plt.plot([],[], '', label='$\\beta = {:.3f}$'.format(beta))
    plt.plot([],[], '', label='$\lambda_{crit} = $' + '${:.3f}$'.format(lambda_crit))
    ax.set_yscale('log')
    ax.set_xlabel('$\lambda = \log [ \mathcal{L}(H1) / \mathcal{L}(H0) ]$')
    ax.set_ylabel('Probability')
    plt.legend()
    plt.title(t)
    plt.grid(True)
    plot_name = 'rate1_{:.2f}rate2_{:.2f}'.format(rate[0], rate[1])
    plt.savefig(plot_name+"_ProjectFigure.png")
    plt.show()
