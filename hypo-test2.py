#! /usr/bin/env python

# imports of external packages to use in our code
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
    #InputFile = ['file.txt','file1.txt']
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
    times= []
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

        times.append(this_hyp)



    LLR = []
    Nmeas = 0

    histograms = [None, None]
    bases = [None, None]
    for h in range(2):
        reshap_array = np.reshape( times[h], -1 )
        bins = np.arange( np.floor(reshap_array.min() ), np.ceil(reshap_array.max() ) )
        values, base = np.histogram(reshap_array, bins=bins, density=True )
        histograms[h] = values
        bases[h] = base

    # loop over all hypotheses
    for h in range(2):
        this_hyp = []
        Nexp = len(times[h])
        for e in range(Nexp):
            Nmeas = len(times[h][e])

            LogLikeRatio = 0.
            ok_LLR = True

            # loop over all measurements to calculate the LLR
            for m in range(Nmeas):

            	prob0 = 0
            	prob1 = 0

            	try:
            		prob_of_H0 = histograms[0][np.digitize(times[h][e][m], bases[0], right=True )]
            		prob_of_H1 = histograms[1][np.digitize(times[h][e][m], bases[1], right=True )]
            		if (prob_of_H0 > 0) and (prob_of_H1 > 0):
            			LogLikeRatio += np.log( prob_of_H1 ) # LLR for input1
            			LogLikeRatio -= np.log( prob_of_H0 ) # LLR for input0

            		else:
            			continue
                    
            	except:
            		continue

                    

            if ok_LLR:
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
    t = "{} measurements / experiment".format(Nmeas)
    weights0 = np.ones_like(array0)/N0
    weights1 = np.ones_like(array1)/N1

    array0.sort()
    array1.sort()
    L_alpha = array0[int(len(array0)*(1-alpha))]
    res = next(i for i,v in enumerate(array1) if v > L_alpha)
    beta = res/len(array1)



    plt.figure()
    ax = plt.axes()
    #plt.hist(bin_val,100,density=True, color='r',alpha=0.5)
    plt.hist(array0,10, weights=weights0, density=True, color='r',alpha=0.5,label='$P(\lambda | Input0)$')
    plt.hist(array1, 10, weights=weights1, density=True, color='b', alpha=0.5,label='$P(\lambda | Input1)$')
    plt.axvline(L_alpha, color='r', linewidth=1, label='$\\lambda_\\alpha$')
    plt.plot([], [], ' ', label="$\\alpha = $"+str(alpha))
    plt.plot([], [], ' ', label="$\\beta = $"+str(beta) ) 
    plt.legend()
    plt.xlabel('$\\lambda = \\log({\\cal L}_{\\mathbb{H}_{1}}/{\\cal L}_{\\mathbb{H}_{0}})$')
    plt.ylabel('Probability')
    plt.title(t)
    plt.grid(True)
    plt.show()    
    plt.savefig("Project2-Figure.png")


    plt.figure()
    ax = plt.axes()
    #plt.hist(bin_val,100,density=True, color='r',alpha=0.5)
    plt.hist(array0,10, weights=weights0, density=True, color='r',alpha=0.5,label='$P(\lambda | Input0)$')
    plt.axvline(L_alpha, color='r', linewidth=1, label='$\\lambda_\\alpha$')
    plt.plot([], [], ' ', label="$\\alpha = $"+str(alpha))
    plt.plot([], [], ' ', label="$\\beta = $"+str(beta) ) 
    plt.legend()
    plt.xlabel('')
    plt.ylabel('Probability')
    plt.title(t)
    plt.grid(True)
    plt.show()    
    plt.savefig("model0-Project2-Figure.png")

    ax = plt.axes()
    #plt.hist(bin_val,100,density=True, color='r',alpha=0.5)
    plt.hist(array1, 10, weights=weights1, density=True, color='b', alpha=0.5,label='$P(\lambda | Input1)$')
    plt.axvline(L_alpha, color='r', linewidth=1, label='$\\lambda_\\alpha$')
    plt.plot([], [], ' ', label="$\\alpha = $"+str(alpha))
    plt.plot([], [], ' ', label="$\\beta = $"+str(beta) ) 
    plt.legend()
    plt.xlabel('')
    plt.ylabel('Probability')
    plt.title(t)
    plt.grid(True)
    plt.show()    
    plt.savefig("model1-Project2-Figure.png")



