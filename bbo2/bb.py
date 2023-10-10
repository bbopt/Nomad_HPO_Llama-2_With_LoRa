import time
# import logging
import sys
import os
import numpy as np
import time

# Output file to parse
single_output_base_eval = 'output_eval'

previousHistoryFileIsProvided = False
previousHistoryFileName = 'history.prev.txt'

# Time stamp for file tag
localtime = time.localtime()
localtime_str = str(localtime.tm_year) + '_' + str(localtime.tm_mon)  + '_' + str(localtime.tm_mday)  + '_' + str(localtime.tm_hour)  + 'h' + str(localtime.tm_min)  + 'm' + str(localtime.tm_sec) + 's'
# localtime_str = '2023_9_25_11h46m54s'

if __name__ == '__main__':

    inputFileName=sys.argv[1]
    X=np.fromfile(inputFileName,sep=" ")

    if previousHistoryFileIsProvided:
        dim = len(X)
        # Read a history file from a run
        rawEvals = np.fromfile(previousHistoryFileName,sep=" ")
        # Each line contains 2 values for X and a single output (objective value)
        nbRows = rawEvals.size/(len(X)+1)
        # print(nbRows)
        # Split the whole array is subarrays
        npEvals = np.split(rawEvals,nbRows)
        for oneEval in npEvals:
            # print('Dim=',dim,' ',oneEval)
            diffX=X-oneEval[0:dim]
            # print(np.linalg.norm(diffX))
            if np.linalg.norm(diffX) < 1E-10:
                fout=oneEval[dim]
                print(fout)
                # print('Find point in file: ',X,' f(x)=',fout)
                exit()


    # Output file name to be parsed for objective function 
    single_output_eval = './' + single_output_base_eval + '_' + localtime_str 

    # Start eval
    syst_cmd = 'python3 eval.py ' + inputFileName + ' ' + localtime_str + ' > ' + single_output_eval
    os.system(syst_cmd)

    # Default value
    bbOutput='Inf'
    with open(single_output_eval, "r") as file:
        lines = file.readlines()
        train_loss = lines[len(lines)-1]["train_loss"]
        eval_loss = lines[len(lines)-2]["eval_loss"]
        train_runtime = lines[len(lines)-1]["train_runtime"]
        eval_runtime = lines[len(lines)-2]["eval_runtime"]
        file.close()
        print(eval_loss, train_loss, eval_runtime, train_runtime, end="\n")
        
