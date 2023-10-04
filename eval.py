import time
import sys
import os
import numpy as np
import time

# Arguments to pass for alpaca training
alpaca_working_dir = '/local1/ctribes/stanford_alpaca-main'
cuda_visible_devices = '0,1,2,3'
nbGPU = 4
model_name_or_path = 'meta-llama/Llama-2-7b-hf'
output_dir_base_alpaca = 'output_dir_LoRa'
num_train_epochs = 0.01
train_batch_size = 4


# Arguments to pass for Instruct-Eval
instruc_eval_dir = '/local1/ctribes/instruct-eval'
single_output_base_instructeval = 'output_instructeval'

# Log file
logAllFileBase = 'logAllOutputs'


# This function must be customized to the problem (order and value of the variables)
def get_parameters_from_Nomad_input(x):
    
    config = dict()
    
#1    "lora_rank": {"_type":"choice", "_value": [4, 8, 16, 32, 64, 128]},
#2    "lora_dropout": {"_type":"choice", "_value": [0, 0.0001,  0.001, 0.01, 0.1, 1]},
#3    "lora_alpha": {"_type":"choice", "uniform":[1, 64]},
#4    "lr":{"_type":"choice", "uniform":[-6, -3]}, 

    
    # 1 -> lora rank
    if x[0] == 1:
        valueR = 4
    elif x[0] == 2:
        valueR = 8
    elif x[0] == 3:
        valueR = 16
    elif x[0] == 4:
        valueR = 32
    elif x[0] == 5:
        valueR = 64
    elif x[0] == 6:
        valueR = 128
    else:
        return config
    config['lora_rank'] = valueR
     
    # 2 -> lora dropout
    if x[1] == 1:
        valueLD = 0
    elif x[1] == 2:
        valueLD = 0.0001
    elif x[1] == 3:
        valueLD = 0.001
    elif x[1] == 4:
        valueLD = 0.01
    elif x[1] == 5:
        valueLD = 0.1
    elif x[1] == 6:
        valueLD = 1.0
    else:
        return config
    config['lora_dropout'] = valueLD
    
    
    # !!!!!
    # 3 -> LoRa alpha
    # !!!!!
    if x[2] > 0:
        valueAl = int(x[2])
    else:
        return config
    # print(config)
    config['lora_alpha'] = valueAl

    # !!!!!
    # 4 -> learning rate: lr = 10^x4
    # !!!!!
    if x[3] < 0:
        valueLR = pow(10,x[3])
    else:
        return config
    # print(config)
    config['lr'] = valueLR

    return config



if __name__ == '__main__':

    inputFileName=sys.argv[1]
    X=np.fromfile(inputFileName,sep=" ")

    inputFileID = inputFileName.split(".")[2]

    localtime_str = sys.argv[2]

    args = get_parameters_from_Nomad_input(X)
    # print(args)
    
    # Base optimization directory
    base_dir = os.getcwd()

    # Log file
    logAllFile = logAllFileBase + '_' + inputFileID + '_' + localtime_str + '.txt'

    # Clean up log files and append input file into log file
    syst_cmd_append  =  'echo #############Alpaca################### >> ' + logAllFile + ' ; cat ' + inputFileName + ' >> ' + logAllFile
    os.system(syst_cmd_append)

    # Alpaca output_dir
    output_dir_alpaca = base_dir + '/' + output_dir_base_alpaca + '_' + inputFileID + '_' + localtime_str + '/'
    
    # Change to alpaca working dir
    os.chdir(alpaca_working_dir)

    syst_cmd_alpaca  = ' source .env/bin/activate ; export HF_HOME=/local1/ctribes/huggingface/ ; export WANDB_MODE=offline ;'
    syst_cmd_alpaca += ' CUDA_VISIBLE_DEVICES=' + cuda_visible_devices + ' torchrun -r 3 --log_dir=' + base_dir + '/TMPlogDirTorchRunAlpaca --nproc_per_node=' + str(nbGPU)
    syst_cmd_alpaca += ' train_with_LoRa.py --model_name_or_path ' + str(model_name_or_path) +' --data_path ./alpaca_data.json --bf16 True '
    syst_cmd_alpaca += ' --output_dir ' + str(output_dir_alpaca) + '  --num_train_epochs ' + str(num_train_epochs) 
    syst_cmd_alpaca += ' --per_device_train_batch_size ' + str(train_batch_size) + ' --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 '
    syst_cmd_alpaca += ' --learning_rate ' + str(args['lr']) +' --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True '
    syst_cmd_alpaca += ' --lora_rank ' + str(args['lora_rank']) 
    syst_cmd_alpaca += ' --lora_dropout '+ str(args['lora_dropout']) 
    syst_cmd_alpaca += ' --lora_alpha ' + str(args['lora_alpha']) 
    syst_cmd_alpaca += ' ; deactivate '

    print(syst_cmd_alpaca)

    os.system(syst_cmd_alpaca)


    os.chdir(base_dir)

    syst_cmd_append  =  'echo ############ Instruct Eval  ################## >> ' + logAllFile
    os.system(syst_cmd_append)

    # Change to Instruct Eval working dir
    os.chdir(instruc_eval_dir)

    # Change output file name
    single_output_instructeval = base_dir + '/' + single_output_base_instructeval + '_' + inputFileID + '_' + localtime_str + '.txt'

    syst_cmd_instructeval  = ' module load anaconda ; conda activate ./condaenv ; export HF_HOME=/local1/ctribes/huggingface/ ; '
    syst_cmd_instructeval += ' export PYTHONNOUSERSITE=1 ; '
    syst_cmd_instructeval += ' python main.py mmlu --model_name llama --model_path ' + model_name_or_path + ' --lora_path ' + output_dir_alpaca  +  ' > ' + single_output_instructeval + ' 2>&1 ; '
    syst_cmd_instructeval += ' conda deactivate '
    os.system(syst_cmd_instructeval)

    # append file into log file
    syst_cmd_append =  'cat ' + single_output_instructeval + ' >> ' +  base_dir + '/' + logAllFile
    os.system(syst_cmd_append)

    os.chdir(base_dir)

