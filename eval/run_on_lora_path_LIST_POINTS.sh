export CUDA_VISIBLE_DEVICES=7
module load anaconda ; conda activate ./condaenv ; export HF_HOME=/local1/ctribes/huggingface/
export PYTHONNOUSERSITE=1

OptimPath="../Nomad_HPO_Llama-2-7B-HF_With_LoRa-3"

List="output_dir_LoRa_2023_10_31_10h25m5s" 

for i in $List
do	
    echo $i
    lora_path=$OptimPath/$i
    python main.py drop --model_name llama --model_path meta-llama/Llama-2-7b-hf --lora_path $lora_path > ${lora_path/dir\_LoRa/"drop"} 2>&1
    echo "################"
done

List="output_dir_LoRa_2023_10_31_10h25m5s output_dir_LoRa_2023_10_30_9h19m9s output_dir_LoRa_2023_10_30_17h56m57s output_dir_LoRa_2023_10_31_7h48m49s output_dir_LoRa_2023_10_29_22h58m42s output_dir_LoRa_2023_10_30_7h35m35s output_dir_LoRa_2023_10_30_3h17m2s output_dir_LoRa_2023_10_30_20h33m49s output_dir_LoRa_2023_10_30_18h49m9s output_dir_LoRa_2023_10_30_12h47m13s output_dir_LoRa_2023_10_31_6h56m42s"  

for i in $List
do
    echo $i
    lora_path=$OptimPath/$i
    python main.py humaneval --model_name llama --model_path meta-llama/Llama-2-7b-hf --lora_path $lora_path > ${lora_path/dir\_LoRa/"humaneval"} 2>&1
    echo "################"
done

