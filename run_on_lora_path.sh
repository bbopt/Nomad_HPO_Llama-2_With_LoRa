for i in ../lora_output/*
do	
    echo $i
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun test_eval_with_LoRa.py --model_name_or_path meta-llama/Llama-2-7b --data_paths ../stanford_alpaca/alpaca_data.json,databricks/databricks-dolly-15k --proportions 0.7,0.3 databricks/databricks-dolly-15k --lora_path $i --output_dir ./output_dir --bf16 True --do_eval True --per_device_eval_batch_size 30 --do_train False --gradient_accumulation_steps 8 --evaluation_strategy "steps" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --logging_steps 1 --tf32 True
    date
    echo '########################################################################'
done
