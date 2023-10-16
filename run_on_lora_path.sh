for i in ../christophe/lora_output/*
do	
    echo $i
    torchrun test_eval_with_LoRa.py --model_name_or_path meta-llama/Llama-2-7b-hf --data_path databricks/databricks-dolly-15k --lora_path $i --output_dir ./output_dir --bf16 True --do_eval True --per_device_eval_batch_size 30 --do_train False --gradient_accumulation_steps 8 --evaluation_strategy "steps" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --logging_steps 1
    date
    echo '########################################################################' 
done
