# bash it in root path
PYTHON_PATH='./' accelerate launch --multi_gpu --gpu_ids '0,1,2,3' --main_process_port 25011 --num_processes 4 train/train_meissonic.py \
        --output_dir  "../CKPT_OUTPUT_PATH" \
        --train_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-4 \
        --max_grad_norm 10 \
        --pretrained_model_name_or_path "meissonflow/meissonic" \
        --text_encoder_architecture 'open_clip' \
        --pretrained_model_architecture 'Meissonic' \
        --training_from_scratch True \
        --instance_dataset 'DATA_TYPE' \
        --instance_data_dir  '../parquets_father_dir/' \
        --resolution 1024 \
        --mixed_precision fp16 \
        --lr_scheduler constant \
        --use_8bit_adam \
        --dataloader_num_workers 64 \
        --validation_prompts \
            'a boy' \
            'A serene mountain landscape with towering snow-capped peaks, a crystal-clear blue lake reflecting the mountains, dense pine forests, and a vibrant orange sunrise illuminating the sky.' \
            'A playful golden retriever puppy with a shiny coat, bounding through a meadow filled with colorful wildflowers, under a bright, clear blue sky.' \
            'A bustling city street at night, illuminated by vibrant neon signs in various colors, with busy pedestrians, street vendors, and a light rain creating reflective puddles on the pavement.' \
            'A majestic, medieval castle perched on a rugged cliffside, overlooking a vast, calm ocean at sunset, with the sky painted in hues of pink, orange, and purple.' \
            'An elegant ballerina in a white tutu, dancing gracefully on a grand stage with ornate, gold-trimmed curtains, under a spotlight that casts a soft glow.' \
            'A cozy, rustic log cabin nestled in a snow-covered forest, with smoke rising from the stone chimney, warm lights glowing from the windows, and a path of footprints leading to the front door.'\
            'A Cute Cat' \
            'A Snow Mountain'\
        --max_train_steps 100000 \
        --checkpointing_steps 1000 \
        --validation_steps 200 \
        --report_to 'wandb' \
        --logging_steps 10