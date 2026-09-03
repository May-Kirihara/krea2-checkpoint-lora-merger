python merge_krea2_lora.py \
    --base "/Path/To/Your/checkpoint.safetensors" \
    --lora "/Path/To/Your/LoRA1.safetensors" \
           "/Path/To/Your/LoRA2.safetensors" \
           "/Path/To/Your/LoRA3.safetensors" \
           "/Path/To/Your/LoRA4.safetensors" \
    --multiplier 1.0 0.7 0.9 0.9\
    --output "merged/mergedmodel.safetensors"