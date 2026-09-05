@echo off
python merge_krea2_lora.py ^
    --base "C:\Path\To\Your\checkpoint.safetensors" ^
    --lora "C:\Path\To\Your\LoRA1.safetensors" ^
           "C:\Path\To\Your\LoRA2.safetensors" ^
           "C:\Path\To\Your\LoRA3.safetensors" ^
           "C:\Path\To\Your\LoRA4.safetensors" ^
    --multiplier 1.0 0.7 0.9 0.9 ^
    --output "merged\mergedmodel.safetensors"

pause
