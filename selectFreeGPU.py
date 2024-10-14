import os
import numpy as np
import torch
import subprocess
import re

def get_gpu_utilization():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
        utilization = [int(x) for x in output.decode('utf-8').strip().split('\n')]
        return utilization
    except:
        print("Error getting GPU utilization. Make sure nvidia-smi is available.")
        return None

def get_free_gpu():
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("No GPUs available. Using CPU.")
        return torch.device("cpu")
    
    # Get GPU utilization
    utilization = get_gpu_utilization()
    
    # Check memory usage and utilization for each GPU
    gpu_scores = []
    for i in range(num_gpus):
        try:
            total_memory = torch.cuda.get_device_properties(i).total_memory
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            available_mem = total_memory - (memory_allocated + memory_reserved)
            available_mem_gb = available_mem / 1024**3
            
            util = utilization[i] if utilization else 0
            
            # Calculate a score based on available memory and utilization
            # We prioritize GPUs with more available memory and lower utilization
            score = available_mem_gb * (100 - util)
            
            gpu_scores.append(score)
            print(f"GPU {i}: {available_mem_gb:.2f} GB available, {util}% utilized, Score: {score:.2f}")
        except Exception as e:
            print(f"Error checking GPU {i}: {str(e)}")
            gpu_scores.append(0)  # Consider this GPU as not suitable
    
    # Find the GPU with the highest score
    best_gpu_id = np.argmax(gpu_scores)
    
    print(f"Selected GPU {best_gpu_id} with score {gpu_scores[best_gpu_id]:.2f}")
    return torch.device(f"cuda:{best_gpu_id}")

device = get_free_gpu()
print(f"Using device: {device}")

