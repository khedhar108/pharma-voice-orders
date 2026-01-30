"""
Runtime resource configuration for CPU/GPU optimization.
Prevents system crashes when running large Whisper models on CPU.
"""

import os

import torch


def configure_runtime(model_id: str = "openai/whisper-tiny") -> dict:
    """
    Configures runtime resources based on model size and available hardware.
    
    For CPU inference:
    - Limits threads to 4-6 to prevent thermal throttling and crashes
    - Sets OMP/MKL threads for BLAS operations
    
    For GPU inference:
    - No thread limiting needed, CUDA handles parallelism
    
    Returns:
        dict with configuration details for logging
    """
    is_gpu = torch.cuda.is_available()
    
    if is_gpu:
        # GPU mode: no CPU thread limiting needed
        config = {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "threads": "N/A (GPU)",
        }
        print(f"[Runtime] GPU Mode → {config['gpu_name']} ({config['vram_gb']} GB VRAM)")
        return config
    
    # CPU mode: apply thread limiting based on model size
    # Extract model size from model_id (e.g., "openai/whisper-large-v3" → "large")
    model_size = "tiny"
    for size in ["large", "medium", "small", "tiny"]:
        if size in model_id.lower():
            model_size = size
            break
    
    # Thread allocation strategy:
    # - Larger models need more RAM per thread, so use fewer threads
    # - Smaller models can use more threads for parallelism
    thread_config = {
        "tiny": 6,    # Light model, can parallelize more
        "small": 6,   # Still light enough
        "medium": 5,  # Moderate memory usage
        "large": 4,   # Heavy model, limit threads to save RAM
    }
    
    threads = thread_config.get(model_size, 4)
    
    # Apply thread limits
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer thread contention
    
    config = {
        "device": "cpu",
        "model_size": model_size,
        "threads": threads,
        "cpu_count": os.cpu_count(),
    }
    
    print(f"[Runtime] CPU Mode → {threads} threads (of {os.cpu_count()} available) for '{model_size}' model")
    
    return config
