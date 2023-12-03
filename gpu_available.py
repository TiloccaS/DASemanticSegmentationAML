import torch

# Verifica se CUDA (GPU support) è disponibile
if torch.cuda.is_available():
    # Ottieni il numero di GPU disponibili
    num_gpus = torch.cuda.device_count()
    
    print(f"GPU disponibili: {num_gpus}")
    for i in range(num_gpus):
        print(f"- GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Nessuna GPU disponibile.")



