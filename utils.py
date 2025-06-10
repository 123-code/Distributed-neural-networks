import os
import torch
import logging
from typing import List, Dict, Any, Optional

def setup_logging(rank: int, log_file: str = None):
    log_format = f'[%(asctime)s] [Rango {rank}] [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

def check_gpu_available() -> Dict[str, Any]:
    gpu_info = {
        'available': torch.cuda.is_available(),
        'count': torch.cuda.device_count(),
        'devices': []
    }
    
    if gpu_info['available']:
        for i in range(gpu_info['count']):
            gpu_info['devices'].append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory / 1024**3,
                'memory_reserved': torch.cuda.memory_reserved(i) / 1024**3 if i < torch.cuda.device_count() else 0,
                'memory_allocated': torch.cuda.memory_allocated(i) / 1024**3 if i < torch.cuda.device_count() else 0
            })
    
    return gpu_info

def print_system_info(rank: int):
    import platform
    import psutil
    
    logging.info(f"Sistema operativo: {platform.system()} {platform.release()}")
    logging.info(f"Procesador: {platform.processor()}")
    logging.info(f"RAM Total: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logging.info(f"RAM Disponible: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    gpu_info = check_gpu_available()
    if gpu_info['available']:
        logging.info(f"GPUs disponibles: {gpu_info['count']}")
        for gpu in gpu_info['devices']:
            logging.info(f"  GPU {gpu['id']}: {gpu['name']}")
            logging.info(f"    Memoria Total: {gpu['memory_total']:.2f} GB")
            logging.info(f"    Memoria Reservada: {gpu['memory_reserved']:.2f} GB")
            logging.info(f"    Memoria En Uso: {gpu['memory_allocated']:.2f} GB")
    else:
        logging.warning("No se detectaron GPUs disponibles. Se usará CPU.")

def split_layers_across_workers(num_layers: int, num_workers: int) -> List[List[int]]:
    layers_per_worker = num_layers // num_workers
    remainder = num_layers % num_workers
    
    layer_distribution = []
    start_idx = 0
    
    for i in range(num_workers):
        end_idx = start_idx + layers_per_worker + (1 if i < remainder else 0)
        layer_distribution.append(list(range(start_idx, end_idx)))
        start_idx = end_idx
    
    return layer_distribution

def check_ports_available(host: str, start_port: int, num_ports: int) -> bool:
    import socket
    
    for port in range(start_port, start_port + num_ports):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.bind((host, port))
        except (OSError, socket.error):
            return False
    return True

def get_available_port(host: str, start_port: int = 29500, max_attempts: int = 100) -> int:
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.bind((host, port))
                return port
        except (OSError, socket.error):
            continue
    
    raise RuntimeError(f"No se pudo encontrar un puerto disponible después de {max_attempts} intentos") 