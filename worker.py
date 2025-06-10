import os
import sys
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import setup_logging, print_system_info

# Función global para RPC (debe estar en ambos módulos)
def simple_generate(text, max_len):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import os
    
    # Deshabilitar autocast para evitar errores en CPU
    os.environ['TORCH_DISABLE_AUTOCAST'] = '1'
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Cargar modelo sin mixed precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,  # Sin device_map automático
        use_cache=False   # Deshabilitar cache para simplificar
    )
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        # Generar con configuración simple
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_len,
            do_sample=False,  # Usar greedy para evitar problemas
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

class QwenWorker(nn.Module):
    def __init__(self, worker_rank, world_size):
        super().__init__()
        self.worker_rank = worker_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{worker_rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
        
        # Cargar modelo
        model_name = "Qwen/Qwen2-0.5B-Instruct"
        print(f"Worker {worker_rank}: Cargando modelo {model_name}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": self.device},
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Solo el worker 0 carga el tokenizer
        if worker_rank == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Calcular qué capas maneja este worker (excluyendo el coordinador del world_size)
        total_layers = len(self.model.model.layers)
        num_workers = world_size - 1  # Excluir el coordinador
        layers_per_worker = total_layers // num_workers
        start_layer = worker_rank * layers_per_worker
        end_layer = start_layer + layers_per_worker if worker_rank < num_workers - 1 else total_layers
        
        self.layer_indices = list(range(start_layer, end_layer))
        logging.info(f"Worker {worker_rank} manejará capas {start_layer} a {end_layer-1}")

    def forward(self, hidden_states, layer_indices=None):
        if layer_indices is None:
            layer_indices = self.layer_indices
            
        with torch.no_grad():
            # Mover hidden_states al dispositivo correcto
            if isinstance(hidden_states, torch.Tensor):
                hidden_states = hidden_states.to(self.device)
            
            # Procesar solo las capas asignadas
            for i in layer_indices:
                if i < len(self.model.model.layers):
                    layer = self.model.model.layers[i]
                    hidden_states = layer(hidden_states)[0]
        
        return hidden_states

    def generate(self, input_ids, attention_mask, max_length=100):
        # Solo el worker 0 maneja la generación completa
        if self.worker_rank != 0:
            return None
            
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return output

def run_worker(worker_rank, world_size, master_addr, master_port):
    log_file = f"worker_rank{worker_rank}.log"
    setup_logging(worker_rank, log_file)
    logging.info("=" * 50)
    logging.info(f"Iniciando trabajador en rango {worker_rank}")
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    try:
        # Inicializar proceso distribuido
        dist.init_process_group(
            backend='gloo',
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=worker_rank,
            world_size=world_size
        )
        
        # Configurar RPC
        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16,
            rpc_timeout=300,
            init_method=f"tcp://{master_addr}:{master_port}"
        )
        
        # Inicializar RPC
        rpc.init_rpc(
            f"worker{worker_rank}",
            rank=worker_rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        
        logging.info(f"Trabajador {worker_rank} listo")
        print_system_info(worker_rank)
        
        # Mantener el worker activo
        while True:
            pass
            
    except Exception as e:
        logging.error(f"Error en el trabajador {worker_rank}: {str(e)}")
        raise
        
    finally:
        logging.info(f"Cerrando trabajador {worker_rank}...")
        rpc.shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()
        logging.info(f"Trabajador {worker_rank} detenido")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Trabajador para inferencia distribuida de Qwen')
    parser.add_argument('--world_size', type=int, required=True,
                      help='Número total de procesos')
    parser.add_argument('--rank', type=int, required=True,
                      help='Rango del proceso actual')
    parser.add_argument('--master_addr', type=str, default='localhost',
                      help='Dirección IP del nodo maestro')
    parser.add_argument('--master_port', type=str, default='29500',
                      help='Puerto para la comunicación')
    
    args = parser.parse_args()
    
    try:
        master_port = int(args.master_port)
    except ValueError:
        print(f"Error: El puerto debe ser un número entero")
        return
    
    try:
        run_worker(
            worker_rank=args.rank,
            world_size=args.world_size,
            master_addr=args.master_addr,
            master_port=master_port
        )
    except Exception as e:
        print(f"Error fatal: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 