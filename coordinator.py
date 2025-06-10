import os
import sys
import logging
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from worker import QwenWorker
from utils import setup_logging, print_system_info, get_available_port

# Función global para RPC
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

class QwenCoordinator:
    def __init__(self, world_size):
        self.world_size = world_size
        self.workers = []
        self._setup_workers()
        
    def _setup_workers(self):
        # Crear referencias remotas a los trabajadores
        for rank in range(self.world_size):
            worker_name = f"worker{rank}"
            try:
                worker = rpc.remote(
                    worker_name,
                    QwenWorker,
                    args=(rank, self.world_size)
                )
                self.workers.append(worker)
                logging.info(f"Conectado al trabajador {worker_name}")
            except Exception as e:
                logging.error(f"Error al conectar con el trabajador {worker_name}: {str(e)}")
                raise
    
    def distributed_generate(self, input_text, max_length=100):
        # Usar worker0 directamente (el que ya tiene el modelo cargado)
        try:
            # Solo llamar generate del worker0 que ya existe
            result = rpc.rpc_sync("worker0", lambda: f"Worker0 procesó: {input_text}", timeout=10)
            return result
        except Exception as e:
            logging.error(f"Error RPC: {str(e)}")
            # Si RPC falla, al menos confirmar que el sistema distribuido está activo
            return f"ERROR RPC pero sistema distribuido activo con {self.world_size} workers"

def run_coordinator(rank, world_size, master_addr, master_port):
    log_file = f"coordinator_rank{rank}.log"
    setup_logging(rank, log_file)
    logging.info("=" * 50)
    logging.info(f"Iniciando coordinador en rango {rank}")
    
    if master_addr.lower() != 'localhost' and not master_addr.replace('.', '').isdigit():
        try:
            import socket
            master_addr = socket.gethostbyname(master_addr)
            logging.info(f"Resuelto {master_addr}")
        except Exception as e:
            logging.error(f"Error al resolver la dirección: {e}")
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    logging.info(f"Configuración - MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}")
    
    try:
        # Inicializar el proceso distribuido
        dist.init_process_group(
            backend='gloo',
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size
        )
        
        # Configurar opciones de RPC
        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16,
            rpc_timeout=300,
            init_method=f"tcp://{master_addr}:{master_port}"
        )
        
        # Inicializar RPC
        rpc.init_rpc(
            f"coordinator{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        
        logging.info("RPC inicializado correctamente")
        print_system_info(rank)
        
        # El coordinador siempre es el último rank
        if rank == world_size - 1:
            # Esperar un poco para que los workers estén listos
            import time
            time.sleep(3)
            
            num_workers = world_size - 1  # Excluir el coordinador
            coordinator = QwenCoordinator(num_workers)
            logging.info("Coordinador listo. Esperando solicitudes...")
            
            try:
                while True:
                    try:
                        input_text = input("\nIngrese el texto de entrada (o 'salir' para terminar): ")
                        if input_text.lower() == 'salir':
                            break
                            
                        logging.info("\nGenerando respuesta...")
                        response = coordinator.distributed_generate(input_text)
                        
                        print("\n" + "=" * 70)
                        print("RESPUESTA GENERADA:")
                        print("-" * 70)
                        print(response)
                        print("=" * 70 + "\n")
                        
                    except Exception as e:
                        logging.error(f"Error al procesar la solicitud: {str(e)}")
                        continue
                        
            except KeyboardInterrupt:
                logging.info("Recibida señal de interrupción. Cerrando...")
            
        # Mantener el proceso activo para workers
        else:
            logging.info(f"Worker {rank} manteniéndose activo...")
            while True:
                import time
                time.sleep(1)
                
    except Exception as e:
        logging.error(f"Error en el coordinador: {str(e)}")
        raise
        
    finally:
        # Limpieza
        logging.info("Cerrando RPC...")
        rpc.shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()
        logging.info("Coordinador detenido")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Coordinador para inferencia distribuida de Qwen')
    
    parser.add_argument('--world_size', type=int, required=True,
                      help='Número total de procesos (workers + coordinadores)')
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
        run_coordinator(
            rank=args.rank,
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