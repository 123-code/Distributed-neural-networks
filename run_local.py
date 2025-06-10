import subprocess
import time
import sys
import os
import signal

def run_qwen_distributed():
    print("🚀 Iniciando sistema distribuido de Qwen con 2 workers...")
    print()
    
    # Configuración
    world_size = 2
    master_addr = "localhost"
    master_port = "29500"
    
    processes = []
    
    try:
        # Iniciar workers en background
        for rank in range(world_size):
            print(f"⚡ Iniciando worker {rank}...")
            cmd = [
                sys.executable, "worker.py",
                "--world_size", str(world_size + 1),  # +1 para incluir el coordinador
                "--rank", str(rank),
                "--master_addr", master_addr,
                "--master_port", master_port
            ]
            
            # Redirigir salida a archivos
            stdout_file = open(f"worker_{rank}_stdout.log", "w")
            stderr_file = open(f"worker_{rank}_stderr.log", "w")
            
            proc = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True
            )
            processes.append((proc, stdout_file, stderr_file))
            
            time.sleep(3)  # Tiempo para inicialización
        
        print("✅ Workers iniciados. Esperando inicialización completa...")
        time.sleep(8)
        
        # Iniciar coordinador en foreground
        print("🎯 Iniciando coordinador...")
        print("=" * 60)
        cmd = [
            sys.executable, "coordinator.py",
            "--world_size", str(world_size + 1),  # +1 para incluir el coordinador
            "--rank", str(world_size),  # El coordinador es el último rank
            "--master_addr", master_addr,
            "--master_port", master_port
        ]
        
        # El coordinador corre en primer plano para interacción
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo sistema...")
    
    finally:
        # Limpiar procesos
        print("🧹 Limpiando procesos...")
        for proc, stdout_file, stderr_file in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            finally:
                stdout_file.close()
                stderr_file.close()
        
        print("✅ Sistema detenido completamente.")

if __name__ == "__main__":
    print("=" * 60)
    print("        SISTEMA DISTRIBUIDO QWEN")
    print("=" * 60)
    run_qwen_distributed() 