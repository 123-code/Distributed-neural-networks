#!/usr/bin/env python3

import sys
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_loading():
    """Prueba cargar el modelo Qwen localmente"""
    print("=== Prueba de Carga del Modelo ===")
    
    try:
        model_name = "Qwen/Qwen2-0.5B-Instruct"
        print(f"Cargando modelo: {model_name}")
        
        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer cargado")
        
        # Cargar modelo
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        print("✓ Modelo cargado")
        
        # Prueba simple de generación
        text = "Hola, ¿cómo estás?"
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generación exitosa: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def test_distributed_imports():
    """Prueba importar las librerías distribuidas"""
    print("\n=== Prueba de Imports Distribuidos ===")
    
    try:
        import torch.distributed as dist
        print("✓ torch.distributed")
        
        import torch.distributed.rpc as rpc
        print("✓ torch.distributed.rpc")
        
        from worker import QwenWorker
        print("✓ QwenWorker")
        
        from coordinator import QwenCoordinator  
        print("✓ QwenCoordinator")
        
        from utils import setup_logging, print_system_info
        print("✓ utils")
        
        return True
        
    except Exception as e:
        print(f"✗ Error en imports: {str(e)}")
        return False

def test_system_info():
    """Muestra información del sistema"""
    print("\n=== Información del Sistema ===")
    
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    import psutil
    print(f"RAM total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"RAM disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB")

def main():
    print("=== Test del Sistema Distribuido Qwen ===\n")
    
    # Pruebas
    test_system_info()
    
    if not test_distributed_imports():
        print("\n❌ Falló la prueba de imports distribuidos")
        return False
    
    print("\n🔄 Probando carga del modelo (esto puede tomar tiempo)...")
    if not test_model_loading():
        print("\n❌ Falló la prueba de carga del modelo")
        return False
    
    print("\n✅ Todas las pruebas pasaron exitosamente!")
    print("\nEl sistema está listo para ejecutarse.")
    print("\nPara ejecutar:")
    print("1. python run_local.py")
    print("2. O usar los scripts manuales según el README")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 