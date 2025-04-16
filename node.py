
import argparse
import grpc
import asyncio
import json
import node_service_pb2
import node_service_pb2_grpc
import numpy as np
import torch
from cifar_model_parts import NeuralNetwork, ModelPart0, ModelPart1, ModelPart2 
import torchvision.transforms as transforms
from PIL import Image
import traceback

NODE_ID = None
MY_ADDRESS = None
MY_PORT = None    
MY_PART_INDEX = -1
NEXT_NODE_ADDRESS = None
FIRST_NODE_ADDRESS = None
IS_LAST_NODE = False
MODEL_WEIGHTS_PATH = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
my_model_part = None

MODEL_PARTS_CLASSES = {
    0: ModelPart0,
    1: ModelPart1,
    2: ModelPart2
}


class NodeServiceImpl(node_service_pb2_grpc.NodeServiceServicer):
    async def SendTensor(self, request: node_service_pb2.TensorRequest, context):
        global my_model_part, NEXT_NODE_ADDRESS, IS_LAST_NODE, FIRST_NODE_ADDRESS

        print(f"\n[{NODE_ID}] Received tensor request_id: {request.request_id}")
        
        print(f"[{NODE_ID}] Incoming Tensor - shape: {list(request.tensor.shape)}, dtype: {request.tensor.dtype}")

        final_output_tensor_proto = None 
        status_message = f"[{NODE_ID}] Error processing tensor."
        try:
      
            input_np = np.frombuffer(
                request.tensor.tensor_data, 
                dtype=np.dtype(request.tensor.dtype)
            ).reshape(request.tensor.shape)
            input_torch = torch.from_numpy(input_np).to(DEVICE)
            print(f"[{NODE_ID}] Deserialized input tensor shape: {input_torch.shape}")

           
            with torch.no_grad():
                output_torch = my_model_part(input_torch)
            print(f"[{NODE_ID}] Computed output tensor shape: {output_torch.shape}")
            output_np = output_torch.cpu().numpy()

     
            if IS_LAST_NODE:
      
                print(f"[{NODE_ID}] Reached final node.")
                predicted_class_index = np.argmax(output_np)
                print(f"[{NODE_ID}] Final Prediction Index: {predicted_class_index}")
                status_message = f"[{NODE_ID}] Processing complete. Prediction: {predicted_class_index}"
             
                final_output_tensor_proto = node_service_pb2.Tensor( 
                    tensor_data=output_np.tobytes(),
                    shape=list(output_np.shape),
                    dtype=str(output_np.dtype)
                )


            else:
           
                print(f"[{NODE_ID}] Forwarding tensor to next node: {NEXT_NODE_ADDRESS}")
                async with grpc.aio.insecure_channel(NEXT_NODE_ADDRESS) as channel:
                    stub = node_service_pb2_grpc.NodeServiceStub(channel)
                    next_request = node_service_pb2.TensorRequest(
                        request_id=request.request_id,
                        tensor=node_service_pb2.Tensor(
                            tensor_data=output_np.tobytes(),
                            shape=list(output_np.shape),
                            dtype=str(output_np.dtype)
                        )
                    )
                    try:
                        next_response = await stub.SendTensor(next_request)
                        print(f"[{NODE_ID}] Response from next node ({NEXT_NODE_ADDRESS}): {next_response.status}")
                        status_message = f"[{NODE_ID}] Forwarded. Next node status: {next_response.status}"
                        
                        if next_response.HasField("output_tensor"):
                            final_output_tensor_proto = next_response.output_tensor
                    except grpc.aio.AioRpcError as e:
                         print(f"!!! [{NODE_ID}] Error calling SendTensor on next node ({NEXT_NODE_ADDRESS}): {e.code()} - {e.details()}")
                         status_message = f"[{NODE_ID}] Error forwarding: {e.details()}"

        except Exception as e:
            print(f"!!! [{NODE_ID}] Error processing tensor: {e}")
            traceback.print_exc()
            status_message = f"[{NODE_ID}] Error: {e}"

       
        return node_service_pb2.TensorResponse(
            status=status_message,               
            output_tensor=final_output_tensor_proto 
        )

    async def HealthCheck(self, request, context):
        print(f"[{NODE_ID}] Health check requested")
        return node_service_pb2.HealthCheckResponse(is_healthy=True)

    async def SendMessage(self, request, context):
        print(f"[{NODE_ID}] Received message from {request.sender_id}")
        return node_service_pb2.MessageReply(confirmation_text=f"[{NODE_ID}] got msg '{request.message_text}'")


async def serve():
    server = grpc.aio.server()
    node_service_pb2_grpc.add_NodeServiceServicer_to_server(NodeServiceImpl(), server)


    listen_addr = f'[::]:{MY_PORT}' 


    try:
        server.add_insecure_port(listen_addr)
        print(f"[{NODE_ID}] Starting gRPC server listening on {listen_addr}")
        await server.start()
        print(f"[{NODE_ID}] Server started successfully.")
        await server.wait_for_termination()
    except RuntimeError as e: 
        print(f"!!! [{NODE_ID}] CRITICAL ERROR: Failed to bind server to {listen_addr}: {e}")
        print(f"!!! Check if the port {MY_PORT} is already in use or if there are network configuration issues.")
    except Exception as e:
        print(f"!!! [{NODE_ID}] An unexpected error occurred during server startup: {e}")
        traceback.print_exc()
    finally:
    
        print(f"[{NODE_ID}] Attempting server shutdown...")
        await server.stop(grace=1) 
        print(f"[{NODE_ID}] Server shutdown process completed.")



async def initiate_inference(input_image_path: str):
    global my_model_part, NEXT_NODE_ADDRESS
    print(f"\n[{NODE_ID}] Initiating inference...")


    transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    try:
        img = Image.open(input_image_path).convert('RGB')
        input_image_torch = transform(img).unsqueeze(0).to(DEVICE)
        print(f"[{NODE_ID}] Loaded input image '{input_image_path}', shape: {input_image_torch.shape}")
    except FileNotFoundError:
        print(f"[{NODE_ID}] Input image '{input_image_path}' not found. Using dummy data.")
        input_image_torch = torch.randn(1, 3, 32, 32).to(DEVICE)
    except Exception as e:
        print(f"[{NODE_ID}] Error loading image: {e}. Using dummy data.")
        input_image_torch = torch.randn(1, 3, 32, 32).to(DEVICE)

  
    print(f"[{NODE_ID}] Running model part 0...")
    with torch.no_grad():
        intermediate_output_torch = my_model_part(input_image_torch)
    intermediate_output_np = intermediate_output_torch.cpu().numpy()
    print(f"[{NODE_ID}] Computed intermediate output shape: {intermediate_output_np.shape}")


    print(f"[{NODE_ID}] Sending intermediate tensor to {NEXT_NODE_ADDRESS}...")
    if not NEXT_NODE_ADDRESS:
        print(f"[{NODE_ID}] ERROR: Cannot initiate inference, NEXT_NODE_ADDRESS is not set.")
        return

    async with grpc.aio.insecure_channel(NEXT_NODE_ADDRESS) as channel:
        stub = node_service_pb2_grpc.NodeServiceStub(channel)
        request = node_service_pb2.TensorRequest(
            request_id="cifar_pipe_001", 
            tensor=node_service_pb2.Tensor(
                tensor_data=intermediate_output_np.tobytes(),
                shape=list(intermediate_output_np.shape),
                dtype=str(intermediate_output_np.dtype)
            )
        )
        try:
            response = await stub.SendTensor(request)
            print(f"[{NODE_ID}] Received final status from pipeline: {response.status}")
         
            if response.HasField("output_tensor"):
                final_tensor_proto = response.output_tensor
                final_result_np = np.frombuffer(
                    final_tensor_proto.tensor_data,
                    dtype=np.dtype(final_tensor_proto.dtype)
                ).reshape(final_tensor_proto.shape)
                predicted_class_index = np.argmax(final_result_np)
             
                print(f"[{NODE_ID}] ***** FINAL PREDICTION (Index): {predicted_class_index} *****")
            else:
                print(f"[{NODE_ID}] Final result was processed on the last node or not returned.")

        except grpc.aio.AioRpcError as e:
            print(f"!!! [{NODE_ID}] SendTensor RPC failed during initiation: {e.code()} - {e.details()}")
        except Exception as e:
            print(f"!!! [{NODE_ID}] Error processing response from pipeline: {e}")
            traceback.print_exc()


async def start_inference_after_delay(delay, image_path):
    print(f"[{NODE_ID}] Waiting {delay}s before initiating inference...")
    await asyncio.sleep(delay)
    print(f"[{NODE_ID}] Delay finished. Initiating inference now.")
    await initiate_inference(image_path) 


if __name__ == '__main__': 
    print("Script started...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_id", required=True, help="Unique ID for this node (e.g., node1)")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file")
    parser.add_argument("--input_image", help="Path to input image (only used by node with part_index 0)")
    args = parser.parse_args()

    NODE_ID = args.node_id
    print(f"Parsed Node ID: {NODE_ID}")

   
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        print(f"ERROR: Config file not found at '{args.config}'")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config file '{args.config}': {e}")
        exit(1)


    my_node_config = next((n for n in config.get('nodes', []) if n.get('id') == NODE_ID), None)
    if not my_node_config:
        print(f"ERROR: Node ID '{NODE_ID}' not found in config file '{args.config}'")
        exit(1)

    MY_ADDRESS = my_node_config.get('address')
    MY_PART_INDEX = my_node_config.get('part_index')
    MODEL_WEIGHTS_PATH = config.get('model_weights')
    num_parts = config.get('num_parts')
    return_node_id = config.get('return_to_node_id')

   
    if None in [MY_ADDRESS, MY_PART_INDEX, MODEL_WEIGHTS_PATH, num_parts]:
        print("ERROR: Config file is missing required fields (address, part_index, model_weights, num_parts)")
        exit(1)

  
    try:
        MY_PORT = int(MY_ADDRESS.split(':')[-1])
    except (ValueError, IndexError):
        print(f"ERROR: Invalid format for MY_ADDRESS '{MY_ADDRESS}'. Expected IP:Port.")
        exit(1)

    IS_LAST_NODE = (MY_PART_INDEX == num_parts - 1)

    if not IS_LAST_NODE:
        next_part_index = MY_PART_INDEX + 1
        next_node_config = next((n for n in config['nodes'] if n.get('part_index') == next_part_index), None)
        if not next_node_config:
             print(f"ERROR: Could not find node config for next part index {next_part_index}")
             exit(1)
        NEXT_NODE_ADDRESS = next_node_config.get('address')
        if not NEXT_NODE_ADDRESS:
             print(f"ERROR: Next node (index {next_part_index}) is missing address in config")
             exit(1)
    else:
        if return_node_id:
            return_node_config = next((n for n in config['nodes'] if n.get('id') == return_node_id), None)
            if return_node_config:
                FIRST_NODE_ADDRESS = return_node_config.get('address')


    print(f"--- Node Configuration ---")
    print(f"  ID: {NODE_ID}")
    print(f"  Full Address (for clients): {MY_ADDRESS}")
    print(f"  Server Listening Port: {MY_PORT}") # Log the port being used
    print(f"  Part Index: {MY_PART_INDEX} / {num_parts - 1}")
    print(f"  Is Last: {IS_LAST_NODE}")
    print(f"  Next Node Address: {NEXT_NODE_ADDRESS}")
    print(f"  Return Node Addr: {FIRST_NODE_ADDRESS}")
    print(f"  Weights: {MODEL_WEIGHTS_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"-------------------------")


    # Load weights and instantiate the correct model part
    try:
        print(f"[{NODE_ID}] Loading full state dict...")
        # Corrected typo mapl_location -> map_location
        full_state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device("cpu"))
        print(f"[{NODE_ID}] Loading model part {MY_PART_INDEX}...")
        temp_full_model = NeuralNetwork() # For initialization reference
        MyModelPartClass = MODEL_PARTS_CLASSES.get(MY_PART_INDEX)
        if not MyModelPartClass:
            print(f"ERROR: No model part class defined for index {MY_PART_INDEX}")
            exit(1)
        my_model_part = MyModelPartClass(temp_full_model).to(DEVICE)
        missing_keys, unexpected_keys = my_model_part.load_state_dict(full_state_dict, strict=False)
        if missing_keys:
            print(f"[{NODE_ID}] WARNING: Missing keys loading state dict: {missing_keys}")
        if unexpected_keys:
            print(f"[{NODE_ID}] WARNING: Unexpected keys loading state dict: {unexpected_keys}")
        my_model_part.eval()
        print(f"[{NODE_ID}] Successfully loaded weights into ModelPart{MY_PART_INDEX}.")
    except FileNotFoundError:
        print(f"[{NODE_ID}] ERROR: Weights file not found at '{MODEL_WEIGHTS_PATH}'")
        exit(1)
    except Exception as e:
        print(f"[{NODE_ID}] ERROR loading model/weights: {e}")
        traceback.print_exc()
        exit(1)

    loop = asyncio.get_event_loop()
    server_task = loop.create_task(serve())
    init_task = None 

    if MY_PART_INDEX == 0:
        if not args.input_image:
             print("WARNING: Node 0 needs --input_image to start inference. Server will run, but no inference initiated.")
             init_task = loop.create_task(asyncio.sleep(3600)) 
        else:
             
             init_task = loop.create_task(start_inference_after_delay(2, args.input_image))
    else:
       
        init_task = loop.create_task(asyncio.sleep(3600)) 

    try:
        print(f"[{NODE_ID}] Running event loop...")
      
        loop.run_until_complete(asyncio.gather(server_task, init_task))
    except KeyboardInterrupt:
        print(f"\n[{NODE_ID}] KeyboardInterrupt received, shutting down...")
    except Exception as e:
        print(f"\n[{NODE_ID}] UNEXPECTED ERROR in main loop: {e}")
        traceback.print_exc()
    finally:
        print(f"[{NODE_ID}] Cleaning up...")

        if server_task and not server_task.done():
            server_task.cancel()
        if init_task and not init_task.done():
            init_task.cancel()

      
        async def gather_cancelled():
            await asyncio.gather(server_task, init_task, return_exceptions=True)

        loop.run_until_complete(gather_cancelled())

        if not loop.is_closed():
             loop.close()
        print(f"[{NODE_ID}] Event loop closed. Exiting.")
