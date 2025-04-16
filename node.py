# node.py
import argparse
import grpc
import asyncio
import json
import node_service_pb2
import node_service_pb2_grpc
import numpy as np
import torch
# Make sure this import works relative to where you run node.py
from cifar_model_parts import NeuralNetwork, ModelPart0, ModelPart1, ModelPart2
import torchvision.transforms as transforms
from PIL import Image
import traceback

# --- Global Variables ---
NODE_ID = None
MY_ADDRESS = None
MY_PORT = None
MY_PART_INDEX = -1
NEXT_NODE_ADDRESS = None
FIRST_NODE_ADDRESS = None # Address for final result callback (optional)
IS_LAST_NODE = False
MODEL_WEIGHTS_PATH = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
my_model_part = None

MODEL_PARTS_CLASSES = {
    0: ModelPart0,
    1: ModelPart1,
    2: ModelPart2
}

# --- gRPC Server Implementation ---
class NodeServiceImpl(node_service_pb2_grpc.NodeServiceServicer):
    async def SendTensor(self, request: node_service_pb2.TensorRequest, context):
        global my_model_part, NEXT_NODE_ADDRESS, IS_LAST_NODE, FIRST_NODE_ADDRESS

        print(f"\n[{NODE_ID}] Received tensor request_id: {request.request_id}")
        print(f"[{NODE_ID}] Incoming Tensor - shape: {list(request.tensor.shape)}, dtype: {request.tensor.dtype}")

        # --- Initialize response variables ---
        # Tensor to potentially send *back* to the *previous* node in the chain
        tensor_to_return_proto = None
        status_message = f"[{NODE_ID}] Error processing tensor."
        # ------------------------------------

        try:
            # 1. Deserialize input tensor
            input_np = np.frombuffer(
                request.tensor.tensor_data, dtype=np.dtype(request.tensor.dtype)
            ).reshape(request.tensor.shape)
            input_torch = torch.from_numpy(input_np.copy()).to(DEVICE) # Use .copy() to potentially avoid write warning
            print(f"[{NODE_ID}] Deserialized input tensor shape: {input_torch.shape}")

            # 2. Run through this node's model part
            with torch.no_grad():
                output_torch = my_model_part(input_torch)
            print(f"[{NODE_ID}] Computed output tensor shape: {output_torch.shape}")
            output_np = output_torch.cpu().numpy()

            # 3. Decide where to send the output
            if IS_LAST_NODE:
                # --- This is the LAST node ---
                print(f"[{NODE_ID}] Reached final node.")
                predicted_class_index = np.argmax(output_np)
                print(f"[{NODE_ID}] Final Prediction Index: {predicted_class_index}")
                status_message = f"[{NODE_ID}] Processing complete. Prediction: {predicted_class_index}"

                # Prepare the *final* tensor to send BACK to the PREVIOUS node (who called this one)
                tensor_to_return_proto = node_service_pb2.Tensor(
                    tensor_data=output_np.tobytes(),
                    shape=list(output_np.shape),
                    dtype=str(output_np.dtype)
                )
                # NOTE: No further forwarding needed from the last node.

            else:
                # --- This is an INTERMEDIATE node ---
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
                        # Get the response from the NEXT node
                        next_response = await stub.SendTensor(next_request)
                        print(f"[{NODE_ID}] Response from next node ({NEXT_NODE_ADDRESS}): {next_response.status}")
                        status_message = f"[{NODE_ID}] Forwarded. Next node status: {next_response.status}"

                        # *** FIX: Only propagate the final tensor back if it exists in the response ***
                        # This handles the case where the *next* node was the *last* one
                        if next_response.HasField("output_tensor"):
                            tensor_to_return_proto = next_response.output_tensor # Propagate the FINAL result back
                        # *** END FIX ***

                    except grpc.aio.AioRpcError as e:
                         print(f"!!! [{NODE_ID}] Error calling SendTensor on next node ({NEXT_NODE_ADDRESS}): {e.code()} - {e.details()}")
                         status_message = f"[{NODE_ID}] Error forwarding: {e.details()}"
                         # Don't propagate tensor on error
                         tensor_to_return_proto = None


        except Exception as e:
            print(f"!!! [{NODE_ID}] Error processing tensor: {e}")
            traceback.print_exc()
            status_message = f"[{NODE_ID}] Error: {e}"
            tensor_to_return_proto = None # Ensure no tensor is returned on error

        # Return status AND the relevant tensor (either final result from self/next, or None)
        # back to the node that CALLED this one.
        return node_service_pb2.TensorResponse(
            status=status_message,
            output_tensor=tensor_to_return_proto # Send back the final tensor if available
        )

    async def HealthCheck(self, request, context):
        print(f"[{NODE_ID}] Health check requested")
        return node_service_pb2.HealthCheckResponse(is_healthy=True)

    async def SendMessage(self, request, context):
        print(f"[{NODE_ID}] Received message from {request.sender_id}")
        return node_service_pb2.MessageReply(confirmation_text=f"[{NODE_ID}] got msg '{request.message_text}'")


# --- Server Startup ---
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
        print(f"!!! Check if port {MY_PORT} is in use or network issues.")
    except Exception as e:
        print(f"!!! [{NODE_ID}] An unexpected error occurred during server startup: {e}")
        traceback.print_exc()
    finally:
        print(f"[{NODE_ID}] Attempting server shutdown...")
        await server.stop(grace=1)
        print(f"[{NODE_ID}] Server shutdown complete.")


# --- Client-side Initiation (only for Node 0) ---
async def initiate_inference(input_image_path: str):
    global my_model_part, NEXT_NODE_ADDRESS
    print(f"\n[{NODE_ID}] Initiating inference...")

    # 1. Load and preprocess image
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


    # 2. Run through first model part
    print(f"[{NODE_ID}] Running model part 0...")
    with torch.no_grad():
        intermediate_output_torch = my_model_part(input_image_torch)
    intermediate_output_np = intermediate_output_torch.cpu().numpy()
    print(f"[{NODE_ID}] Computed intermediate output shape: {intermediate_output_np.shape}")

    # 3. Send to the next node
    print(f"[{NODE_ID}] Sending intermediate tensor to {NEXT_NODE_ADDRESS}...")
    if not NEXT_NODE_ADDRESS:
        print(f"[{NODE_ID}] ERROR: Cannot initiate inference, NEXT_NODE_ADDRESS is not set.")
        return

    async with grpc.aio.insecure_channel(NEXT_NODE_ADDRESS) as channel:
        stub = node_service_pb2_grpc.NodeServiceStub(channel)
        request = node_service_pb2.TensorRequest(
            request_id="cifar_pipe_001", # Generate unique IDs in real use
            tensor=node_service_pb2.Tensor(
                tensor_data=intermediate_output_np.tobytes(),
                shape=list(intermediate_output_np.shape),
                dtype=str(intermediate_output_np.dtype)
            )
        )
        try:
            # This call now waits for the *entire* pipeline to finish
            # and potentially gets the final result back.
            response = await stub.SendTensor(request)
            print(f"[{NODE_ID}] Received final status from pipeline: {response.status}")

            # Process final result if it was propagated back by the chain
            if response.HasField("output_tensor"):
                final_tensor_proto = response.output_tensor
                final_result_np = np.frombuffer(
                    final_tensor_proto.tensor_data,
                    dtype=np.dtype(final_tensor_proto.dtype)
                ).reshape(final_tensor_proto.shape)
                predicted_class_index = np.argmax(final_result_np)
                # You might need CIFAR class names list here
                print(f"[{NODE_ID}] ***** FINAL PREDICTION (Index): {predicted_class_index} *****")
            else:
                # This might happen if the last node didn't return the tensor
                # or an intermediate node failed to propagate it.
                print(f"[{NODE_ID}] Final result was processed but not received back by initiating node.")

        except grpc.aio.AioRpcError as e:
            print(f"!!! [{NODE_ID}] SendTensor RPC failed during initiation: {e.code()} - {e.details()}")
        except Exception as e:
            print(f"!!! [{NODE_ID}] Error processing response from pipeline: {e}")
            traceback.print_exc()

# Helper async function to start inference after delay
async def start_inference_after_delay(delay, image_path):
    print(f"[{NODE_ID}] Waiting {delay}s before initiating inference...")
    await asyncio.sleep(delay)
    print(f"[{NODE_ID}] Delay finished. Initiating inference now.")
    await initiate_inference(image_path) # Make sure to await the async function

# --- Main Execution ---
if __name__ == '__main__':
    print("Script started...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_id", required=True, help="Unique ID for this node (e.g., node1)")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file")
    parser.add_argument("--input_image", help="Path to input image (only used by node with part_index 0)")
    args = parser.parse_args()

    NODE_ID = args.node_id
    print(f"Parsed Node ID: {NODE_ID}")

    # Load configuration
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

    # Find my details and next node's address
    my_node_config = next((n for n in config.get('nodes', []) if n.get('id') == NODE_ID), None)
    if not my_node_config:
        print(f"ERROR: Node ID '{NODE_ID}' not found in config file '{args.config}'")
        exit(1)

    MY_ADDRESS = my_node_config.get('address')
    MY_PART_INDEX = my_node_config.get('part_index')
    MODEL_WEIGHTS_PATH = config.get('model_weights')
    num_parts = config.get('num_parts')
    return_node_id = config.get('return_to_node_id')

    # Basic validation
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
    else: # If this IS the last node
        NEXT_NODE_ADDRESS = None # No next node to forward to
        if return_node_id: # Check if we need to know the originator's address
            return_node_config = next((n for n in config['nodes'] if n.get('id') == return_node_id), None)
            if return_node_config:
                FIRST_NODE_ADDRESS = return_node_config.get('address')


    print(f"--- Node Configuration ---")
    print(f"  ID: {NODE_ID}")
    print(f"  Full Address (for clients): {MY_ADDRESS}")
    print(f"  Server Listening Port: {MY_PORT}")
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
        full_state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device("cpu"))
        print(f"[{NODE_ID}] Loading model part {MY_PART_INDEX}...")
        temp_full_model = NeuralNetwork()
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

    # Start server and potentially initiate inference
    loop = asyncio.get_event_loop()
    server_task = loop.create_task(serve())
    init_task = None

    if MY_PART_INDEX == 0:
        if not args.input_image:
             print("WARNING: Node 0 needs --input_image to start inference. Server will run, but no inference initiated.")
             init_task = loop.create_task(asyncio.sleep(3600)) # Sleep indefinitely
        else:
             init_task = loop.create_task(start_inference_after_delay(2, args.input_image))
    else:
        # Non-initiating nodes just run the server indefinitely
        init_task = loop.create_task(asyncio.sleep(3600)) # Sleep indefinitely

    try:
        print(f"[{NODE_ID}] Running event loop...")
        # Run both tasks concurrently
        loop.run_until_complete(asyncio.gather(server_task, init_task))
    except KeyboardInterrupt:
        print(f"\n[{NODE_ID}] KeyboardInterrupt received, shutting down...")
    except Exception as e:
        print(f"\n[{NODE_ID}] UNEXPECTED ERROR in main loop: {e}")
        traceback.print_exc()
    finally:
        print(f"[{NODE_ID}] Cleaning up...")
        # Cancel tasks gracefully
        if server_task and not server_task.done():
            server_task.cancel()
        if init_task and not init_task.done():
            init_task.cancel()

        # Allow tasks to finish cancellation
        async def gather_cancelled():
           # Use return_exceptions=True to prevent gather from raising CancelledError itself
           await asyncio.gather(server_task, init_task, return_exceptions=True)

        try:
            # Run the gather operation within the existing loop if it's still running
            if loop.is_running():
                loop.run_until_complete(gather_cancelled())
            else:
                # If the main loop stopped unexpectedly, we might need a temporary loop
                asyncio.run(gather_cancelled())
        except RuntimeError as e:
             print(f"[{NODE_ID}] Error during cleanup gather: {e}") # Handle loop already closed error

        if not loop.is_closed():
             loop.close()
        print(f"[{NODE_ID}] Event loop closed. Exiting.")
