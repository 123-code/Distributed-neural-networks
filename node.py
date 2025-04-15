import grpc 
import asyncio
import argsparse
import json 
import node_service_pb2
import node_service_pb2_grpc
import numpy as np 
import torch 
from cifar_model_parts import NeuralNetwork,ModelPart0, ModelPart1, ModelPart2


NODE_ID = None
MY_ADDRESS = None
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
    async def SendTensor(self,request:node_service_pb2.TensorRequest,context):
        global my_model_part,NEXT_NODE_ADDRESS,IS_LAST_NODE,FIRST_NODE_ADDRESS

        print(f"\n[{NODE_ID}] Received tensor request_id: {request.request_id}")
        print(f"[{NODE_ID}] Incoming Tensor - shape: {list(request.tensor.shape)}, dtype: {request.tensor.dtype}")

        final_output_tensor_proto = None
        status_message = f"[{NODE_ID}] Error processing tensor."
        try:
            #deserialize 
            input_np = np.frombuffer(
                request.tensor.tensor_data,dtype=np.dtype(request.tensor.dtype)).reshape(request.tensor.shape)
            
            input_torch = torch.from_numpy(input_np).to(DEVICE)
            print(f"[{NODE_ID}] Received tensor with shape: {input_torch.shape}")

            with torch.no_grad():
                output_torch = my_model_part(input_torch)
            print(f"[{NODE_ID}] Computed output tensor shape: {output_torch.shape}")
            output_np = output_torch.cpu().numpy()


            if IS_LAST_NODE:
                print("reached final node")
                predicted_class_index = np.argmax(output_np)
                print(f"[{NODE_ID}] Final Prediction Index: {predicted_class_index}")
                status_message = f"[{NODE_ID}] Processing complete. Prediction: {predicted_class_index}"
                final_output_tensor = node_service_pb2.Tensor(
                tensor_data = output_np.tobytes(),
                shape = list(output_np.shape),
                dtype = str(output_np.dtype)
                )    
            else:
                print(f"[{NODE_ID}] Forwarding tensor to next node: {NEXT_NODE_ADDRESS}")
                status_message = f"[{NODE_ID}] Forwarded tensor successfully."

                async with grpc.aio.insecure_channel(NEXT_NODE_ADDRESS) as channel:
                    stub = node_service_pb2_grpc.NodeServiceStub(channel)
                    next_request = node_service_pb2.TensorRequest(
                        request_id = request.request_id,
                        tensor = node_service_pb2.Tensor(
                            tensor_data = output_np.tobytes(),
                            shape = list(output_np.shape),
                            dtype = str(output_np.dtype)
                        )
                    )
                    next_response = await stub.SendTensor(next_request)
                    print(f"[{NODE_ID}] Response from next node ({NEXT_NODE_ADDRESS}): {next_response.status}")

                    if next_response.HasField("output_tensor"):
                        final_output_tensor_proto = next_response.output_tensor
        
        except Exception as e:
            print(f"!!! [{NODE_ID}] Error processing tensor: {e}")
            status_message = f"[{NODE_ID}] Error: {e}"
        return node_service_pb2.TensorResponse(
            status_message = status_message,
            output_tensor = final_output_tensor.proto
        )
    
    async def HealthCheck(self,request,context):
        return node_service_pb2.HealthCheckResponse(
            is_healthy = True
        )
    async def SendMessage(self,request,context):
        return node_service_pb2.MessageReply(confirmation_text=f"[{NODE_ID}] got msg")
    


async def serve():
        server = grpc.aio.server()
        node_service_pb2_grpc.add_NodeServiceServicer_to_server(NodeServiceImpl(),server)
        server.add_insecure_port(MY_ADDRESS)
        print(f"[{NODE_ID}] Server listening on {MY_ADDRESS}")
        await server.start()
        await server.wait_for_termination()


async def initiate_inference(input_image_path:str):
    global my_model_part,NEXT_NODE_ADDRESS
    print("starting inference")

    transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    try:
        input_image_torch = torch.randn(1,3,32,32).to(DEVICE)   
    except Exception as e:
        print(f"Error creating input tensor: {e}")
        return
    
    with torch.no_grad():
        intermediate_output_torch = my_model_part(input_image_torch)
    intermediate_output_np = intermediate_output_torch.cpu().numpy()
    print(f"[{NODE_ID}] Computed intermediate output shape: {intermediate_output_np.shape}")
    print(f"[{NODE_ID}] Sending intermediate tensor to {NEXT_NODE_ADDRESS}")

    async with grpc.aio.insecure_channel(NEXT_NODE_ADDRESS) as channel:
        stub = node_service_pb2_grpc.NodeServiceStub(channel)
        request = node_service_pb2.TensorRequest(
            request_id = "cifar_pipe_001",
            tensor = node_service_pb2.Tensor(
                tensor_data = intermediate_output_np.tobytes(),
                shape = list(intermediate_output_np.shape),
                dtype = str(intermediate_output_np.dtype)   
            )
        )
        try:
            response = await stub.SendTensor(request)
            print(f"[{NODE_ID}] Received final status from pipeline: {response.status}")
            if response.HasField("output_tensor"):
                final_tensor_proto = response.output_tensor
                final_result_np = np.from_buffer(
                    final_tensor_proto.tensor_data,
                    dtype = np.dtype(final_tensor_proto.dtype)          

                ).reshape(final_tensor_proto.shape)
                predicted_class_index = np.argmax(final_result_np)
                print(f"[{NODE_ID}] Final Prediction Index: {predicted_class_index}")
        except grpc.aio.AioRpcError as e:
            print(f"[{NODE_ID}] Error sending tensor: {e.code()} - {e.details()}")

if __name__ == 'main':
    parser.argparse.ArgumentParser()
    parser.add_argument("--node_id",required=True,help="unique id for this node")
    parser.add_argument("--config",required=True,help = "path to the json config file")
    parser.add_argument("--input_image",help="path to the input image")
    args = parser.parse_args()

    NODE_ID = args.node_id

    with open(args.config,'r') as f:
        config = json.load(f)
    my_node_config = next((n for n in config['nodes'] if n['id'] == NODE_ID),None)
    if not my_node_config:
        print(f"ERROR: Node ID '{NODE_ID}' not found in config file '{args.config}'")
        exit(1)
    


    MY_ADDRESS = my_node_config['address']
    MY_PART_INDEX = my_node_config['part_index']
    MODEL_WEIGHTS_PATH = config['model_weights']
    num_parts = config['num_parts']
    IS_LAST_NODE = (MY_PART_INDEX == num_parts - 1)


    if not IS_LAST_NODE:
        next_part_index = MY_PART_INDEX + 1
        next_node_config = next((n for n in config['nodes'] if n['part_index'] == next_part_index),None)
        if not next_node_config:
             print(f"ERROR: Could not find node config for next part index {next_part_index}")
             exit(1)
        NEXT_NODE_ADDRESS = next_node_config['address']
    else:
        return_node_config = next((n for n in config['nodes'] if n['id'] == config['return_to_node_id']),None)
        if return_node_config:
            FIRST_NODE_ADDRESS = return_node_config['address']


    print(f"--- Node Configuration ---")
    print(f"  ID: {NODE_ID}")
    print(f"  Address: {MY_ADDRESS}")
    print(f"  Part Index: {MY_PART_INDEX} / {num_parts - 1}")
    print(f"  Is Last: {IS_LAST_NODE}")
    print(f"  Next Node: {NEXT_NODE_ADDRESS}")
    print(f"  Return Node: {FIRST_NODE_ADDRESS}")
    print(f"  Weights: {MODEL_WEIGHTS_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"-------------------------")

    try:
        full_state_dict = torch.load(MODEL_WEIGHTS_PATH,mapl_location=torch.device("cpu"))
        temp_full_model = NeuralNetwork()
        MyModelPartClass = MODEL_PARTS_CLASSES.get(MY_PART_INDEX)
        if not MyModelPartClass:
            print(f"ERROR: No model part class defined for index {MY_PART_INDEX}")
            exit(1)
        my_model_part = MyModelPartClass(temp_full_model).to(DEVICE)
        my_model_part.load_state_dict(full_state_dict,strict=False)
        my_model_part.eval()
        print("succesfully loaded model weights")
    except Exception as e:

        print(f"Error loading model weights: {e}")
        exit(1)
    loop = asyncio.get_event_loop()
    server_task = loop.create_task(serve()  )
    if MY_PART_INDEX == 0:
        if not args.input_image:
             print("WARNING: Node 0 needs --input_image to start inference. Server will run, but no inference initiated.")
             # Start a task that just waits, allowing server to run
             init_task = loop.create_task(asyncio.sleep(3600)) # Sleep for an hour
        else:
            # Give server a moment to start before initiating
            init_task = loop.create_task(asyncio.sleep(2).continue_with(initiate_inference(args.input_image)))
    else:
        init_task = asyncio.sleep(0)
    try:
        loop.run_until_complete(asyncio.gather(server_task,init_task))
    except KeyboardInterrupt:
        print("apagando...")
    finally:
        server_task.cancel()
        init_task.cancel()
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.close()
        print("server stopped")
