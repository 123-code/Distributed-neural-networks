import grpc
import asyncio
import node_service_pb2
import node_service_pb2_grpc
import numpy as np

async def run():
    print("Starting client...")  
    try:
        async with grpc.aio.insecure_channel('192.168.1.202:50051') as channel:
            print("Channel created...") 
            stub = node_service_pb2_grpc.NodeServiceStub(channel)
            tensor_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            shape = tensor_data.shape
            tensor_data_bytes = tensor_data.tobytes()
            request = node_service_pb2.TensorRequest(
                request_id="tensor001",
                tensor_data=tensor_data_bytes,
                shape=shape,
                dtype="float32"
            )
            print("calling send message")
            message_request = node_service_pb2.MessageRequest(
                sender_id = "client001",
                message_text = "Hello from client",
            )
            try:
        
                reply = await stub.SendTensor(request)
                print(f"Received reply: {reply.status}")
            except grpc.aio.AioRpcError as e:
                print(f"RPC failed: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"An error occurred: {e}") 

if __name__ == '__main__':
    asyncio.run(run())