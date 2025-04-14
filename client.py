import grpc
import asyncio
import node_service_pb2
import node_service_pb2_grpc
import numpy as np

W1 = np.random.rand(3,5).astype(np.float32)
b1 = np.random.rand(5).astype(np.float32)

def relu(x):
    return np.maximum(0,x)


async def run():
    print("Starting client...")  
    try:
        async with grpc.aio.insecure_channel('192.168.1.202:50051') as channel:
            print("Channel created...") 
            stub = node_service_pb2_grpc.NodeServiceStub(channel)
            print("\n--- Performing Client Computation ---")
            input_data = np.random.rand(1,3).astype(np.float32)
            print("f input data shape:{input_data.shape}")
            linear1_output = input_data @ W1 + b1
            intermediate_output = relu(linear1_output)
            print(f"Intermediate output shape: {intermediate_output.shape}")
            request = node_service_pb2.TensorRequest(
                request_id = "compute_req_001",
                tensor = node_service_pb2.Tensor(
                    tensor_data = intermediate_output.tobytes(),
                    shape = list(intermediate_output.shape),
                    dtype= str(intermediate_output.dtype)
                )
            )
            try:
                response = await stub.SendTensor(request)
                print("fserver response:{response.status}")
                if response.HasField("output_tensor"):
                    final_tensor_proto = response.output_tensor
                    final_result = np.frombuffer(
                        final_tensor_proto.tensor_data,
                        dtype = np.dtype(final_tensor_proto.dtype)
                    ).reshape(final_tensor_proto.shape)
                else:
                    print("server did not return output tensor")
            except grpc.aio.AioRpcError as e :
                print(f"send tensor failed:{e.code()} - {e.details()}")

    except Exception as e:
        print(f"An error occurred: {e}") 

if __name__ == '__main__':
    asyncio.run(run())