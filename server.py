import grpc
import asyncio
import node_service_pb2
import node_service_pb2_grpc
import numpy as np

W2 = np.random.rand(5, 2).astype(np.float32)
b2 = np.random.rand(2).astype(np.float32)


class NodeServiceImpl(node_service_pb2_grpc.NodeServiceServicer):
    async def SendMessage(self, request: node_service_pb2.MessageRequest, context):
        print(f"Received message: {request.message_text} from {request.sender_id}")
        reply_text = f"Received data: {request.message_text}"
        return node_service_pb2.MessageReply(confirmation_text=reply_text)

    async def HealthCheck(self, request: node_service_pb2.Empty, context):
        print("health check received")
        return node_service_pb2.HealthCheckResponse(is_healthy=True)

    async def SendTensor(self, request: node_service_pb2.TensorRequest, context):
        print(f"Received tensor with request_id: {request.request_id}")
        print(f"Incoming Tensor metadata - shape: {list(request.tensor.shape)}, dtype: {request.tensor.dtype}")
        output_tensor_proto = None
        status_message = ""

        try:
            intermediate_tensor = np.frombuffer(
                request.tensor.tensor_data,
                dtype=np.dtype(request.tensor.dtype)
            ).reshape(request.tensor.shape)
            print(f"Server received intermediate tensor with shape: {intermediate_tensor.shape}")

            if intermediate_tensor.shape[-1] != W2.shape[0]:
                raise ValueError(f"Input tensor last dim {intermediate_tensor.shape[-1]} != W2 first dim {W2.shape[0]}")
            final_output = intermediate_tensor @ W2 + b2
            print(f"Server computed final output with shape: {final_output.shape}")
            status_message = "Tensor processed successfully. Returning final output."
        except Exception as e:
            print(e)
            status_message = str(e)
        
        output_tensor_proto = node_service_pb2.Tensor(
            tensor_data = final_output.tobytes(),
            shape = list(final_output.shape),
            dtype = str(final_output.dtype)
        )
        return node_service_pb2.TensorResponse(status=status_message, result_tensor=output_tensor_proto)


async def serve():
    server = grpc.aio.server()
    node_service_pb2_grpc.add_NodeServiceServicer_to_server(NodeServiceImpl(), server)
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    print(f"Server listening on {listen_addr}")
    await server.start()
    await server.wait_for_termination()


if __name__ == '__main__':
    asyncio.run(serve())