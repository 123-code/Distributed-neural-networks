import grpc 
import asyncio 
import node_service_pb2
import node_service_pb2_grpc
import numpy as np 

class NodeServiceImpl(node_service_pb2_grpc.NodeServiceServicer):
    async def SendMessage(self,request:node_service_pb2.MessageRequest,context):
        print(f"Received message: {request.message_text} from {request.sender_id}")
        reply_text = f"Received data: {request.message_text}"
        return node_service_pb2.MessageReply(confirmation_text=reply_text)
    
    async def HealthCheck(self,request:node_service_pb2.Empty,context):
        print("health check received")
        return node_service_pb2.HealthCheckResponse(is_healthy=True)
    
    async def SendTensor(self, request:node_service_pb2.TensorRequest, context):
        print(f"Received tensor with request_id: {request.request_id}")
        print(f"Tensor shape: {request.shape}, dtype: {request.dtype}")

        try:
            tensor = np.frombuffer(request.tensor_data,dtype=np.dtype(request.dtype)).reshape(request.shape)
            print(f"Successfully deserialized tensor with shape: {tensor.shape}")
        except Exception as e:
            print(f"Could not deserialize tensor: {e}")

        return node_service_pb2.TensorResponse(status="Tensor received successfully")
     



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
