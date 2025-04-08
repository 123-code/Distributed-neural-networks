import grpc
import asyncio
import node_service_pb2
import node_service_pb2_grpc
import numpy as np

async def run():
    print("Starting client...")  # Debugging line
    try:
        async with grpc.aio.insecure_channel('192.168.1.202:50051') as channel:
            print("Channel created...")  # Debugging line
            stub = node_service_pb2_grpc.NodeServiceStub(channel)
            print("calling send message")
            message_request = node_service_pb2.MessageRequest(
                sender_id = "client001",
                message_text = "Hello from client",
            )
            try:
                reply = await stub.SendMessage(message_request)
                print(f"Received reply: {reply.confirmation_text}")
            except grpc.aio.AioRpcError as e:
                print(f"RPC failed: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"An error occurred: {e}")  # Catch-all error

if __name__ == '__main__':
    asyncio.run(run())