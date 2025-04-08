import grpc 
import asyncio 
import node_service_pb2
import node_service_pb2_grpc
import numpy as np

async def run():

    async with grpc.aio.insecure_channel('localhost:50051') as channel:
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