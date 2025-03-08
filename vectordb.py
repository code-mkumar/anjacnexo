from pymilvus import MilvusClient

milvus = MilvusClient("your_milvus_server")
milvus.create_collection(
    collection_name="my_collection",
    dimension=1024  # Adjust based on your vector size
)

connections.connect(
    alias="default", 
    host="localhost",  # Replace with your Milvus server host
    port="19530"  # Replace with your Milvus server port if different
)
