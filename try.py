import chromadb
client = chromadb.Client()
print(client.heartbeat())

#心跳检测，用来确认客户端是否正常运行，返回一整个时间戳