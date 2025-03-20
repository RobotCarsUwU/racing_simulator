def sendMessage(sock, msg):
    byteMessage = bytes(msg, "utf-8")
    sock.send(byteMessage)

def recvResponse(sock):
    response = sock.recv(1024)
    response.decode()
    return response
