from Lib.actionMessage import recvResponse, sendMessage

def getPosition(sock):
    sendMessage(sock, ";GET_POSITION:")
    response = recvResponse(sock)
    return response

def getSpeed(sock):
    sendMessage(sock, ";GET_SPEED;")
    response = recvResponse(sock)
    return response

def getSteering(sock):
    sendMessage(sock, ";GET_STEERING;")
    response = recvResponse(sock)
    return response

def getInfosRaycast(sock):
    sendMessage(sock, ";GET_INFOS_RAYCAST;")
    response = recvResponse(sock)
    return response

def retrieveData(sock):
    data = []
    data.append(getPosition(sock))
    data.append(getSpeed(sock))
    data.append(getSteering(sock))
    data.append(getInfosRaycast(sock))
    print(data)
    return 0

