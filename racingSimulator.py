import socket
from Data.retrieveData import retrieveData
from Lib.actionMessage import recvResponse, sendMessage

def createConnection():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("0.0.0.0", 8085))
    return sock

def launchSimulation(sock):
    while(1):
        sendMessage(sock, ";SET_SPEED:1;")
        recvResponse(sock)
        retrieveData(sock)

def racingSimulator():
    sock = createConnection()
    launchSimulation(sock)
