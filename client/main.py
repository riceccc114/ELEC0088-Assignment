#Simple UDP socket client
import socket
import sys
from datetime import datetime
greeting_flag=0
# HOST = ''
# PORT = 8888
# Datagram (udp) socket
try:
    #create a SOCKET
    s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg :
    print('Failed to create socket')
    sys.exit()
host= 'localhost'
port=8888
while(1):
    try:
        if greeting_flag == 0:
            greeting_flag = 1
            msg_connected = "Connection successfully established!".encode()
            s.sendto(msg_connected, (host, port))
            d = s.recvfrom(1024)
            reply = d[0].decode()
            addr = d[1]
            # print the message received from server
            print('Oracle :' + reply)
        else:
            msg = input('Please enter message to send: (For example: Can you give me the maximum temperature for tomorrow?)\n ').encode()
            # Set the whole string
            s.sendto(msg, (host, port))
            # receive data from client (data, addr)
            d= s.recvfrom(1024)
            reply = d[0].decode()
            addr = d[1]
            #print the message received from server
            print('Oracle :' + reply)
    #print error message
    except socket.error as msg:
        print('Error Code:' + str(msg[0]) + 'Message' + msg[1])
        sys.exit()
