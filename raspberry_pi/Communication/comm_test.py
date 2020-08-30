import socket


HOST = '192.168.0.220'
PORT = 5005
BUFFER_SIZE = 47



with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(BUFFER_SIZE)
            print(data)
            action = (data.decode("utf-8")).split(',')
            for a in action:
                print(float(a))
            if not data:
                break
