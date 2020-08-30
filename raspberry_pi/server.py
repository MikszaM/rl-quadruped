import socket
import servos

HOST = '192.168.0.220'
PORT = 5005
BUFFER_SIZE = 47


pwm = servos.init_controller()
servos.zero_feet(pwm)
servos.zero_legs(pwm)

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
            servos.set_servos(pwm, action)
            if not data:
                break