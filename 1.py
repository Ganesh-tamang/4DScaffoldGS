import socket

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the local port that is tunneled to the remote server
server_address = ("localhost", 9000)
client_socket.connect(server_address)

# Send data to the server
client_socket.sendall(b"Hello from the local machine!")

# Receive data from the server
response = client_socket.recv(1024)
print(f"Received: {response.decode()}")

# Close the connection
client_socket.close()
