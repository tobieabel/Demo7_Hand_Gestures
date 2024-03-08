import subprocess
import time
import socket

def turn_off_display():
    subprocess.run(["xset", "-display", ":0.0", "dpms", "force", "off"])
    #subprocess.run("vcgencmd display_power 0", shell=True)
    
def turn_on_display():
    subprocess.run(["xset", "-display", ":0.0", "dpms", "force", "on"])
    #subprocess.run(["vcgencmd", "display_power", "1"])

def play_video(video_path):
    subprocess.run(["cvlc", "--fullscreen", "--no-osd", video_path])
    
server_ip = "0.0.0.0"
server_port=12345
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))

turn_off_display()
time.sleep(1)

while True:
    
    server_socket.listen(1)
    print(f"Server listening on port{server_port}")
    client_socket, client_address = server_socket.accept()
    print(f"Connection made from {client_address}")
    turn_on_display()
    instruction_bytes = client_socket.recv(1024)
    instruction = instruction_bytes.decode('utf-8')
    print(instruction)
    
    play_video("York ATS Promo.mp4")
    
    play_video("Scary_Nun.mp4")
    
    turn_off_display()
    

client_socket.close()
server_socket.close()
