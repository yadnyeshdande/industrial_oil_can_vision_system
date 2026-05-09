import socket
import sys

def scan_ports(ip, ports):
    """Scan multiple ports to find open ones"""
    print(f"Scanning {ip} for open ports...\n")
    open_ports = {}

    for port in ports:
        try:
            sock = socket.create_connection((ip, port), timeout=1)
            sock.close()
            open_ports[port] = True
            print(f"✓ Port {port:5d} - OPEN")
        except (socket.timeout, ConnectionRefusedError, OSError):
            print(f"✗ Port {port:5d} - closed/timeout")

    return open_ports

# Common ports to check
ports = [
    80,      # HTTP
    443,     # HTTPS
    554,     # RTSP
    8000,    # HTTP alt
    8080,    # HTTP alt
    8554,    # RTSP alt
    8888,    # HTTP alt
    9000,    # Various services
    3389,    # RDP
]

if len(sys.argv) > 1:
    ip = sys.argv[1]
else:
    ip = "192.168.1.249"

print("="*60)
print("CP Plus Camera - Port Scanner")
print("="*60)
print(f"Target: {ip}\n")

open_ports = scan_ports(ip, ports)

print("\n" + "="*60)
print("Results:")
print("="*60)

if not open_ports:
    print("No open ports found!")
    sys.exit(1)

print(f"\nOpen ports: {', '.join(map(str, open_ports.keys()))}")

# Suggest next steps based on open ports
if 80 in open_ports:
    print(f"\n[+] HTTP port 80 is open!")
    print(f"    Try: http://{ip} in your browser")
    print(f"    This should show the camera web interface")

if 443 in open_ports:
    print(f"\n[+] HTTPS port 443 is open!")
    print(f"    Try: https://{ip} in your browser")

if 8000 in open_ports:
    print(f"\n[+] HTTP alternative (8000) is open!")
    print(f"    Try: http://{ip}:8000 in your browser")

if 8080 in open_ports:
    print(f"\n[+] HTTP alternative (8080) is open!")
    print(f"    Try: http://{ip}:8080 in your browser")

if 554 not in open_ports:
    print(f"\n[!] Port 554 (RTSP) is NOT open")
    print(f"    RTSP may use a different port. Check:")
    if 8554 in open_ports:
        print(f"    - Port 8554 (found OPEN)")
    else:
        print(f"    - Port 8554")
    print(f"    - Camera settings for RTSP port")

print("\n" + "="*60)
print("Next steps:")
print("="*60)
if 80 in open_ports or 8000 in open_ports or 8080 in open_ports:
    print("1. Access camera web interface to find RTSP URL")
    print("2. Look for Stream/RTSP settings")
    print("3. Check RTSP port number")
    print("4. Find correct stream path")
