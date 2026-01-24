import os

import serial
import serial.tools.list_ports


def check_ports():
    print("Listing available serial ports:")
    ports = serial.tools.list_ports.comports()
    for p in ports:
        print(f"  {p.device} - {p.description} - {p.hwid}")

    target_ports = ["/dev/ttyUSB0", "/dev/ttyUSB1"]

    print("\nChecking target ports availability:")
    opened_ports = []

    for port in target_ports:
        if not os.path.exists(port):
            print(f"  {port}: NOT FOUND")
            continue

        try:
            s = serial.Serial(port, 1000000, timeout=0.1)
            print(f"  {port}: OPENED successfully")
            opened_ports.append(s)
        except serial.SerialException as e:
            print(f"  {port}: FAILED to open. Error: {e}")

    if len(opened_ports) == 2:
        print("\nBoth ports opened successfully simultaneously!")
        print("Closing ports...")
        for s in opened_ports:
            s.close()
    else:
        print(f"\nOnly {len(opened_ports)} ports opened.")
        if len(opened_ports) > 0:
            for s in opened_ports:
                s.close()


if __name__ == "__main__":
    check_ports()
