"""
===========================================================
WAVESHARE 16-CHANNEL MODBUS TCP ETHERNET RELAY
===========================================================

Device:
    Waveshare Modbus POE ETH Relay (16 Channel)

Communication:
    Modbus TCP over Ethernet

Relay IP:
    192.168.1.200

Port:
    502 (Standard Modbus TCP Port)

Modbus Device ID / Slave ID:
    1

Library Used:
    pymodbus

Install:
    pip install pymodbus

===========================================================
WHAT THIS CODE TEACHES
===========================================================

1. How to connect to the relay board
2. How to turn ON a relay
3. How to turn OFF a relay
4. How to read relay state/status
5. How to toggle a relay
6. How to turn ALL relays ON
7. How to turn ALL relays OFF
8. How to read all relay states
9. How Modbus coil addresses work
10. Proper cleanup and disconnect

===========================================================
MODBUS ADDRESSING
===========================================================

Relay Channel    Coil Address
-------------    ------------
Relay 1          0
Relay 2          1
Relay 3          2
...
Relay 16         15

IMPORTANT:
Modbus addresses start from 0 internally.

So:
    Relay 1 -> address 0
    Relay 16 -> address 15

===========================================================
"""

from pymodbus.client import ModbusTcpClient
import time


class WaveshareRelay:
    """
    Class to control Waveshare Modbus TCP Relay Board
    """

    # -------------------------------------------------------
    # CONSTRUCTOR
    # -------------------------------------------------------
    def __init__(
        self,
        ip="192.168.1.200",
        port=502,
        device_id=1
    ):
        """
        Connect to relay board

        Parameters:
            ip         -> Relay board IP address
            port       -> Modbus TCP port (usually 502)
            device_id  -> Modbus slave/device ID
        """

        self.ip = ip
        self.port = port
        self.device_id = device_id

        # Create Modbus TCP client
        self.client = ModbusTcpClient(
            host=self.ip,
            port=self.port
        )

        # Attempt connection
        connected = self.client.connect()

        if not connected:
            raise Exception(
                f"Could not connect to relay board at {ip}:{port}"
            )

        print(f"[CONNECTED] Relay board connected at {ip}:{port}")

    # -------------------------------------------------------
    # INTERNAL HELPER
    # -------------------------------------------------------
    def _validate_channel(self, channel):
        """
        Ensure channel is between 1 and 16
        """

        if not 1 <= channel <= 16:
            raise ValueError(
                "Channel must be between 1 and 16"
            )

    # -------------------------------------------------------
    # TURN RELAY ON
    # -------------------------------------------------------
    def relay_on(self, channel):
        """
        Turn ON a relay

        Example:
            relay.relay_on(1)
        """

        self._validate_channel(channel)

        # Modbus coil address
        address = channel - 1

        # Write TRUE to coil
        result = self.client.write_coil(
            address=address,
            value=True,
            device_id=self.device_id
        )

        if result.isError():
            print(f"[ERROR] Failed to turn ON relay {channel}")
            return False

        print(f"[ON] Relay {channel} turned ON")
        return True

    # -------------------------------------------------------
    # TURN RELAY OFF
    # -------------------------------------------------------
    def relay_off(self, channel):
        """
        Turn OFF a relay

        Example:
            relay.relay_off(1)
        """

        self._validate_channel(channel)

        address = channel - 1

        # Write FALSE to coil
        result = self.client.write_coil(
            address=address,
            value=False,
            device_id=self.device_id
        )

        if result.isError():
            print(f"[ERROR] Failed to turn OFF relay {channel}")
            return False

        print(f"[OFF] Relay {channel} turned OFF")
        return True

    # -------------------------------------------------------
    # READ RELAY STATUS
    # -------------------------------------------------------
    def relay_status(self, channel):
        """
        Read current relay state

        Returns:
            True  -> Relay ON
            False -> Relay OFF
            None  -> Read failed

        Example:
            state = relay.relay_status(1)
        """

        self._validate_channel(channel)

        address = channel - 1

        # Read 1 coil
        result = self.client.read_coils(
            address=address,
            count=1,
            device_id=self.device_id
        )

        if result.isError():
            print(f"[ERROR] Could not read relay {channel}")
            return None

        state = result.bits[0]

        if state:
            print(f"[STATUS] Relay {channel} is ON")
        else:
            print(f"[STATUS] Relay {channel} is OFF")

        return state

    # -------------------------------------------------------
    # TOGGLE RELAY
    # -------------------------------------------------------
    def toggle(self, channel):
        """
        Toggle relay state

        ON  -> OFF
        OFF -> ON
        """

        current_state = self.relay_status(channel)

        if current_state is None:
            return False

        if current_state:
            return self.relay_off(channel)
        else:
            return self.relay_on(channel)

    # -------------------------------------------------------
    # TURN ALL RELAYS ON
    # -------------------------------------------------------
    def all_on(self):
        """
        Turn ON all 16 relays
        """

        values = [True] * 16

        result = self.client.write_coils(
            address=0,
            values=values,
            device_id=self.device_id
        )

        if result.isError():
            print("[ERROR] Failed to turn ON all relays")
            return False

        print("[ON] All relays turned ON")
        return True

    # -------------------------------------------------------
    # TURN ALL RELAYS OFF
    # -------------------------------------------------------
    def all_off(self):
        """
        Turn OFF all 16 relays
        """

        values = [False] * 16

        result = self.client.write_coils(
            address=0,
            values=values,
            device_id=self.device_id
        )

        if result.isError():
            print("[ERROR] Failed to turn OFF all relays")
            return False

        print("[OFF] All relays turned OFF")
        return True

    # -------------------------------------------------------
    # READ ALL RELAY STATES
    # -------------------------------------------------------
    def read_all_states(self):
        """
        Read status of all 16 relays

        Returns:
            List of True/False values
        """

        result = self.client.read_coils(
            address=0,
            count=16,
            device_id=self.device_id
        )

        if result.isError():
            print("[ERROR] Failed to read relay states")
            return None

        states = result.bits[:16]

        print("\n========== RELAY STATES ==========")

        for i, state in enumerate(states, start=1):

            state_text = "ON" if state else "OFF"

            print(f"Relay {i:02d} : {state_text}")

        print("==================================\n")

        return states

    # -------------------------------------------------------
    # PULSE RELAY
    # -------------------------------------------------------
    def pulse(
        self,
        channel,
        duration=1.0
    ):
        """
        Turn relay ON for a specific duration,
        then automatically turn it OFF.

        Useful for:
            - Door locks
            - Push button simulation
            - Trigger pulses

        Example:
            relay.pulse(1, duration=0.5)
        """

        print(f"[PULSE] Relay {channel} for {duration} sec")

        self.relay_on(channel)

        time.sleep(duration)

        self.relay_off(channel)

    # -------------------------------------------------------
    # DISCONNECT
    # -------------------------------------------------------
    def close(self):
        """
        Close Modbus connection
        """

        self.client.close()

        print("[DISCONNECTED] Connection closed")


# ===========================================================
# EXAMPLE USAGE
# ===========================================================

if __name__ == "__main__":

    # -------------------------------------------------------
    # CONNECT TO RELAY BOARD
    # -------------------------------------------------------

    relay = WaveshareRelay(
        ip="192.168.1.200",
        port=502,
        device_id=1
    )

    # -------------------------------------------------------
    # TURN ON RELAY 1
    # -------------------------------------------------------

    relay.relay_on(1)

    time.sleep(1)

    # -------------------------------------------------------
    # READ STATUS OF RELAY 1
    # -------------------------------------------------------

    state = relay.relay_status(1)

    print("Relay 1 state =", state)

    # -------------------------------------------------------
    # TURN OFF RELAY 1
    # -------------------------------------------------------

    relay.relay_off(1)

    time.sleep(1)

    # -------------------------------------------------------
    # TOGGLE RELAY 2
    # -------------------------------------------------------

    relay.toggle(2)

    time.sleep(1)

    relay.toggle(2)

    # -------------------------------------------------------
    # PULSE RELAY 3 FOR 2 SECONDS
    # -------------------------------------------------------

    relay.pulse(3, duration=2)

    # -------------------------------------------------------
    # TURN ON MULTIPLE RELAYS
    # -------------------------------------------------------

    relay.relay_on(4)
    relay.relay_on(5)
    relay.relay_on(6)

    # -------------------------------------------------------
    # READ ALL STATES
    # -------------------------------------------------------

    relay.read_all_states()

    time.sleep(2)

    # -------------------------------------------------------
    # TURN ALL RELAYS OFF
    # -------------------------------------------------------

    relay.all_off()

    # -------------------------------------------------------
    # DISCONNECT
    # -------------------------------------------------------

    relay.close()

"""
===========================================================
EXPECTED OUTPUT
===========================================================

[CONNECTED] Relay board connected at 192.168.1.200:502

[ON] Relay 1 turned ON
[STATUS] Relay 1 is ON
Relay 1 state = True

[OFF] Relay 1 turned OFF

[PULSE] Relay 3 for 2 sec

========== RELAY STATES ==========
Relay 01 : OFF
Relay 02 : OFF
Relay 03 : OFF
Relay 04 : ON
Relay 05 : ON
Relay 06 : ON
...
==================================

[OFF] All relays turned OFF

[DISCONNECTED] Connection closed

===========================================================
"""