import pyhid_usb_relay

relay = pyhid_usb_relay.find()

print(relay.state)
print("Toggeling relay")

relay.toggle_state(1)

print(relay.state)