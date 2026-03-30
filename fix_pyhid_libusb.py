#!/usr/bin/env python3
"""
Fix pyhid_usb_relay libusb DLL location issue
Run this after installing pyhid_usb_relay and libusb packages
"""
import pathlib
import shutil
import sys

def fix_libusb_dll():
    """Copy libusb-1.0.dll to the location pyhid_usb_relay expects."""
    try:
        # Locate the libusb package
        libusb_base = pathlib.Path('.')
        
        # Search for libusb in site-packages
        import libusb
        libusb_base = pathlib.Path(libusb.__file__).parent
        
        # Source DLL location (new libusb package structure)
        windows_x64_dll = libusb_base / '_platform/windows/x86_64/libusb-1.0.dll'
        
        if not windows_x64_dll.exists():
            print(f"❌ Error: Source DLL not found at {windows_x64_dll}")
            print("Make sure 'libusb' package is installed: pip install libusb")
            return False
        
        # Create destination directory
        x64_dir = libusb_base / 'x64'
        x64_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy DLL
        dest_dll = x64_dir / 'libusb-1.0.dll'
        shutil.copy2(windows_x64_dll, dest_dll)
        
        print(f"✅ Success! Copied libusb-1.0.dll to {dest_dll}")
        
        # Verify it works
        import pyhid_usb_relay
        relay = pyhid_usb_relay.find()
        print(f"✅ pyhid_usb_relay working! Relay state: {relay.state}")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return False

if __name__ == '__main__':
    success = fix_libusb_dll()
    sys.exit(0 if success else 1)