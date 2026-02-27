import pyautogui
import keyboard
import threading
import time

clicking = False

def auto_click():
    global clicking
    while True:
        if clicking:
            pyautogui.click()
            time.sleep(0.01)  # speed (0.01 = ~100 CPS)

def start_clicking():
    global clicking
    clicking = True
    print("Clicking started")

def stop_clicking():
    global clicking
    clicking = False
    print("Clicking stopped")

# Hotkeys
keyboard.add_hotkey("F6", start_clicking)
keyboard.add_hotkey("F7", stop_clicking)

print("F6 = Start clicking | F7 = Stop clicking")

# Run clicking thread
threading.Thread(target=auto_click, daemon=True).start()

# Keep script running
keyboard.wait()