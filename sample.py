import random
import ctypes

a = random.randint(1, 10)
b = int(input("Enter a number between 1 and 10: "))

if a == b:
    print("You guessed it!")
else:
    print("Wrong! Turning off screen...")
    ctypes.windll.user32.SendMessageW(0xFFFF, 0x0112, 0xF170, 2)
