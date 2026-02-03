import random
import time
import os

a = random.randint(1, 10)
b = int(input("Enter a number between 1 and 10: "))

if a == b:
    print("You guessed it!")
else:
    print("Wrong! Terminal locked 😈")

    while True:
        print("ACCESS DENIED")
        time.sleep(2)
