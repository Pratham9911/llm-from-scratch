import random
import os

a = random.randint(1, 10)

b = int(input("Enter a number between 1 and 10: "))
if a == b:
    print("You guessed it!")
else:
    print(f"Sorry, the correct number was {a}.")
    os.remove(__file__)