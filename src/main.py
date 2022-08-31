__author__ = "Anil Yelin"

from util.data import Data
from framework.robustness_check import RobustnessCheck

def menu():
    while True:
        print("Automated XAI Framework v0.0.1")
        print("++++++++++++++++++++++++++++++++++++++++++")
        print()
        print("Select one from the following options")
        print("[1] - Show Data")
        print("[2] - Robustness Check")
        print("[3] - To quit")
        print()
        print("++++++++++++++++++++++++++++++++++++++++++")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            obj = Data()
            obj.showHead()
        elif choice == 2:
            obj = RobustnessCheck()
            obj.prepareData()
            obj.test()
        elif choice == 3:
            print("Quitting program")
            break 
        else:
            print("No valid input, quitting program")
            break 
if __name__ == "__main__":
    menu()