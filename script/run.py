from lib.procedure.train import train
import sys
import os

_root_path = os.path.join(os.path.dirname(__file__) , '..')
sys.path.insert(0, _root_path)

if __name__ == '__main__':
    train()