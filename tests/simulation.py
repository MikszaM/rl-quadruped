import pybullet as p 
from time import sleep
p.connect(p.GUI)
p.loadURDF("simulation/robot.urdf") 
sleep(100) 