import sys
import time
sys.path.append('../galatae-api/')
from robot import Robot

#Glob variables
vitesse = 50
posReplis = (300,0,300,90,0)
screwM3_6 =(390,0,120,180,0)
screwM3_8 =(418,28,120,180,0)
screwM3_10 =(362,-28,120,180,0)

# Initialisation du robot
r = Robot('/dev/ttyACM0')
time.sleep(3)
r.set_joint_speed(vitesse)
r.reset_pos()
r.go_to_point([300,0,150,180,0])

# r.go_to_point(posReplis)
# r.go_to_point(screwM3_6)
# time.sleep(2)
# r.go_to_point(screwM3_8)
# time.sleep(2)
# r.go_to_point(screwM3_10)
# #r.go_to_foetus_pos()
# print("Fin test")