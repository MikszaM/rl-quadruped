import Adafruit_PCA9685
import time

PWM_MIN = 73
PWM_MAX = 546
PWM_MID = (PWM_MAX+PWM_MIN)//2
LEGS = [0, 2, 12, 14]
FEET = [1, 3, 13, 15]
SERVOS = LEGS + FEET
SERVOS.sort()
def test_servo(pwm, channel):
    pwm.set_pwm(channel, 0, PWM_MIN)
    print("In min position")
    time.sleep(2)
    pwm.set_pwm(channel, 0, PWM_MAX)
    print("in max position")
    time.sleep(2)
    pwm.set_pwm(channel, 0, PWM_MID)
    print("in mid position")
    time.sleep(2)


def test_legs_servos(pwm):
    for pin in LEGS:
        test_servo(pwm, pin)


def test_feet_servos(pwm):
    for pin in FEET:
        test_servo(pwm, pin)


def init_controller():
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(50)
    return pwm


def normalize_value(value, minimum, maximum):
    value = maximum if value > maximum else value
    value = minimum if value < minimum else value
    return (value-minimum)/(maximum-minimum)


def scale_value(value, minimum, maximum):
    return int(minimum + (maximum-minimum)*value)


def set_servos(pwm, values):
    for i in range(8):
        set_servo(pwm,SERVOS[i],values[i])


def set_servo(pwm, number, value):
    norm = normalize_value(value, -1, 1)
    if number in FEET:
        if number in [3, 13]:
            scaled = scale_value(norm, PWM_MAX, PWM_MIN)
        else:
            scaled = scale_value(norm, PWM_MIN, PWM_MAX)
    elif number in LEGS:
        if number == 2:
            scaled = scale_value(norm, 450, 122)
        elif number == 0:
            scaled = scale_value(norm, 200, 516)
        elif number == 12:
            scaled = scale_value(norm, 436, 125)
        elif number == 14:
            scaled = scale_value(norm, 180, 506)
    pwm.set_pwm(number, 0, scaled)


def zero_legs(pwm):
    for leg in LEGS:
        set_servo(pwm,leg,0)


def zero_feet(pwm):
    for foot in FEET:
        set_servo(pwm,foot,-1)

def walk_try(pwm):
    print("Walk try started")
    
    print("Walk try finished")

def dance_for_me(pwm):
    print("Dance started")
    poses = []
    #poses.append([50, 0, 50, 0, 50, 0, 50, 0])
    #poses.append([100, 0, 100, 0, 100, 0, 100, 0])
    #poses.append([50, 0, 50, 0, 50, 0, 50, 0])
    #poses.append([40, 75, 40, 75, 40, 75, 40, 75])
    #poses.append([40, 60, 40, 60, 40, 75, 40, 75])
    #poses.append([0, 0, 0, 0, 60, 60, 60, 60])
    #poses.append([0, 0, 0, 0, 30, 30, 30, 30])
    #poses.append([0, 0, 0, 0, 0, 0, 0, 0])
    for pose in poses:
        set_servos(pwm, pose)
        time.sleep(3)
    print("Dance finished")

if __name__ == "__main__":
    pwm = init_controller()
    zero_feet(pwm)
    
    zero_legs(pwm)
    print('Initial positions set')
    valuef = 0
    valuel = 0
    flag = True
    while True:
        char = input()
        if char == 'w':
            valuef += 0.05
        elif char == 's':
            valuef -= 0.05
        elif char == 'e':
            valuef += 0.002
        elif char == 'd':
            valuef -= 0.002
        elif char == 'r':
            valuel += 0.05
        elif char == 'f':
            valuel -= 0.05
        elif char == 't':
            valuel += 0.002
        elif char == 'g':
            valuel -= 0.002
        elif char == 'p':
            dance_for_me(pwm)
            flag = False
        elif char == 'l':
            flag = True
        elif char == 'z':
            zero_feet(pwm)
            zero_legs(pwm)

        
        print(f'Current value feet: {valuef}')
        print(f'Current value legs: {valuel}')
        #pwm.set_pwm(13, 0, value)
        if flag:
            for foot in FEET:
                set_servo(pwm, foot, valuef)
            for leg in LEGS:
                set_servo(pwm, leg, valuel)
