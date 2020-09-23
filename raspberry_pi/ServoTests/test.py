import Adafruit_PCA9685

PWM_MIN = 73
PWM_MAX = 546
PWM_MID = (PWM_MAX+PWM_MIN)//2

def init_controller():
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(50)
    return pwm

if __name__=="__main__":
    pwm = init_controller()
    channel = 13
    while True:
        # Move servo on channel 12 between extremes.
        pwm.set_pwm(channel, 0, PWM_MIN)
        print("In min position")
        input()
        pwm.set_pwm(channel, 0, PWM_MAX)
        print("in max position")
        input()
        pwm.set_pwm(channel, 0, PWM_MID)
        print("in mid position")
        input()