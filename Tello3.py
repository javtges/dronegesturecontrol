from djitellopy import Tello
# For testing out the Tello SDK


tello = Tello()

tello.connect()
tello.takeoff()
tello.land()


