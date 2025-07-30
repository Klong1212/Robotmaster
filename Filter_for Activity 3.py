from robomaster import robot
import time
from collections import deque
import statistics
import math

# ============ Parameters ============
WINDOW_SIZE = 5
DIST_THRESHOLD = 40       # cm
FILTER_TYPE = "lowpass"   # เลือก: "moving", "median", "lowpass"
SAMPLE_RATE = 5           # Hz (ep_sensor.sub_distance(freq=5))
CUTOFF_FREQ = 1.0         # Hz (สำหรับ Low-pass)

# ============ Filters ============

class MovingAverageFilter:
    def __init__(self, window_size):
        self.values = deque(maxlen=window_size)

    def filter(self, new_value):
        self.values.append(new_value)
        return sum(self.values) / len(self.values)

class MedianFilter:
    def __init__(self, window_size):
        self.values = deque(maxlen=window_size)

    def filter(self, new_value):
        self.values.append(new_value)
        return statistics.median(self.values)

class LowPassFilter:
    def __init__(self, cutoff_freq, sample_rate):
        self.dt = 1.0 / sample_rate
        self.alpha = (2 * math.pi * cutoff_freq * self.dt) / (2 * math.pi * cutoff_freq * self.dt + 1)
        self.last_output = None

    def filter(self, new_value):
        if self.last_output is None:
            self.last_output = new_value
        else:
            self.last_output = self.alpha * new_value + (1 - self.alpha) * self.last_output
        return self.last_output
