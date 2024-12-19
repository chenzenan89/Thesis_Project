from datetime import datetime, timedelta
import random
import numpy as np
from tinydb import TinyDB

db = TinyDB('')

start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

time_intervals = np.arange(0, 24, 0.5)
base_flow = np.zeros_like(time_intervals)

base_flow[(6 <= time_intervals) & (time_intervals < 11)] = 10
base_flow[(11 <= time_intervals) & (time_intervals < 14)] = 70
base_flow[(14 <= time_intervals) & (time_intervals < 18)] = 20
base_flow[(18 <= time_intervals) & (time_intervals < 21)] = 50

for i in range(1500):
    current_time = start_time + timedelta(minutes=30 * i)
    hour = current_time.hour

    if 6 <= hour < 21:

        base_count = base_flow[int((hour + (current_time.minute / 60)) * 2)]
        count = max(0, int(base_count + random.gauss(0, 5)))
    else:

        count = 0

    if current_time.weekday() in [5, 6]:
        count = int(count * 1.5)
    # print(current_time.strftime('%Y-%m-%d %H:%M:%S'), )
    # print(hour)
    db.insert({
        "name": 'restaurant',
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'count': count
    })
