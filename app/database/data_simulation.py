from datetime import datetime, timedelta
import random
import numpy as np
from tinydb import TinyDB

db = TinyDB('/home/chen/Thesis_Project/app/database/test_1.json')

start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

time_intervals = np.arange(0, 24, 0.5)
base_flow = np.zeros_like(time_intervals)

base_flow[(6 <= time_intervals) & (time_intervals < 11)] = 10
base_flow[(11 <= time_intervals) & (time_intervals < 14)] = 70
base_flow[(14 <= time_intervals) & (time_intervals < 18)] = 20
base_flow[(18 <= time_intervals) & (time_intervals < 21)] = 50

# 创建 1500 条数据
for i in range(1500):
    current_time = start_time + timedelta(minutes=30 * i)
    hour = current_time.hour

    if 6 <= hour < 21:
        # 添加噪声
        base_count = base_flow[int((hour + (current_time.minute / 60)) * 2)]
        count = max(0, int(base_count + random.gauss(0, 5)))  # 添加正态噪声
    else:
        # 非营业时间流量为 0
        count = 0

    # # 添加节假日和周末效应
    # holidays = ['2025-01-01', '2025-12-25']
    # if current_time.strftime('%Y-%m-%d') in holidays:
    #     count += random.randint(20, 50)  # 节假日流量增加

    if current_time.weekday() in [5, 6]:  # 周末
        count = int(count * 1.5)  # 周末流量增加
    print(current_time.strftime('%Y-%m-%d %H:%M:%S'), )
    print(hour)
    # 存储到 TinyDB
    db.insert({
        "name": 'restaurant',
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'count': count
    })
# for i in range(1500):

#     current_time = start_time + timedelta(minutes=30 * i)
#     hour = current_time.hour

#     if 6 <= hour < 11:  # Morning (low)
#         count = random.randint(0, 20)
#     elif 11 <= hour < 14:  # Lunch (high)
#         count = random.randint(40, 100)
#     elif 14 <= hour < 18:  # Afternoon (moderate)
#         count = random.randint(10, 30)
#     elif 18 <= hour < 21:  # Dinner (high-moderate)
#         count = random.randint(0, 60)
#     else:  # Late night and early morning (very low)
#         count = random.randint(0, 0)

#     print(current_time.strftime('%Y-%m-%d %H:%M:%S'), )
#     print(hour)
#     # Insert each record into TinyDB with a timestamp and count
#     db.insert({
#         "name": 'restaurant',
#         'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
#         'count': count
#     })

# print(
#     "TinyDB database created with 1000 records following a daily occupancy pattern."
# )

# # "2000": {
# #     "name": "restaurant",
# #     "timestamp": "2024-12-17 15:30:00",
# #     "count": 27
# # }
