
import pandas as pd
from ultralytics import YOLO
import asyncio
import pytz
from datetime import datetime
import json




def get_bd_time():
    bd_timezone = pytz.timezone("Asia/Dhaka")
    time_now = datetime.now(bd_timezone)
    current_time = time_now.strftime("%I:%M:%S %p")
    return current_time


model = YOLO('AI_Models/best.pt')

class_names = {0: 'Ferdous', 1: 'Himel', 2: 'Minhaz', 3: 'Munna', 4: 'Raihan', 5: 'Rakib', 6: 'Sakib', 7: 'Shahadot'}

def detect_objects_sync(model, url):
    results = model(url)
    detections_list = []
    detection_threshold = 0.1
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > detection_threshold:
                class_name = class_names.get(int(class_id), 'Unknown')
                detections_list.append([x1, y1, x2, y2, score, class_id, class_name])

    head = ["x1", "y1", "x2", "y2", "score", "class_id", "class_name"]
    df = pd.DataFrame(detections_list, columns=head)

    name_counts = df['class_name'].value_counts().to_dict()

    result_dict = {}
    for index, row in df.iterrows():
        name = row['class_name']
        result_dict[name] = name_counts.get(name, 0)

    return result_dict

async def detect_objects(url):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, detect_objects_sync, model, url)

async def detect_seq(url):
    results = await detect_objects(url)
    r = json.dumps(results)
    return r


