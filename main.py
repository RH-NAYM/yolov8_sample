import json
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import aiohttp
from aiohttp import ClientSession
from io import BytesIO
import asyncio

model = YOLO('AI_Models/best.pt')

async def img(img_url, session):
    async with session.get(img_url) as response:
        img_data = await response.read()
        return BytesIO(img_data)

async def det(img_content):
    img = Image.open(img_content)
    result = model(img)
    
    detection = {}
    data = json.loads(result[0].tojson())
    if len(data) == 0:
        res = {"AI": "No Detection"}
        detection.update(res)
    else:
        df = pd.DataFrame(data)
        name_counts = df['name'].value_counts().sort_index()
        
        for name, count in name_counts.items():
            res = {name: count}
            detection.update(res)
        
    # print(detection)
    return detection

async def mainDet(url):
    async with ClientSession() as session:
        image = await img(url, session)
        detec = await det(image)
        reply = json.dumps(detec)
        return reply


