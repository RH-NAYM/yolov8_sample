from main import *
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from typing import List, Union
import uvicorn
import json
import torch
import logging
import pytz
from datetime import datetime

logging.basicConfig(filename="Unilever_Log.log",
                    filemode='w')
logger = logging.getLogger("Unilever")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("Unilever_Log.log")
logger.addHandler(file_handler)
total_done = 0
total_error = 0

app = FastAPI()

class Item(BaseModel):
    url : str

def get_bd_time():
    bd_timezone = pytz.timezone("Asia/Dhaka")
    time_now = datetime.now(bd_timezone)
    current_time = time_now.strftime("%I:%M:%S %p")
    return current_time


async def process_item(item: Item):
    try:
        result = await mainDet(item.url)
        result = json.loads(result)
        return result
    finally:
        torch.cuda.empty_cache()
        pass

async def process_items(items: Union[Item, List[Item]]):
    print(type(items))
    if type(items)==list:
        coroutines = [process_item(item) for item in items]
        results = await asyncio.gather(*coroutines)
    else:
        results = await process_item(items)
    return results

@app.get("/status")
async def status():
    return "AI Server in running"



@app.post("/unilever")
async def create_items(items: Union[Item, List[Item]]):
    try:
        results = await process_items(items)
        print("Result Sent to User:", results)
        print("###################################################################################################")
        print(items)
        return results
    except Exception as e:
        global total_error
        total_error += 1
        logger.info(f"Time:{get_bd_time()}, Failed Execution : {total_error}, Payload:{items}")
        logger.error(str(e))
        return {"AI": f"Error: {str(e)}"}
    finally:
        global total_done
        total_done +=1
        logger.info(f"Time:{get_bd_time()}, Successfull Execution : {total_done}, Payload:{items}, Response : {results}")
        torch.cuda.empty_cache()
        pass

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=5656)
    finally:
        torch.cuda.empty_cache()
