import os
import json
from datetime import datetime
from typing import Optional, List
from fastapi import UploadFile, File, HTTPException, APIRouter, Depends

from model.train import trainer
from model.logs.set_log import logger
from endpoints.data_model import Item
from config.setting import SettingsManager
from database.database_manage import Database

# Set up a FastAPI router
app_collect = APIRouter()

# Path where database files are stored
PATH = '/home/chen/Thesis_Project/app/database/data'

# Load configuration settings
setting_info = SettingsManager()


@app_collect.post("/api/collect")
async def get_something(files: Optional[List[UploadFile]] = File(None)):
    # Check if files are provided
    if not files:
        return {"message": "No files uploaded"}

    erro_file_dic = {}
    data = []

    for file in files:
        # Check file type
        if file.content_type != "application/json":
            erro_file_dic[
                file.
                filename] = "Invalid file format. Only JSON files are accepted."
        else:
            try:
                # Read and parse JSON content
                content = await file.read()
                data_json = json.loads(content)
                a = Item(**data_json)  # Validate data with the Item model
                data.append(a)
            except (json.JSONDecodeError, ValueError):
                erro_file_dic[
                    file.
                    filename] = "Invalid JSON content. Please check the info section"

    # Handle errors for invalid files
    if erro_file_dic:
        erro_massage = ", ".join(f"Error file: {key} || Error reason: {value}"
                                 for key, value in erro_file_dic.items())
        raise HTTPException(status_code=400, detail=erro_massage)

    return {"message": "successful"}


@app_collect.post("/api/data_collect")
async def data_collect(data: Item = Depends()):
    # Extract data fields
    name = data.name
    count = data.count
    now = datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    # Save data to a JSON file
    path = os.path.join(PATH, f'{name}_database.json')
    database = Database(path=path)
    database.insert({'name': name, 'timestamp': time, 'count': count})

    # Log the operation
    log_work = logger(name)
    total_count = len(database.all_check())
    log_work.update_first_line(
        f"Data inserted for project {name}. Total records: {total_count}")

    # Get project-specific settings
    project = setting_info.get_project_config(name)
    basic_threshold = project.get("basic_threshold")
    threshold = project.get("threshold")

    # Check if the model should be trained
    if total_count == basic_threshold or (total_count > basic_threshold and
                                          (total_count - basic_threshold) %
                                          threshold == 0):
        data = database.all_check()
        sliding_window_data = data[
            -basic_threshold:]  # Get the latest data for training
        train_work = trainer(project)

        # Log training process
        log_work.log.info(f"Model starts training for project {name}")
        train_work.train_save(data=sliding_window_data)
        log_work.log.info(f"Model trained successfully for project {name}.")
        log_work.log.info(
            f"Data range: {sliding_window_data[0].get('timestamp')} to {sliding_window_data[-1].get('timestamp')}"
        )

    return {"message": "successful"}
