import json
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse
from typing_extensions import Annotated

from endpoints.data_model import Info_check
from model.train import trainer
from model.data_manage import Data_engineering
from config.setting import SettingsManager

# Set up the FastAPI router for visualization
app_visvalize = APIRouter()

# Load Jinja2 templates for HTML rendering
templates = Jinja2Templates(
    directory="/home/chen/Thesis_Project/app/templates")


@app_visvalize.get("/api/visvalize")
async def get_something(request: Request,
                        info: Annotated[Info_check, Query()] = None):
    """
    Endpoint to visualize data or get predictions in JSON or HTML format.
    Args:
        request (Request): The HTTP request object.
        info (Info_check): Query parameters for the visualization.

    Returns:
        JSONResponse or an HTML TemplateResponse based on `info.type`.
    """
    # Get project-specific settings
    project = SettingsManager().get_project_config(project_name=info.name)

    # Generate predictions for the specified time range
    _, prediction = trainer(project).prediction(start_day=info.starDay,
                                                end_day=info.endDay,
                                                freq=info.timePeriod)

    # Process and prepare data for visualization
    data = Data_engineering().dynamic_display(prediction,
                                              startHour=info.startHour,
                                              endHour=info.endHour,
                                              min=project.get("min"),
                                              max=project.get("max"))

    # If JSON format is requested, return a JSON response
    if info.type == 'json':
        return JSONResponse(content=json.loads(data.get("json")),
                            status_code=200)

    # Otherwise, render the data as an HTML table
    return templates.TemplateResponse(
        "table.html",
        {
            "request": request,
            "row_headers":
            data.get("row_headers"),  # Row headers for the table
            "col_headers":
            data.get("col_headers"),  # Column headers for the table
            "values": data.get("values"),  # Table cell values
            "colors": data.get("colors"),  # Cell background colors
            "zip": zip,  # Zip function for Jinja2 template
        })
