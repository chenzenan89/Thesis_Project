from fastapi import APIRouter
from fastapi.templating import Jinja2Templates
from fastapi import Request

app_root = APIRouter()

templates = Jinja2Templates(
    directory="/home/chen/Thesis_Project/app/templates")


@app_root.get("/")
async def main(request: Request):

    return templates.TemplateResponse("index.html", {'request': request})
