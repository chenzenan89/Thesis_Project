from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from endpoints.collect import app_collect as app1
from endpoints.visvalize import app_visvalize as app2
from endpoints.root import app_root as app0

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory="/home/chen/Thesis_Project/app/templates/static"),
    name="static")

app.include_router(app0, tags=["root"])
app.include_router(app1, tags=["collection"])
app.include_router(app2, tags=["visvaliztion"])

if __name__ == '__main__':
    uvicorn.run("main:app", port=8090, reload=True)
