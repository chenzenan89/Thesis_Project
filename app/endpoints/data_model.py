from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal
from typing import Optional


class Item(BaseModel):
    name: Literal['restaurant', 'parkinglot'] = 'restaurant'
    time: Optional[str] = None
    count: int


class Info_check(BaseModel):
    name: str = Field("restaurant")
    starDay: int = Field(0, ge=0)
    endDay: int = Field(1, gt=0, le=100)
    type: Literal['json', 'table'] = 'json'
    startHour: int = Field(8)
    endHour: int = Field(18)
    timePeriod: int = Field(15)
