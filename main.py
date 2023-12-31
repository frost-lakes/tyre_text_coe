# main.py

import os
from typing import Any, Optional, List
from typing_extensions import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, Body, status, BackgroundTasks
from fastapi.responses import Response
from pydantic import ConfigDict, BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from sse_starlette import EventSourceResponse

from bson import ObjectId
import motor.motor_asyncio
from pymongo import ReturnDocument

from io import BytesIO
import base64
import tensorflow as tf
import numpy as np
from PIL import Image
import logging
import time
import asyncio

# load pre-trained saved model
model = tf.keras.models.load_model(f"{os.getcwd()}/saved_model")

# FastAPI setup
app = FastAPI(
    title="Tyretext API",
    summary="Extract image from text and manage car data using this API.",
)

# MongoDB setup
client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost") #os.environ["MONGODB_URL"])
db = client.assemblydb
car_collection = db.get_collection("cars")

# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]

# DB update flag setup
db_updated = 1

async def img2txt(file: bytes):
    # Open the image
    image = Image.open(BytesIO(file))
    # Resize the image to 32x32 pixels
    resized_image = image.resize((32, 32), Image.BICUBIC)
    # Convert the resized image to grayscale
    output_image = resized_image.convert("L")
    #loading grayscale image of size (32,32)
    test_image=np.array(output_image)
    #reshaping image
    test_image = test_image.reshape(1, 32, 32, 1)

    # predicting number in input image
    model_prediction = model.predict(test_image)
    # taking maximum probability numbery
    prediction_class = tf.transpose(tf.argmax(model_prediction, axis=2))
    list_form = np.array(prediction_class).tolist()
    string_form = listToString(list_form)
    return string_form

async def listToString(s):
     # initialize an empty string
    str1 = ""
     # traverse in the string
    for ele in s[0]:
        if ele != 10:
            str1 += str(ele)
     # return string
    return str1

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Connection successful!"}

# Route for image to text conversion
@app.post("/upload_img/", tags=["Image to Text"])
async def accept_img_demo(
    file: Annotated[bytes, File()],
    token: Annotated[str, Form()],
):
    txt_out = img2txt(file)

    return {
        "file_size": len(file),
        "output": txt_out,
        "received_str": token,
        }

# Classes and routes for db manipulation
class Wheel(BaseModel):
    location: str = Field(...)
    vin: str = Field(...)
    make: str = Field(...)
    size: str = Field(...)

class CarModel(BaseModel):
    """
    Container for a single car record.
    """

    # The primary key for the CarModel, stored as a `str` on the instance.
    # This will be aliased to `_id` when sent to MongoDB,
    # but provided as `id` in the API requests and responses.
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    mod_num: str = Field(...)
    axle_num: int = Field(...)
    wheels_data: list[Wheel] = Field(...) # | None = None
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "mod_num": "ZXBVNMC",
                "axle_num": 5,
                "wheels_data": [
                    {
                        "location": "Front Axle 1 LH",
                        "vin": "CXX123401",
                        "make": "Diablo",
                        "size": "35",
                    },
                    {
                        "location": "Front Axle 1 RH",
                        "vin": "CYY123403",
                        "make": "Rouge",
                        "size": "35",
                    }
                ],
            }
        },
    )

class UpdateCarModel(BaseModel):
    """
    A set of optional updates to be made to a document in the database.
    """

    mod_num: Optional[str] = None
    axle_num: Optional[int] = None
    wheels_data: Optional[list[Wheel]] = None
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
        json_schema_extra={
            "example": {
                "mod_num": "ZXBVNMC",
                "axle_num": 5,
                "wheels_data": [
                    {
                        "location": "Front Axle 1 LH",
                        "vin": "CXX123401",
                        "make": "Diablo",
                        "size": "35",
                    },
                    {
                        "location": "Front Axle 1 RH",
                        "vin": "CYY123403",
                        "make": "Rouge",
                        "size": "35",
                    }
                ],
            }
        },
    )

class CarCollection(BaseModel):
    """
    A container holding a list of `CarModel` instances.

    This exists because providing a top-level array in a JSON response can be a [vulnerability](https://haacked.com/archive/2009/06/25/json-hijacking.aspx/)
    """

    cars: List[CarModel]

@app.post(
    "/cars/",
    response_description="Add new car record",
    response_model=CarModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
    tags=["CRUD"],
)
async def create_car(car: CarModel = Body(...)):
    """
    Insert a new car record.

    A unique `id` will be created and provided in the response.
    """
    new_car = await car_collection.insert_one(
        car.model_dump(by_alias=True, exclude=["id"])
    )
    created_car = await car_collection.find_one(
        {"_id": new_car.inserted_id}
    )

    # Set db_updated flag
    global db_updated
    db_updated = 1

    return created_car


@app.get(
    "/cars/",
    response_description="List all car records",
    response_model=CarCollection,
    response_model_by_alias=False,
    tags=["CRUD"],
)
async def list_cars():
    """
    List all of the car data in the database.

    The response is unpaginated and limited to 1000 results.
    """
    return CarCollection(cars=await car_collection.find().to_list(1000))


@app.get(
    "/cars/{id}",
    response_description="Get a single car record",
    response_model=CarModel,
    response_model_by_alias=False,
    tags=["CRUD"],
)
async def show_car(id: str):
    """
    Get the record for a specific car, looked up by `id`.
    """
    if (
        car := await car_collection.find_one({"_id": ObjectId(id)})
    ) is not None:
        return car

    raise HTTPException(status_code=404, detail=f"Car {id} not found")


@app.put(
    "/cars/{id}",
    response_description="Update a car record",
    response_model=CarModel,
    response_model_by_alias=False,
    tags=["CRUD"],
)
async def update_car(id: str, car: UpdateCarModel = Body(...)):
    """
    Update individual fields of an existing car record.

    Only the provided fields will be updated.
    Any missing or `null` fields will be ignored.
    """
    car = {
        k: v for k, v in car.model_dump(by_alias=True).items() if v is not None
    }

    if len(car) >= 1:
        update_result = await car_collection.find_one_and_update(
            {"_id": ObjectId(id)},
            {"$set": car},
            return_document=ReturnDocument.AFTER,
        )
        if update_result is not None:
            
            # Set db_updated flag
            global db_updated
            db_updated = 1

            return update_result
        else:
            raise HTTPException(status_code=404, detail=f"Car {id} not found")

    # The update is empty, but we should still return the matching document:
    if (existing_car := await car_collection.find_one({"_id": id})) is not None:
        return existing_car

    raise HTTPException(status_code=404, detail=f"Car {id} not found")


@app.delete("/cars/{id}", response_description="Delete a car record", tags=["CRUD"],)
async def delete_car(id: str):
    """
    Remove a single car record from the database.
    """
    delete_result = await car_collection.delete_one({"_id": ObjectId(id)})

    if delete_result.deleted_count == 1:
        
        # Set db_updated flag
        global db_updated
        db_updated = 1

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    raise HTTPException(status_code=404, detail=f"Car {id} not found")

# Route for database update notifications
@app.get("/updates/", tags=["Updates"])
async def update_notify(request: Request):
      
    async def check_db():
        global db_updated
        i = 0
        
        while True:
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                break
            
            if db_updated:
                # the contents of the new event
                i += 1
                db_updated = 0
                msg = "The database has been updated at %s." % time.strftime("%H:%M:%S", time.localtime())
                # send the event to the client with yield: EventSourceResponse requires the
                # event be returned as a dictionary, where keys (id, data, and event)
                # correspond to SSE properties
                yield {
                    "id": i,
                    "data": msg,
                    "event": "DB_UPDATED"
                }
    
            await asyncio.sleep(1)

    return EventSourceResponse(check_db())

# Routes for testing
async def print_request(request):
    print(f'request header       : {dict(request.headers.items())}' )
    print(f'request query params : {dict(request.query_params.items())}')  
    try : 
        print(f'request json         : {await request.json()}')
    except Exception as err:
        # could not parse json
        print(f'request body         : {await request.body()}')

@app.post("/test/", tags=["Test"])
async def test_request(request: Request):
    try:
        await print_request(request)
        return {"status": "OK"}
    except Exception as err:
        logging.error(f'could not print REQUEST: {err}')
        return {"status": "ERR"}
    
@app.post("/testsaveb64/", tags=["Test"])
async def input_request_b64(
    fileb64: Annotated[str, Form()],
    token: Annotated[str, Form()]
):
    with open('saved_file_b64', 'wb') as save_file:
        save_file.write(base64.b64decode(fileb64))
    return token

@app.post("/testsave/", tags=["Test"])
async def input_request(
    file: Annotated[bytes, File()],
    token: Annotated[str, Form()],
):
    with open('file_saved', 'wb') as save_file:
        save_file.write(file)
    return token