# main.py

from typing import Annotated, Any
from fastapi import FastAPI, File, Form, Depends, HTTPException, Request, Body
from io import BytesIO
import base64
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
#import crud, models, schemas
from database import SessionLocal, engine
#from sqlalchemy.orm import Session
import logging

# load pre-trained saved model
model = tf.keras.models.load_model(f"{os.getcwd()}/saved_model")

app = FastAPI()

def img2txt(file: bytes):
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

def listToString(s):
     # initialize an empty string
    str1 = ""
     # traverse in the string
    for ele in s[0]:
        if ele != 10:
            str1 += str(ele)
     # return string
    return str1

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def root():
    return {"message": "Connection successful!"}

@app.post("/upload_img/")
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

@app.post("/add_platform/")
async def add_platform(requeststr: Any = Body(None)):
    with open("platform_db.json", "a") as write_file:
        write_file.write(requeststr)
    return {"status": "OK"}

@app.get("/fetch_platformdb/")
async def fetch_platformdb():
    with open("platform_db.json", "r") as file:
        data = file.read()
    return data



"""
@app.post("/create_vehicle/")
async def create_vehicle(
    id: Annotated[str, Form()],
    model: Annotated[str, Form()],
    n_axles: Annotated[int, Form()],
    n_wheels: Annotated[int, Form()],
    db: Session = Depends(get_db),
    **wheeldata
):
    try:
        vehicle = models.Vehicle(id, model, n_axles, n_wheels)
        crud.init_vehicle(db, vehicle)
        for i in range(n_wheels):
            crud.update_wheel(id, wheeldata)
        return {"message": "Vehicle succesfully created!"}
    except:
        return {"message": "ERROR: Vehicle creation failed!"}
"""

async def print_request(request):
    print(f'request header       : {dict(request.headers.items())}' )
    print(f'request query params : {dict(request.query_params.items())}')  
    try : 
        print(f'request json         : {await request.json()}')
    except Exception as err:
        # could not parse json
        print(f'request body         : {await request.body()}')

@app.post("/test/")
async def test_request(request: Request):
    try:
        await print_request(request)
        return {"status": "OK"}
    except Exception as err:
        logging.error(f'could not print REQUEST: {err}')
        return {"status": "ERR"}
    
@app.post("/testsaveb64/")
async def input_request_b64(
    fileb64: Annotated[str, Form()],
    token: Annotated[str, Form()]
):
    with open('saved_file_b64', 'wb') as save_file:
        save_file.write(base64.b64decode(fileb64))
    return token

@app.post("/testsave/")
async def input_request(
    file: Annotated[bytes, File()],
    token: Annotated[str, Form()],
):
    with open('file_saved', 'wb') as save_file:
        save_file.write(file)
    return token

"""
@app.post("/uploadVIN/")
async def accept_img_with_identifiers(
    file: Annotated[bytes, File()],
    vehicle_id: Annotated[str, Form()],
    wheel_id: Annotated[str, Form()],
):
    txt_out = img2txt(file)

    with open('image.jpg', 'wb') as save_file:
        save_file.write(file)

    #crud.update_vin(txt_out, vehicle_id, wheel_id)

    return {
        "file_size": len(file),
        "output": txt_out,
        "message": "Successful",
        }
"""

"""
@app.get("/view_vehicles/")
async def view_vehicles_db_request(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    vehicleModels = crud.get_vehicles(db, skip=skip, limit=limit)
    return vehicleModels

@app.get("/view_wheel_data/{vehicle_id}", response_model=schemas.VehicleModel)
def read_vehicle_wheel_data(vehicle_id: int, db: Session = Depends(get_db)):
    db_vehicle = crud.get_vehicle(db, vehicle_id=vehicle_id)
    if db_vehicle is None:
        raise HTTPException(status_code=404, detail="VehicleModel not found")
    return db_vehicle"""