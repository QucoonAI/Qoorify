from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import os 
from src.anti_spoof import liveliness_left,liveliness_right,Qoorify_spoof,Qoorify_KYC

app = FastAPI()

@app.post("/liveliness/right/")
async def liveliness_right_endpoint(image: UploadFile = File(...)):
    """
    Endpoint to check the liveliness of a right face image. Returns a JSON response with the result of the check.
    """
    temp_dir = "./temp_images"
    os.makedirs(temp_dir, exist_ok=True)

    temp_image_path = os.path.join(temp_dir, "temp_image.jpg")
    with open(temp_image_path, "wb") as f:
        f.write(await image.read())

    result = liveliness_right(temp_image_path)

    # Cleanup temporary image
    os.remove(temp_image_path)

    return JSONResponse(content={"liveliness_right": result})


@app.post("/liveliness/left/")
async def liveliness_left_endpoint(image: UploadFile = File(...)):
    """
    Endpoint to check the liveliness of a left face image. Returns a JSON response with the result of the check.
    """
    temp_dir = "./temp_images"
    os.makedirs(temp_dir, exist_ok=True)

    temp_image_path = os.path.join(temp_dir, "temp_image.jpg")
    with open(temp_image_path, "wb") as f:
        f.write(await image.read())

    result = liveliness_left(temp_image_path)

    # Cleanup temporary image
    os.remove(temp_image_path)

    return JSONResponse(content={"liveliness_left": result})


@app.post("/spoof/")
async def spoof_endpoint(image: UploadFile = File(...)):
    """
    Endpoint to check if an image is spoofed or not. Returns a JSON response with the result of the check. True if live image, False if spoofed.
    """
    temp_dir = "./temp_images"
    os.makedirs(temp_dir, exist_ok=True)

    temp_image_path = os.path.join(temp_dir, "temp_image.jpg")
    with open(temp_image_path, "wb") as f:
        f.write(await image.read())

    result = Qoorify_spoof(temp_image_path)

    # Cleanup temporary image
    os.remove(temp_image_path)

    return JSONResponse(content={"not-spoof": result})


@app.post("/kyc/")
async def kyc_check(right_image: UploadFile = File(...), 
                    left_image: UploadFile = File(...), 
                    spoof_image: UploadFile = File(...)):
    """
    KYC verification endpoint. This endpoint takes three images as input:
    - Right face image
    - Left face image
    - Spoof image

    The endpoint returns a JSON response with the status of the KYC verification.
    """
    image1 = right_image
    image2 = left_image
    image3 = spoof_image
    # Save uploaded images temporarily
    temp_dir = "./temp_images"
    os.makedirs(temp_dir, exist_ok=True)

    image_paths = []
    for idx, image in enumerate([image1, image2, image3], start=1):
        temp_image_path = os.path.join(temp_dir, f"image{idx}.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(await image.read())
        image_paths.append(temp_image_path)

    # Collect results from individual checks
    liveliness_right_result = liveliness_right(image_paths[0])
    liveliness_left_result = liveliness_left(image_paths[1])
    spoof_result = Qoorify_spoof(image_paths[2])

    # Cleanup temporary images
    for path in image_paths:
        os.remove(path)

    # Define KYC verification logic
    if liveliness_right_result and liveliness_left_result and not spoof_result:
        return JSONResponse(content={"status": "Passed"})
    else:
        return JSONResponse(content={"status": "Failed"})

