# Melanoma Binary Segmentation Lab
 
This FastAPI service allows clients to upload an image and perform neural network (NN) inference on it. The server resizes the image, processes it through a specified model, and returns the inference results as a PNG image.  

## Build documentations

`sphinx-build -b html docs/source docs/build`

And in docs/build you may find generated docs

## Installation  
### Docker
1. Just build and run docker with bash file- it will build and run container
    ```bash
    sh build_docker.sh
    ```

### Manual
1. **Create and activate a virtual environment:**  
    ```bash  
    python3 -m venv venv  
    source venv/bin/activate  
    ```  
   
2. **Install the required dependencies:**  
    ```bash  
    pip install -r requirements.txt  
    ```  
   
## Running the Server  (for manual usage)
   
To start the FastAPI server, run:  
```bash  
uvicorn application:app --log-level info  
```  
   
The server will be available at `http://127.0.0.1:8000`.  
   
## API Endpoints  
   
### `/nn_inference`  
   
#### Description  
   
This endpoint accepts an image file and a neural network type, performs inference using the specified model, and returns the processed image.  
   
#### Request  
   
- **Method**: `POST`  
- **Content-Type**: `multipart/form-data`  
   
#### Parameters  
   
| Name     | Type       | Description                      |  
|----------|------------|----------------------------------|  
| `file`   | `UploadFile` | The BMP image file to be processed. |  
| `nn_type` | `Form`      | The type of neural network model to use for inference. |  

choose `nn_type` from ("segnet", "unet", "naive"), by default- segnet
   
#### Example cURL Request  
   
```bash  
curl -X POST "http://127.0.0.1:8000/nn_inference" \  
-F "file=@path/to/your/image.bmp" \  
-F "nn_type=model_type"
-o segmentation.png
```  
   
#### Response  
   
- **Content-Type**: `image/png`  
- Returns the image with inference results.  

#### Example of usage
1. Via bash file `client_curl.bash` which equal next:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/nn_inference' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@IMD036.bmp' \
  -F "nn_type=unet" \
  -o segmentation.png
```

2. Via python script:
    ```bash
    source venv/bin/activate && python client_python.py
    ```
   
## Logging  
   
The server logs the following information:  
- Shape of the processed image.  
- Type of the model used for inference.  
- Shape of the inference results.  
   
## Additional Notes  
   
- Ensure that the `get_container`, `get_model_from_container`, `read_file`, `resize`, `imread`, `create_img`, and `logger` functions/utilities are defined and properly implemented in your codebase.  
- The example assumes that the image processing libraries (e.g., `skimage`, `torch`, `numpy`) are installed and properly configured.  
   
## Contributing  
   
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or improvements.  
   
## License  
   
This project is licensed under the MIT License.  
```  
   
This `README.md` provides a comprehensive overview of the FastAPI endpoint, including installation instructions, API usage, and detailed explanations of the endpoint's functionality. Make sure to adjust paths, URLs, and other details according to your actual project setup.
```

## Project tree

```
.
├── .env
├── .gitignore
├── Dockerfile
├── IMD036.bmp
├── README.md
├── application.py
├── artifacts
│   └── models
│       ├── NaiveCNN
│       │   └── NaiveCNN_dice_250e.pt
│       ├── SegNet
│       │   └── SegNet_dice_250e.pt
│       ├── SegNet_1
│       │   └── SegNet_dice_250e.pt
│       └── UNet
│           └── UNet_dice_250e.pt
├── build_docker.sh
├── clients
│   ├── client_curl.bash
│   └── client_python.py
├── docs
│   ├── Makefile
│   ├── make.bat
│   └── source
│       ├── _static
│       ├── _templates
│       ├── application.rst
│       ├── conf.py
│       ├── index.rst
│       ├── nn_models.rst
│       ├── provider.rst
│       ├── train_losses.rst
│       ├── trainer.rst
│       └── utils.rst
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   └── provider
│   │       ├── gdrive_weights.py
│   │       ├── iph2_provider.py
│   │       ├── ph2_provider.py
│   │       └── provider_dependency_injection.py
│   ├── endpoints
│   │   ├── __init__.py
│   │   └── nn_inference.py
│   ├── metrics
│   │   ├── __init__.py
│   │   ├── losses
│   │   │   ├── __init__.py
│   │   │   ├── bce_loss.py
│   │   │   ├── dice_loss.py
│   │   │   ├── focal_loss.py
│   │   │   └── iloss.py
│   │   ├── metrics_dependency_injection.py
│   │   └── quality
│   │       ├── __init__.py
│   │       └── iou.py
│   ├── nn_models
│   │   ├── __init__.py
│   │   ├── naive_cnn.py
│   │   ├── nn_dependency_injection.py
│   │   ├── segnet.py
│   │   └── unet.py
│   ├── trainer
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── trainer_dependency_injection.py
│   └── utils
│       ├── get_model.py
│       └── read_files.py
└── train.py
```
