import requests
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


url = "http://127.0.0.1:8000/nn_inference"  
photo_id = "043"


file_path = f"../dataset/PH2Dataset/PH2 Dataset images/IMD{photo_id}/IMD{photo_id}_Dermoscopic_Image/IMD{photo_id}.bmp"  
output_path = "segmenataion.bmp"  


with open(file_path, "rb") as file:  
    files = {
        "file": (file_path, file, "image/bmp"),
    } 
    body = {
        "nn_type": "unet",
    }
    response = requests.post(url, files=files, data=body)  
  
fp = io.BytesIO(response.content)

with fp:
    img = mpimg.imread(fp)
plt.imshow(img)
plt.show()
