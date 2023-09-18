import torch
import cv2 
import easyocr

import cv2 
cam = cv2.VideoCapture(0)

### load model 
model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model = "best.pt", force_reload=True)

def load_model(imagepath = "GreenParking/0000_00532_b.jpg"):

    img = cv2.imread(imagepath)
    # Inference
    results = model(imagepath)
    results.pandas().xyxy[0]    
    image_draw = img.copy()
    
    for i in range(len(results.pandas().xyxy[0])):
        x_min, y_min, x_max, y_max, conf, clas = results.xyxy[0][i].numpy()
        width = x_max - x_min
        height = y_max - y_min

        if clas == 0:
            image_draw = cv2.rectangle(image_draw, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            path_detect = f'results_detect/{imagepath.split("/")[-1]}'
            cv2.imwrite(path_detect, image_draw)

    return imagepath, int(x_min), int(y_min), int(width), int(height)

### crop object
def crop(imagepath, x, y, w, h): 
    image = cv2.imread(imagepath)
    crop_img = image[y:y+h, x:x+w]
    path_crop = f"results_crop/{imagepath.split('/')[-1]}"
    cv2.imwrite(path_crop, crop_img)
    
    return path_crop

### extract value 
def OCR(path): 
    IMAGE_PATH = path
    reader = easyocr.Reader(['en'])
    result = reader.readtext(IMAGE_PATH)
    plate = ' '.join(detect[1] for detect in result)
    print("EXTRACT: ", plate)
    return

def main(): 
    try:
        path, x, y, w, h = load_model()
        croppath = crop(path, x, y, w, h)
        OCR(croppath)
    except: 
        print("No detected plate")
        pass

if __name__ == '__main__': 
    main()