  
from facenet_pytorch import MTCNN
import cv2
from PIL import Image,ImageDraw
import numpy as np
import io
import base64
def main(data):
    decoded_data=base64.b64decode(data)
    np_data=np.fromstring(decoded_data,np.uint8)
    img=cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    
    mtcnn=MTCNN(keep_all=True)
    #convert gray image
    frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    frame=Image.fromarray(frame)

    boxes,probs,points=mtcnn.detect(frame,landmarks=True)
    draw=ImageDraw.Draw(frame)
    try:
        for i,(box,point) in enumerate(zip(boxes,points)):
            draw.rectangle(box.tolist(),width=2)
    except:
        pass
    
    #convert image to PIL image
    pil_im=frame
    
    #convert image to Byte
    buff=io.BytesIO()
    pil_im.save(buff,format="PNG")
    #conver it again to base64
    img_str=base64.b64encode(buff.getvalue())
    return ""+str(img_str,'utf-8')