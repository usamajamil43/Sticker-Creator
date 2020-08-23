from django.shortcuts import render
import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import splprep, splev
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
import numpy

# Create your views here.
def home(request):
    return render(request, 'index.html')


def upload(request):
    
    if request.method == 'POST':
        
        uploaded_file = request.FILES['image_file']
        filename = uploaded_file.name
        fs = FileSystemStorage()
        fs.save(filename, uploaded_file)
        INVTRANS=False
        BorderValue=10
        colorCheck=None

        

        #-- Read image -----------------------------------------------------------------------
        img = cv2.imread('./media/'+filename)
        #img = cv2.resize(img, (600,600))
        img1=img.copy()
        img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        shape=img.shape
        
        count=0
        if(img2[0,0]>=127):
            count+=1;
        if(img2[shape[0]-1,0]>=127):
            count+=1;
        if(img2[0,shape[1]-1]>=127):
            count+=1;
        if(img2[shape[0]-1,shape[1]-1]>=127):
            count+=1;
        if count>=3:
            colorCheck=img2[0,0]
            INVTRANS=True
        

  
    
        if INVTRANS:
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img2[i,j]>=colorCheck-15 and img2[i,j]<=colorCheck+15:
                        for k in range(3):
                            img[i,j,k]=0
                    else:
                        for k in range(3):
                            img[i,j,k]=255
    #                img[i,j,0]=abs(255-img[i,j,0])
    #                img[i,j,1]=abs(255-img[i,j,1])
    #                img[i,j,2]=abs(255-img[i,j,2])


        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        
# blur threshold image
        gray = cv2.GaussianBlur(thresh, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)
        gray = skimage.exposure.rescale_intensity(gray, in_range=(0,127), out_range=(0,255)).astype(np.uint8)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]


        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=3)
    
   
        mask = np.zeros_like(gray)  
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#    for i in range(len(contours)):
#        check=True
#        for j in range(len(contours)):
#            if i!=j and contours[i].any in contours[j]:
#                check=False
#        if check:  
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=-1)
    

        mask_stack = np.dstack([mask]*3)   
        mask_u8 = np.array(mask,np.uint8)
        back = np.zeros(mask.shape,np.uint8)
        back[mask_u8 == 0] = 255
        
        border = np.zeros(thresh.shape)    
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(border, contours, -1, (255, 255, 255), BorderValue)
    
      
            
    
   

        masked = mask_stack * img1  # Blend  
        masked = (masked * 255).astype('uint8')

        masked[:,:,0][back == 255] = 192
        masked[:,:,1][back == 255] = 192
        masked[:,:,2][back == 255] = 192
        
        borcolor=255
        if not INVTRANS:
          borcolor=0 
        for i in range(border.shape[0]):
            for j in range(border.shape[1]):
                if border[i][j]==255:
                    masked[i][j][0]=borcolor
                    masked[i][j][1]=borcolor
                    masked[i][j][2]=borcolor


        maskedRe=cv2.resize(masked,(512,512))
        
        cv2.imwrite('img.jpg', maskedRe)
            #cv2.imshow(<image>)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()          
	# Save
    try:
        with open('./media/img.jpg', "rb") as f:
            return HttpResponse(f.read(), content_type="image/jpeg")
    except IOError:
        red = Image.new('RGBA', (1, 1), (255,0,0,0))
        response = HttpResponse(content_type="image/jpeg")
        red.save(response, "JPEG")
        return response
