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
        BLUR = 5
        CANNY_THRESH_1 = 10
        CANNY_THRESH_2 = 100
        MASK_DILATE_ITER = 10
        MASK_ERODE_ITER = 20
        MASK_COLOR = (220,220,220) # In BGR format
        print('./media/'+filename,"arbaz")

        #-- Read image -----------------------------------------------------------------------
        img = cv2.imread('./media/'+filename)
        img = cv2.resize(img, (600,600))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


        #-- Edge detection -------------------------------------------------------------------
        edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
        edges = cv2.dilate(edges, None)
        # edges = cv2.erode(edges, None)

        # edges = cv2.dilate(edges, None, iterations=2)

        #-- Find contours in edges, sort by area ---------------------------------------------
        
        contour_info = []
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Previously, for a previous version of cv2, this line was: 
        #  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Thanks to notes from commenters, I've updated the code but left this note




        for c in contours:
            contour_info.append((
              c,
              cv2.isContourConvex(c),
              cv2.contourArea(c),
          ))
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        
        
        max_contour = contour_info[0]
        mask = np.zeros(edges.shape)
        
        for c in contour_info:
            cv2.fillConvexPoly(mask, c[0], (255))
        # edges = cv2.dilate(edges, None, iterations=2)
        
        
        mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
        #back=edges==0
        #mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
        #mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
        mask_stack = np.dstack([mask]*3)
        mask_u8 = np.array(mask,np.uint8)
        back = np.zeros(mask.shape,np.uint8)
        back[mask_u8 == 0] = 255
        # Create 3-channel alpha mask
        #-- Blend masked img into MASK_COLOR background --------------------------------------
        #mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
        #img         = img.astype('float32') / 255.0                 #  for easy blending
    
                         # Convert back to 8-bit 

        #back=edges==0
        border = cv2.Canny(mask_u8, CANNY_THRESH_1, CANNY_THRESH_2)
        border = cv2.dilate(border, None, iterations=3)
         


        masked = (mask_stack * img)  # Blend
        masked = (masked * 255).astype('uint8')
        
        masked[:,:,0][back == 255] = 230
        masked[:,:,1][back == 255] = 230
        masked[:,:,2][back == 255] = 230
        
        
        
        
        cv2.imwrite('./media/img.jpg',masked)
        #cv2.imshow('img.jpg',masked)
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
