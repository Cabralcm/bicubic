import math
import numpy as np
from scipy import misc
import cv2
import time, sys, os

####################################################################################################
# Bicubic 2D Image Interpolation
#
# Includes two methods of Bicubic Image Interpolation
# (1) Brute Force Interpolation  - Where each pixel is computed by solving a bicubic equation
# (2) Efficient Interpolation - Where column of pixels is first computed by solving a cubic equation
# and then the row pixels are computed by solving a cubic equation
#
####################################################################################################
class Bicubic:

    def __init__(self, image_file, scale = 2, pad = True, bicubic = True, show = False, outputDir = "./"):
        self.image = cv2.imread(image_file)
        self.height, self.width, self.channels = self.image.shape
        self.image = np.asarray(self.image)

        image_name = image_file.split("/")[-1]
        self.oFile = f"{outputDir}bicubic_{image_name}"

        self.pad = pad
        self.scale = scale
        self.h_scaled = 0
        self.w_scaled = 0
        self.show = show
        #Execute Bicubic Algorithm
        if (bicubic):
            self.bicubic()

    #Python Image Library (Pillow) pixel channel look-up table
    #Reference:
    def mode_lookup(self, mode):
        mode_dict = {"1": 1,
                    "L": 1,
                    "RGB": 3,
                    "RGBA": 4,
                    "CMYK": 4,
                    "YCbCr": 3,
                    "LAB": 3,
                    "HSV": 3,
                    "I": 1,
                    "F": 1}
        if mode in mode_dict:
            return mode_dict[mode] 
        else:
            return 1 #Default single channel

    def cubic_v1(self,array, x_pos):
        return (array[1] + 0.5*x_pos*(array[2] - array[0] + x_pos*( 2*array[0] - 5.0*array[1] + 4.0*array[2] -array[3]))) 

    def cubic_v2(array, val):
        x3 = ( (-1/2)*array[0] +(3/2)*array[1] -(3/2)*array[2] + (1/2)*array[3] )*(x**3) 
        x2 = ( array[0] -(5/2)*array[1] + 2*array[2] -(1/2)*array[3] ) * (x**2)
        x1 = ( (-1/2)*array[0] + (1/2)*array[2] ) * (x)
        x0 = array[1] 
        return x3 + x2 + x1 + x0

    def cubic(self, x_pos, array_0, array_1, array_2, array_3):
        return (-0.5*array_0 + 3/2*array_1 - 3/2*array_2 + 1/2*array_3)*(x_pos**3) + (array_0 -5/2*array_1 + 2*array_2 - (1/2)*array_3)*(x_pos**2) + (-1/2*array_0 + 2/3*array_0)*x_pos + array_1

    def bicubic(self):
        #Start timer
        start = time.time()

        #Create Scaled Image
        self.h_scaled, self.w_scaled =self.height*self.scale, self.width*self.scale
        scaled_image = np.zeros((self.h_scaled, self.w_scaled, self.channels))
        
        #Copy original image into larger scaled image
        scaled_image[0:self.h_scaled:2, 0:self.w_scaled:2, 0:self.channels] = self.image[:, :, 0:self.channels]
        scaled_image = self.pad_image(scaled_image, self.h_scaled, self.w_scaled, self.channels)
        
        ratio = 1/self.scale

        odd_height_array = range(1,self.h_scaled,2)
        odd_width_array = range(1,self.w_scaled,2)
        even_height_array = range(0, self.h_scaled,2)
        width_array= range(self.w_scaled)

        ###First Iteration
        row = np.arange(0,self.h_scaled,2)
        for col in range(1,self.w_scaled,2):
            scaled_image[row,col, 0] = (-1/16)*scaled_image[row,col-3,0] + (9/16)*scaled_image[row,col-1,0] + (9/16)*scaled_image[row,col+1,0] + (-1/16)*scaled_image[row,col-3,0]
            scaled_image[row,col, 1] = (-1/16)*scaled_image[row,col-3,1] + (9/16)*scaled_image[row,col-1,1] + (9/16)*scaled_image[row,col+1,1] + (-1/16)*scaled_image[row,col-3,1]
            scaled_image[row,col, 2] = (-1/16)*scaled_image[row,col-3,2] + (9/16)*scaled_image[row,col-1,2] + (9/16)*scaled_image[row,col+1,2] + (-1/16)*scaled_image[row,col-3,2]

        # ###Second Iteration
        row = np.arange(1,self.h_scaled,2)
        for col in range(self.w_scaled):
            scaled_image[row,col, 0] = (-1/16)*scaled_image[row-3,col,0] + (9/16)*scaled_image[row-1,col,0] + (9/16)*scaled_image[row+1,col,0] + (-1/16)*scaled_image[row-3,col,0]
            scaled_image[row,col, 1] = (-1/16)*scaled_image[row-3,col,1] + (9/16)*scaled_image[row-1,col,1] + (9/16)*scaled_image[row+1,col,1] + (-1/16)*scaled_image[row-3,col,1]
            scaled_image[row,col, 2] = (-1/16)*scaled_image[row-3,col,2] + (9/16)*scaled_image[row-1,col,2] + (9/16)*scaled_image[row+1,col,2] + (-1/16)*scaled_image[row-3,col,2]

        #Remove the padded aspect of the image
        scaled_image = scaled_image[:self.h_scaled, :self.w_scaled, 0:self.channels]
        
        #Bicubic algorithm finished, record end time 
        end = time.time() - start
        print(f"{self.oFile}")
        runtime_s = f"Runtime in seconds: {end}" 
        runtime_m = f"Runtime in minutes: {end/60}"
        print(runtime_s)
        print(runtime_m)
        
        #Write Output Image to File
        cv2.imwrite(f"{self.oFile}", scaled_image)
        
        if (self.show):
            #Read in image to preview
            img = cv2.imread(f"{self.oFile}")
            cv2.imshow(f"{self.oFile}",img)
            cv2.waitKey()
        
        return scaled_image

    #Ensure RGB value doesn't go outside bounds of 0 and 255
    def setRGBValue(self,value):
        if value <= 0:
            return 0
        elif value >= 255:
            return 255
        else:
            return value

    def getScaledBounds(self, np_array, row, col, channel):
        if(col >= self.w_scaled):
            col = self.w_scaled - 1
        if(col <0):
            col = 0
        if(row >= self.h_scaled):
            row = self.h_scaled - 1
        if(row <0):
            row = 0
        return np_array[row, col, channel]

    def getBounds(self, np_array, row, col, channel):
        if(col >= self.width):
            col = self.width - 1
        if(col <0):
            col = 0
        if(row >= self.height):
            row = self.height - 1
        if(row <0):
            row = 0
        return np_array[int(row), int(col), int(channel)]
        #return np_array.item(row, col, channel)
    
    # Pad Image
    #Extend existing pixels around the border of the original image
    #to the extended borders on the padded image
    def pad_image(self,image, height, width, channels, pad_scale=8):
        if channels == "RGB":
            channels = 3
        ##RGB image is if channels == 3.
        lower = pad_scale //2
        padded_image = np.zeros((height+pad_scale, width + pad_scale, channels))  #Create larger empty array. 
        padded_image[lower:height + lower, lower:width+lower, 0:channels] = image ## Put original image into larger empty array.
        #Left Rectangle
        padded_image[lower:height+lower , 0:lower , 0:channels] = image[ 0:height , 0:1 , 0:channels]
        #Bottom Rectangle
        padded_image[height+lower:height+pad_scale, lower:width+lower, 0:channels] = image[height-1:height , 0:width , 0:channels]
        #Right Rectangle
        padded_image[lower:height+lower, width+lower:width+pad_scale, 0:channels] = image[ 0:height , width-1:width , 0:channels]
        #Top Rectangle
        padded_image[0:lower, lower:width+lower, 0:channels] = image[0:1, 0:width, 0:channels]
        #Top Left Corner
        padded_image[0:lower, 0:lower, 0:channels] = image[0:1, 0:1, 0:channels]
        #Top Right Corner
        padded_image[0:lower, width+lower:width+pad_scale, 0:channels] = image[0:1, width-1:width, 0:channels]
        #Bottom Left Corner
        padded_image[height+lower:height+pad_scale, 0:lower, 0:channels] = image[height-1:height, 0:1, 0:channels]
        #Bottom Right Corner
        padded_image[height+lower:height+pad_scale, width+lower:width+pad_scale, 0:channels] = image[0:1, width-1:width, 0:channels]
        return padded_image

####################################################################################################
### END of Bicubic Class ###
####################################################################################################

####################################################################################################
# Bicubic Brute Force Method
# Brute Force method of determining Bicubic Interpolation
####################################################################################################
class Bicubic_Naive:

    def __init__(self, image_file, scale = 2, pad = False, outputDir="./"):
        self.image = cv2.imread(image_file)
        self.height, self.width, self.channels = self.image.shape
        self.image = np.asarray(self.image)

        image_name = image_file.split("/")[-1]
        self.oFile = f"{outputDir}bicubic_{image_name}"
        
        self.h_scaled = 0
        self.w_scaled = 0
        self.a00 = self.a01 = self.a02 = self.a03 = 0
        self.a10 = self.a11 = self.a12 = self.a13 = 0
        self.a20 = self.a21 = self.a22 = self.a23 = 0
        self.a30 = self.a31 = self.a32 = self.a33 = 0
        self.scale = scale
        #Execute Algorithm
        self.bicubic()    

    # https://github.com/yunabe/codelab/blob/master/misc/terminal_progressbar/progress.py
    def get_progressbar_str(self,progress):
        END = 170
        MAX_LEN = 30
        BAR_LEN = int(MAX_LEN * progress)
        return ('Progress:[' + '=' * BAR_LEN +
                ('>' if BAR_LEN < MAX_LEN else '') +
                ' ' * (MAX_LEN - BAR_LEN) +
                '] %.1f%%' % (progress * 100.))

    def bicubic(self):
        #Start timer
        start = time.time()

        #Create Scaled Image
        self.h_scaled, self.w_scaled =self.height*self.scale, self.width*self.scale
        scaled_image = np.zeros((self.h_scaled, self.w_scaled, self.channels))

        ratio = 1/self.scale

        cubic_array = [[0,0,0,0],
                       [0,0,0,0],
                       [0,0,0,0],
                       [0,0,0,0]]
        inc = 0
        for row in range(self.h_scaled):
            orig_row = row * ratio #Update Row
            inter_row = math.floor(orig_row)
            y = orig_row - inter_row
            y = int(y)
            for col in range(self.w_scaled): #Update Column
                orig_col = col * ratio
                inter_col = math.floor(orig_col)
                x = orig_col - inter_col
                x = int(x)
                for channel in range(self.channels): #RGB channels
                    channel = int(channel)
                    cubic_array[0][0] = self.getBounds(self.image, inter_row-1, inter_col-1, channel)
                    cubic_array[0][1] = self.getBounds(self.image, inter_row-1, inter_col , channel)
                    cubic_array[0][2] = self.getBounds(self.image, inter_row-1, inter_col+1, channel)
                    cubic_array[0][3] = self.getBounds(self.image, inter_row-1, inter_col+2, channel)

                    cubic_array[1][0] = self.getBounds(self.image, inter_row, inter_col-1, channel)
                    cubic_array[1][1] = self.getBounds(self.image, inter_row, inter_col , channel)
                    cubic_array[1][2] = self.getBounds(self.image, inter_row, inter_col+1, channel)
                    cubic_array[1][3] = self.getBounds(self.image, inter_row, inter_col+2, channel)

                    cubic_array[2][0] = self.getBounds(self.image, inter_row+1, inter_col-1, channel)
                    cubic_array[2][1] = self.getBounds(self.image, inter_row+1, inter_col , channel)
                    cubic_array[2][2] = self.getBounds(self.image, inter_row+1, inter_col+1, channel)
                    cubic_array[2][3] = self.getBounds(self.image, inter_row+1, inter_col+2, channel)
 
                    cubic_array[3][0] = self.getBounds(self.image, inter_row+2, inter_col-1, channel)
                    cubic_array[3][1] = self.getBounds(self.image, inter_row+2, inter_col , channel)
                    cubic_array[3][2] = self.getBounds(self.image, inter_row+2, inter_col+1, channel)
                    cubic_array[3][3] = self.getBounds(self.image, inter_row+2, inter_col+2, channel)

                    self.bicubicCoefficients(cubic_array)
                    scaled_image[row,col,channel] = self.getBicubicValue(y, x)
                
                    # Print progress
                    inc += 1
                    sys.stderr.write('\r\033[K' + self.get_progressbar_str(inc/(self.channels * self.h_scaled * self.w_scaled)))
                    sys.stderr.flush()
        
        #Bicubic algorithm finished, record end time 
        end = time.time() - start
        print(f"{self.oFile}")
        runtime_s = f"Runtime in seconds: {end}" 
        runtime_m = f"Runtime in minutes: {end/60}"
        print(runtime_s)
        print(runtime_m)
        
        #Write Output Image to File
        cv2.imwrite(f"{self.oFile}", scaled_image)
        
        if (self.show):
            #Read in image to preview
            img = cv2.imread(f"{self.oFile}")
            cv2.imshow(f"{self.oFile}",img)
            cv2.waitKey()
        return

    def getBounds(self, np_array, row, col, channel):
        if(col >= self.width):
            col = self.width - 1
        if(col <0):
            col = 0
        if(row >= self.height):
            row = self.height - 1
        if(row <0):
            row = 0
        return np_array[row, col, channel]

    #Ensure RGB value doesn't go outside bounds of 0 and 255
    def setRGBValue(self,value):
        if value <= 0:
            return 0
        elif value >= 255:
            return 255
        else:
            return value

    def bicubicCoefficients(self, array):
        self.a00 = array[1][1]
        self.a01 = -0.5*array[1][0] + 0.5*array[1][2]
        self.a02 = array[1][0] - 2.5*array[1][1] + 2*array[1][2] - 0.5*array[1][3]
        self.a03 = -0.5*array[1][0] + 1.5*array[1][1] - 1.5*array[1][2] + 0.5*array[1][3]
        self.a10 = -0.5*array[0][1] + 0.5*array[2][1]
        self.a11 = 0.25*array[0][0] - 0.25*array[0][2] - 0.25*array[2][0] + 0.25*array[2][2]
        self.a12 = -.5*array[0][0] + 1.25*array[0][1] - array[0][2] + .25*array[0][3] + .5*array[2][0] - 1.25*array[2][1] + array[2][2] -.25*array[2][3]
        self.a13 = .25*array[0][0] - .75*array[0][1] + .75*array[0][2] - .25*array[0][3] - .25*array[2][0] + .75*array[2][1] - .75*array[2][2] + .25*array[2][3]
        self.a20 = array[0][1] - 2.5*array[1][1] + 2*array[2][1] - .5*array[3][1]
        self.a21 = -.5*array[0][0] + .5*array[0][2] + 1.25*array[1][0] - 1.25*array[1][2] - array[2][0] + array[2][2] + .25*array[3][0] - .25*array[3][2]
        self.a22 = array[0][0] - 2.5*array[0][1] + 2*array[0][2] - .5*array[0][3] - 2.5*array[1][0] + 6.25*array[1][1] - 5*array[1][2] + 1.25*array[1][3] + 2*array[2][0] - 5*array[2][1] + 4*array[2][2] - array[2][3] - .5*array[3][0] + 1.25*array[3][1] - array[3][2] + .25*array[3][3]
        self.a23 = -.5*array[0][0] + 1.5*array[0][1] - 1.5*array[0][2] + .5*array[0][3] + 1.25*array[1][0] - 3.75*array[1][1] + 3.75*array[1][2] - 1.25*array[1][3] - array[2][0] + 3*array[2][1] - 3*array[2][2] + array[2][3] + .25*array[3][0] - .75*array[3][1] + .75*array[3][2] - .25*array[3][3]
        self.a30 = -.5*array[0][1] + 1.5*array[1][1] - 1.5*array[2][1] + .5*array[3][1]
        self.a31 = .25*array[0][0] - .25*array[0][2] - .75*array[1][0] + .75*array[1][2] + .75*array[2][0] - .75*array[2][2] - .25*array[3][0] + .25*array[3][2]
        self.a32 = -.5*array[0][0] + 1.25*array[0][1] - array[0][2] + .25*array[0][3] + 1.5*array[1][0] - 3.75*array[1][1] + 3*array[1][2] - .75*array[1][3] - 1.5*array[2][0] + 3.75*array[2][1] - 3*array[2][2] + .75*array[2][3] + .5*array[3][0] - 1.25*array[3][1] + array[3][2] - .25*array[3][3]
        self.a33 = .25*array[0][0] - .75*array[0][1] + .75*array[0][2] - .25*array[0][3] - .75*array[1][0] + 2.25*array[1][1] - 2.25*array[1][2] + .75*array[1][3] + .75*array[2][0] - 2.25*array[2][1] + 2.25*array[2][2] - .75*array[2][3] - .25*array[3][0] + .75*array[3][1] - .75*array[3][2] + .25*array[3][3]
        return 

    def getBicubicValue(self,x_pos, y_pos):
        x2 = x_pos * x_pos
        x3 = x2 * x_pos
        y2 = y_pos * y_pos
        y3 = y2 * y_pos 
        return ((self.a00 + self.a01 * y_pos + self.a02 * y2 + self.a03 * y3) + 
                (self.a10 + self.a11 * y_pos + self.a12 * y2 + self.a13 * y3) * x_pos +
                (self.a20 + self.a21 * y_pos + self.a22 * y2 + self.a23 * y3) * x2 +
                (self.a30 + self.a31 * y_pos + self.a32 * y2 + self.a33 * y3) * x3 )

    # Pad Image
    #Extend existing pixels around the border of the original image
    #to the extended borders on the padded image
    def pad_image(self,image, height, width, channels):
        if channels == "RGB":
            channels = 3
        ##RGB image is if channels == 3.
        padded_image = np.zeros((height+4, width + 4, channels))  #Create larger empty array. 
        padded_image[2:height + 2, 2:width+2, 0:channels] = image ## Put original image into larger empty array.
        #Left Rectangle
        padded_image[2:height+2 , 0:2 , 0:channels] = image[ 0:height , 0:1 , 0:channels]
        #Bottom Rectangle
        padded_image[height+2:height+4, 2:width+2, 0:channels] = image[height-1:height , 0:width , 0:channels]
        #Right Rectangle
        padded_image[2:height+2, width+2:width+4, 0:channels] = image[ 0:height , width-1:width , 0:channels]
        #Top Rectangle
        padded_image[0:2, 2:width+2, 0:channels] = image[0:1, 0:width, 0:channels]
        #Top Left Corner
        padded_image[0:2, 0:2, 0:channels] = image[0:1, 0:1, 0:channels]
        #Top Right Corner
        padded_image[0:2, width+2:width+4, 0:channels] = image[0:1, width-1:width, 0:channels]
        #Bottom Left Corner
        padded_image[height+2:height+4, 0:2, 0:channels] = image[height-1:height, 0:1, 0:channels]
        #Bottom Right Corner
        padded_image[height+2:height+4, width+2:width+4, 0:channels] = image[0:1, width-1:width, 0:channels]
        return padded_imageg83

####################################################################################################
### END of Bicubic_Naive Class                                                                   ###
####################################################################################################


####################################################################################################
#Compare above Bicubic Algorithm with OpenCV Bicubic Algorithm
#Compute Mean Squared Error for Image
####################################################################################################

def mse_direct_compare(image, show=False, outputDir="./"):
    image_name = image.split("/")[-1]
    oFile = f"{outputDir}OpenCV2_bicubic_{image_name}"
    img = cv2.imread(image)
    cv2_scaled_image = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(oFile, cv2_scaled_image)
    
    scaled_image = Bicubic(image, show = False, bicubic= False).bicubic()
    mse_error = np.sum( (scaled_image.astype("float") - cv2_scaled_image.astype("float") )**2  )
    #Shape[0] is the Height, Shape[1] is the Width
    mse_error /= float(scaled_image.shape[0] * scaled_image.shape[1])
    #Read in image to preview
    if(show):
        img = cv2.imread(oFile)
        cv2.imshow(oFile,img)
        cv2.waitKey()

    return mse_error

def mse(input_image):
    image = cv2.imread(input_image)
    small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC )
    small_file = f"Small_{input_image}"
    cv2.imwrite(small_file, small)
    scaled_image = Bicubic(small_file, show = False, bicubic= False).bicubic()
    cv2_scaled_image = cv2.resize(small, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    
    mse_error = np.sum( (image.astype("float") - cv2_scaled_image.astype("float") )**2  )
    mse_error /= float(scaled_image.shape[0] * scaled_image.shape[1])
    print(f"CV2 vs Original {mse_error}")

    mse_error = np.sum( (image.astype("float") - scaled_image.astype("float") )**2  )
    mse_error /= float(scaled_image.shape[0] * scaled_image.shape[1])
    print(f"Custom Bicubic vs Original {mse_error}")

    mse_error = np.sum( (scaled_image.astype("float") - cv2_scaled_image.astype("float") )**2  )
    mse_error /= float(scaled_image.shape[0] * scaled_image.shape[1])
    print(f"Custom Bicubic vs CV2 {mse_error}")
    return

####################################################################################################
# Helper Functions
####################################################################################################

def getFiles(directory, show = False):
    fullPathFiles = []
    for dirpath, dirnames, files in os.walk(directory):
        for file_name in files:
            fullPath = f"{dirpath}{file_name}"
            if show:
                print(fullPath)
            fullPathFiles.append(fullPath)
    return fullPathFiles


####################################################################################################
# Code Execution for Bicubic Algorithm
####################################################################################################
def main():

    CWD = os.getcwd()
    IMAGES = getFiles("./Images/")
    OUTPUT_DIR = f"{CWD}/OutputImages/"
    
    #Choose which function to run
    #True = Run
    #False = Don't Run
    FAST = True
    NAIVE = False

    # Choose to show the image in a separate window (via Python OpenCV)
    SHOW_IMAGE = False
    
    mse_flag = False
    mse_direct_compare_flag = False

    print("Starting Bicubic Interpolation.")

    #Run Efficient Algorithm
    if(FAST):
        for file in IMAGES:
            Bicubic(file, show=SHOW_IMAGE, outputDir= OUTPUT_DIR)
    #Run Brute Force Algorithm (WARNING: VERY SLOW, 8-10mins per picture)
    if(NAIVE):
        for file in IMAGES:
            Bicubic_Naive(file, show=SHOW_IMAGE, outputDir = OUTPUT_DIR)
    
    #Compare CV2 Bicubic with this Project's Bicubic on a shrunk image, scaled up to normal size
    if(mse_flag):
        for file in IMAGES:
            mse(file)

    #Compare CV2 Bicubic with this Project's Bicubic on a normal image, scaled up 2x in size
    if(mse_direct_compare_flag):
        for file in IMAGES:
            mse_direct_compare(file, outputDir = OUTPUT_DIR)
    
    print("Done.")
    return

#Run the main function
if __name__ == "__main__":
    main()
