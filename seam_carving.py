# CS6475 - Fall 2022

import numpy as np
import scipy as sp
import cv2 as cv
import scipy.signal                     # option for a 2D convolution library
from matplotlib import pyplot as plt    # for optional plots

import copy


""" Project 2: Seam Carving

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

References
----------
See the following papers, available in Canvas under Files:

(1) "Seam Carving for Content-Aware Image Resizing"
    Avidan and Shamir, 2007
    
(2) "Improved Seam Carving for Video Retargeting"
    Rubinstein, Shamir and Avidan, 2008
    
FORBIDDEN:
    1. OpenCV functions SeamFinder, GraphCut, and CostFunction are
    forbidden, along with any similar functions that may be in the class environment.
    2. Numeric Metrics functions of error or similarity; e.g. SSIM, RMSE. 
    These must be coded from their mathematical equations. Write your own versions. 

GENERAL RULES:
    1. ALL CODE USED IN THIS ASSIGNMENT to generate images, red-seam images,
    differential images, and comparison metrics must be included in this file.
    2. YOU MAY NOT USE any library function that essentially completes
    seam carving or metric calculations for you. If you have questions on this,
    ask on Ed. **Usage may lead to zero scores for the project and review
    for an honor code violation.**
    3. DO NOT CHANGE the format of this file. You may NOT change existing function
    signatures, including the given named parameters with defaults.
    4. YOU MAY ADD FUNCTIONS to this file, however it is your responsibility
    to ensure that the autograder accepts your submission.
    5. DO NOT IMPORT any additional libraries other than the ones imported above.
    You should be able to complete the assignment with the given libraries.
    6. DO NOT INCLUDE code that prints, saves, shows, displays, or writes the
    images, or your results. If you have code in the functions that 
    does any of these operations, comment it out before autograder runs.
    7. YOU ARE RESPONSIBLE for ensuring that your code executes properly.
    This file has only been tested in the course environment. Any changes you make
    outside the areas annotated for student code must not impact the autograder
    system or your performance.
    
FUNCTIONS:
    returnYourName
    IMAGE GENERATION:
        beach_back_removal
        dolphin_back_insert with redSeams=True and False
        dolphin_back_double_insert
        bench_back_removal with redSeams=True and False
        bench_for_removal with redSeams=True and False
        car_back_insert
        car_for_insert
    COMPARISON METRICS:
        difference_image
        numerical_comparison
"""


def returnYourName():
    # This function returns your name as shown on your Gradescope Account.
    return 'Richard Walsh Praetorius'


def image_gradient(image, EoH = False, kernel=11, borderType=cv.BORDER_CONSTANT):
    
    #input: openCV image (y,x,3)
    #       kernel border size
    
    #output:flattened array of gradient values
    #       flattened array of histogram-normalized gradient values
    
    
    #initialize    
    y,x,z = image.shape
    
 
    # attempt to eliminate smaller detail gradients, retaining the major information
    image = cv.medianBlur(image,ksize=3)
    
    #take gradients
    g_image_x = cv.Sobel(image,cv.CV_64F,1,0,ksize=3)
    g_image_y = cv.Sobel(image,cv.CV_64F,0,1,ksize=3)
      
    g_image = np.sum(np.absolute(g_image_x) + np.absolute(g_image_y), axis=2)

    #g_image = np.sum((g_image_x**2 + g_image_y**2) ** 0.5, axis = 2) #magnitude?

    # histogram feature
    if EoH:
        
        b=kernel//2
        g_image_hist = np.ones((y,x), dtype=np.float64)    
        
        #get histogram of sections
        for why in range (b,y+b):
            for ecks in range (b,x+b):
                
                data = np.histogram(g_image[why-b:why+b+1, ecks-b:ecks+b+1],8)
                max_loc = np.argmax(data[0])
            
                g_image_hist[why-b,ecks-b] = data[1][max_loc+1] #highest count bin value
        
        return g_image/g_image_hist

    else:

        return g_image



def backward_energy (g_image, orient = 'v'):

    #finds overall cumulative neergy in image
    #inputs gradient image and orientation information
    #returns overall cumulative energy in same shape
       
    wrk_image = cv.copyMakeBorder(g_image,1,1,1,1,cv.BORDER_REPLICATE)
    
    y,x = wrk_image.shape #overall sizes with border

    if orient == 'v': #vertical seams
        
        #fill in cumulative energy
        for why in range(2,y-1):
            for ecks in range(1,x-1):
                wrk_image[why,ecks] += np.amin(wrk_image[why-1,ecks-1:ecks+2])
            wrk_image[why,0] = wrk_image[why,1]
            wrk_image[why,x-1] = wrk_image[why,x-2]
    
    elif orient == 'h': #horizontal seams
        
        for ecks in range(2,x-1):
            for why in range(1,y-1):
                wrk_image[why,ecks] += np.amin(wrk_image[why-1:why+2,ecks-1])
            wrk_image[0,ecks] = wrk_image[1,ecks]
            wrk_image[y-1,ecks] = wrk_image[y-2,ecks]

    else:
        print('bkd_energy orient ERROR')
 
    #resize total cumulative energy function, slice it down to size
    out_img = np.atleast_3d(wrk_image[1:y-1,1:x-1])

    return out_img


def forward_energy (g_image, orient = 'v'):

    #input gradient image (flat), orientation
    #return cumulative forward energy
   
    wrk_image = cv.copyMakeBorder(g_image,1,1,1,1,cv.BORDER_REPLICATE)
   
    cum_energy = copy.copy(wrk_image)

    y,x = wrk_image.shape #overall sizes with border
    
    #fill in cumulative energy
    # P(i,j) is an optional weight
    
    if orient == 'v':
        for why in range(2,y-1):
            for ecks in range(1,x-1):
                    
                r= wrk_image[why-1,ecks]
                t= wrk_image[why,ecks-1]
                u= wrk_image[why,ecks+1]
                
                b= abs(u-t)
                a= b+abs(r-t)
                c= b+abs(r-u)
                
                cum_energy[why,ecks] = min(cum_energy[why-1,ecks-1]+a, cum_energy[why-1,ecks]+b, cum_energy[why-1,ecks+1]+c)

            wrk_image[why,0] = wrk_image[why,1]
            wrk_image[why,x-1] = wrk_image[why,x-2]


    elif orient =='h':
        for ecks in range(2,x-1):
            for why in range(1,y-1):
                    
                r= wrk_image[why,ecks-1]
                t= wrk_image[why-1,ecks]
                u= wrk_image[why+1,ecks]
                
                b= abs(u-t)
                a= b+abs(r-t)
                c= b+abs(r-u)
                
                cum_energy[why,ecks] = min(cum_energy[why-1,ecks-1]+a, cum_energy[why,ecks-1]+b, cum_energy[why+1,ecks-1]+c)
    
    else:
        print('fwd_energy orient ERROR')
    

    #resize total cumulative energy function
    out_img = np.atleast_3d(cum_energy[1:y-1,1:x-1])

    return out_img #float64


def seam_trace(cumult, orient = 'v'):
    
    #Seam trace
    #finds lowest energy seam
    #input energy map, previous seams and other masks
    #output single seam np array as BOOLEAN
    
    cum_energy = cv.copyMakeBorder(cumult,1,1,1,1,cv.BORDER_CONSTANT, None, np.amax(cumult))
    y,x = cum_energy.shape
    seam = np.zeros((y,x), dtype=np.bool_)

    if orient == 'v':

        #find lowest energy seam ends in last row
        seam_end = np.argmin(cum_energy[-2,:])
        seam[-2,seam_end] = True
            
        for why in range(3,y+1): #trace back up through gradient image
            
            w = why*-1
            low_energy = np.argmin(cum_energy[w, seam_end-1:seam_end+2]) - 1
            seam_end += low_energy #returns index of first lowest-x
            seam[w,seam_end] = True

    elif orient == 'h':

        #find lowest energy seam ends in last column
        seam_end = np.argmin(cum_energy[:,-2])
        seam[seam_end,-2] = True
            
        for ecks in range(3,x+1): #trace back up through gradient image
            
            e = ecks*-1
            low_energy = np.argmin(cum_energy[seam_end-1:seam_end+2, e]) - 1
            seam_end += low_energy #returns index of first lowest-y
            seam[seam_end,e] = True

    else:
        print('seam_trace orient ERROR')
    
    seam = seam[1:y-1,1:x-1]

    return seam #bool_



def seam_delete(image, seam, orient = 'v'): 

    #input image with mask of SINGLE seam
    #input list of old seams
    #orientation of seam
    
    #returns an image that is x-1 or y-1 resized, without seam
 
    wrk_image = image.copy()
    y,x,z = wrk_image.shape

    seam_points = np.where(seam)
    seam_point_list=[]
    
    for point in range(len(seam_points[0])):
        seam_point_list.append((seam_points[0][point], seam_points[1][point]))
    

    if orient == "v":
        
        for crd in seam_point_list:
            
            wrk_image[crd[0], crd[1]:x-1, :] = wrk_image[crd[0], crd[1]+1:x, :]
            
        wrk_image = wrk_image[:,:x-1, :]


    elif orient == 'h':

        for crd in seam_point_list:
            
            wrk_image[crd[0]:y-1, crd[1], :] = wrk_image[crd[0]+1:y, crd[1], :]
            
        wrk_image = wrk_image[:y-1,:, :]

    else:
        print('seam_delete orientation error')
        return image

    return wrk_image

    
def seam_add(image, seam, old_seam, orient = 'v'):
    #input the image to be stretched, the seams, and the orientation of the seams
    #add seam to image, average the two next to it
    #automatically detects if seam or image being expanded
    #only does one at a time, not contract then expand. DOESN'T WORK
      
         
    wrk_image = image.copy()
        
    seam_points = np.where(seam)
    seam_point_list=[]
    for point in range(len(seam_points[0])):
        seam_point_list.append((seam_points[0][point], seam_points[1][point])) 
    
    y,x,z = wrk_image.shape

    ys, xs = old_seam.shape
    
    if orient == ("v"):
        
        #create space
        new_col = np.zeros((y,1,z))
        wrk_image = np.concatenate((wrk_image,new_col),axis=1)
        
        new_col1 = np.zeros((y,1), dtype = np.bool_)
        out_seam = np.concatenate((old_seam,new_col1),axis=1)
        
        #process image & masks
        for crd in seam_point_list:
          
            wrk_image[crd[0], crd[1]+1:x+1,:] = wrk_image[crd[0], crd[1]:x, :]
            wrk_image[crd[0], crd[1], :] = (wrk_image[crd[0], crd[1], :]+wrk_image[crd[0], crd[1]-1, :])/2
   
            out_seam[crd[0], crd[1]+1:xs+1] = out_seam[crd[0], crd[1]:xs]
        
        out_seam += seam            

    elif orient == 'h':
        #create space
        new_col = np.zeros((1,x,z))
        wrk_image = np.concatenate((wrk_image,new_col),axis=0)
        
        new_row1 = np.zeros((1,x), dtype = np.bool_)
        out_seam = np.concatenate((old_seam,new_row1),axis=0)
            
        #process image & masks
        for crd in seam_point_list:
            #copy info to correct position
            wrk_image[crd[0]+1:y+1, crd[1], :] = wrk_image[crd[0]:y, crd[1], :]
            
            #overwrite seam area with average surrounding pixels
            wrk_image[crd[0], crd[1], :] = (wrk_image[crd[0], crd[1], :]+wrk_image[crd[0]-1, crd[1], :])/2 
            
            #copy info to correct position
            out_seam[crd[0]+1:y+1, crd[1]] = out_seam[crd[0]:y, crd[1]]
        
        #overwrite current seam info
        out_seam += seam


    else:
        print('seam_add orient error')


    return wrk_image, out_seam


def seam_reverse(image, mask, seams, orient='v'):

    #adds seams after finding seams through deletion
    #mask is all seams put together

    wrk_image = image.copy()
    yo,xo,zo = wrk_image.shape

    wrk_mask = mask.copy()

    if orient == ("v"):
        
        #create space
        new_col = np.zeros((yo,seams+1,zo))
        wrk_image = np.concatenate((wrk_image,new_col),axis=1)
        
        new_col1 = np.zeros((yo,seams+1), dtype = np.bool_)
        wrk_mask = np.concatenate((wrk_mask,new_col1),axis=1)

        ym, xm = wrk_mask.shape
        
        #process image & masks
        for i in range(ym):
            for j in range(1,xm-1):
                if wrk_mask[i,j]:
                    wrk_image[i, j+1:xm,:] = wrk_image[i, j:xm-1, :]
                    wrk_image[i, j, :] = (wrk_image[i, j, :]+wrk_image[i, j-1, :])/2
                    wrk_mask[i, j+1:xm] = wrk_mask[i, j:xm-1]
                    wrk_mask[i,j+1] = False
                    

    out_img = wrk_image[:,:xm-1,:]
    out_mask = wrk_mask[:,:xm-1]
                    
                            
    return out_img, out_mask


def red_seam(image, mask, expand = False):
    '''

    Parameters
    ----------
    image : Mat type
        Original image.
    seam_list : list of numpy arrays
        Same size as image.

    Returns
    -------
    IMage with seams overlaid in red. BGR people!

    '''    

    #create mask from seam deletion through list expansion
    #mask already created for seam insertion
    
    wrk_image=image.copy()
    
    if type(mask) == list:
        
        dummy = np.zeros(mask[-1].shape, dtype = np.bool_)
        
        mask[-1] = np.atleast_3d(mask[-1])
        
        for n in range(len(mask)-1,0,-1):
            
            temp_mask, dummy = seam_add(mask[n], mask[n-1], dummy)
            mask[n-1] = temp_mask + np.atleast_3d(mask[n-1])

        wrk_mask = mask[0][:,:,0]

    else:
        wrk_mask = mask

    
    seam_points = np.where(wrk_mask)
    seam_point_list=[]
    
    for point in range(len(seam_points[0])):
        seam_point_list.append((seam_points[0][point], seam_points[1][point]))
    
    for point in seam_point_list:
        wrk_image[point[0],point[1],:] = [0, 0, 255]
    
    return wrk_image, wrk_mask



# -------------------------------------------------------------------
""" IMAGE GENERATION
    *** ALL IMAGES SUPPLIED OR RETURNED ARE EXPECTED TO BE UINT8 ***
    Parameters and Returns are as follows for all of the removal/insert 
    functions:

    Parameters
    ----------
    image : numpy.ndarray (dtype=uint8)
        Three-channel image of shape (r,c,ch). 
    seams : int
        Integer value of number of vertical seams to be inserted or removed.
        NEVER HARDCODE THE NUMBER OF SEAMS, we check other values in the autograder.
    redSeams : boolean
        Boolean variable; True = produce a red seams image, False = no red seams
        
    Returns
    -------
    numpy.ndarray (dtype=uint8)
        An image of shape (r, c_new, ch) where c_new = new number of columns.
        Make sure you deal with any needed normalization or clipping, so that 
        your image array is complete on return.
"""

def beach_back_removal(image, seams=300, redSeams=False):
    """ Use the backward method of seam carving from the 2007 paper to remove
   the required number of vertical seams in the provided image. Do NOT hard-code the
    number of seams to be removed.
    """
    # for seam removal:
    # create list of masks for red_seam processing
    # red seams over ORIGINAL image

    

    out_img = image.copy()
    old_seam = []
   
    for q in range(seams):
        
        #print(q)
        #if q%10 == 0: #creates new energy gradient - 40 second process
        grad_image = image_gradient(out_img)
        nrg = backward_energy(grad_image)

        seam = seam_trace(nrg)
        old_seam.append(seam)
        out_img = seam_delete(out_img, seam)


        #if q%10 != 0: #slices seam out of energy instead of retaking gradient
        #    nrg = seam_delete(nrg, seam)

    if redSeams:
        out_img, dummy = red_seam(image, old_seam)
    
    return out_img.astype(np.uint8)


def dolphin_back_insert(image, seams=100, redSeams=False):
    """ Similar to Fig 8c and 8d from 2007 paper. Use the backward method of seam carving to 
    insert vertical seams in the image. Do NOT hard-code the number of seams to be inserted.
    
    This function is called twice:  dolphin_back_insert with redSeams = True
                                    dolphin_back_insert without redSeams = False
    """

    # for seam addition
    # red_seam mask processing done in-function
    # 

    out_img = image.copy()
    #y,x,z = image.shape
    #old_seam = np.zeros((y, x-1), dtype = np.bool_) #initial case for expansion
    
    #grad_image = image_gradient(out_img)
    #nrg = backward_energy(grad_image)

    #for q in range(seams):
        
        #print(q)
        #if q%50 == 0: #creates new energy gradient - 40 second process
        #grad_image = image_gradient(out_img)
        #nrg = backward_energy(grad_image)
        

        #seam = seam_trace(nrg)
        #out_img, old_seam = seam_add(out_img, seam, old_seam)
        #out_img = out_img.astype(np.uint8)

        #nrg, dummy = seam_add(nrg, seam, old_seam[:,:-1])

        #if q%50 != 0: #creates seam into energy instead of refinding gradient
        #    nrg, temp = seam_add(nrg, seam, old_seam)

    old_seam = []
   
    for q in range(seams):
        
        #print(q)
        #if q%10 == 0: #creates new energy gradient - 40 second process
        grad_image = image_gradient(out_img)
        nrg = backward_energy(grad_image)

        seam = seam_trace(nrg)
        old_seam.append(seam)
        out_img = seam_delete(out_img, seam)

    dummy, wrk_mask = red_seam(image, old_seam)
    out_img, mask = seam_reverse(image, wrk_mask, seams)

    if redSeams:
        out_img, dummy = red_seam(out_img, mask)
    
    return out_img.astype(np.uint8)


def dolphin_back_double_insert(image, seams=100, redSeams=False):
    """ Similar to Fig 8f from 2007 paper. Use the backward method of seam carving to 
    insert vertical seams by performing two insertions, each of size seams, in the image.  
    i.e. insert seams, then insert seams again.  
    Do NOT hard-code the number of seams to be inserted.
    """
    
    out_img = image.copy()

    out_img = dolphin_back_insert(image,seams=seams)
    out_img = dolphin_back_insert(out_img,seams=seams)

    return out_img




def bench_back_removal(image, seams=225, redSeams=False):
    """ Similar to Fig 8 from 2008 paper. Use the backward method of seam carving to 
    remove vertical seams in the image. Do NOT hard-code the number of seams to be removed.
    
    This function is called twice:  bench_back_removal, redSeams = True
                                    bench_back_removal, redSeams = False
    """
    

    EoH = False
    
    
    out_img = image.copy()
    old_seam = []
   
    for q in range(seams):
        
        #print(q)
        if EoH and q%25 == 0: #creates new energy gradient - 40 second process
            grad_image = image_gradient(out_img, EoH = True)
            nrg = backward_energy(grad_image)
        else: 
            grad_image = image_gradient(out_img)
            nrg = backward_energy(grad_image)


        seam = seam_trace(nrg)
        old_seam.append(seam)
        out_img = seam_delete(out_img, seam)



        if EoH and q%25 != 0: #slices seam out of energy instead of retaking gradient
            nrg = seam_delete(nrg, seam)


    if redSeams:
        out_img, dummy = red_seam(image, old_seam)
    
    return out_img.astype(np.uint8)


def bench_for_removal(image, seams=225, redSeams=False):
    """ Similar to Fig 8 from 2008 paper. Use the forward method of seam carving to 
    remove vertical seams in the image. Do NOT hard-code the number of seams to be removed.
    
    This function is called twice:  bench_for_removal, redSeams = True
                                    bench_for_removal, redSeams = False
  """
    

    EoH = False
    
    out_img = image.copy()
    old_seam = []
   
    for q in range(seams):
        
        if EoH and q%25 == 0: #creates new energy gradient - 40 second process
            grad_image = image_gradient(out_img, EoH = True)
            nrg = forward_energy(grad_image)
        else: 
            grad_image = image_gradient(out_img)
            nrg = forward_energy(grad_image)

        seam = seam_trace(nrg)
        old_seam.append(seam)
        out_img = seam_delete(out_img, seam)


        if EoH and q%25 != 0: #slices seam out of energy instead of retaking gradient
            nrg = seam_delete(nrg, seam)

    if redSeams:
        out_img, dummy = red_seam(image, old_seam)
    
    return out_img.astype(np.uint8)


def car_back_insert(image, seams=170, redSeams=False):
    """ Fig 9 from 2008 paper. Use the backward method of seam carving to insert
    vertical seams in the image. Do NOT hard-code the number of seams to be inserted.
    """
    out_img = image.copy()
    
    old_seam = []
   
    for q in range(seams):
        
        #print(q)
        #if q%10 == 0: #creates new energy gradient - 40 second process
        grad_image = image_gradient(out_img)
        nrg = backward_energy(grad_image)

        seam = seam_trace(nrg)
        old_seam.append(seam)
        out_img = seam_delete(out_img, seam)

    dummy, wrk_mask = red_seam(image, old_seam)
    out_img, mask = seam_reverse(image, wrk_mask, seams)

    if redSeams:
        out_img, dummy = red_seam(out_img, mask)
    
    return out_img.astype(np.uint8)


def car_for_insert(image, seams=170, redSeams=False):
    """ Similar to Fig 9 from 2008 paper. Use the forward method of seam carving to 
    insert vertical seams in the image. Do NOT hard-code the number of seams to be inserted.
    """
    out_img = image.copy()
    
    old_seam = []
   
    for q in range(seams):
        
        #print(q)
        #if q%10 == 0: #creates new energy gradient - 40 second process
        grad_image = image_gradient(out_img)
        nrg = forward_energy(grad_image)

        seam = seam_trace(nrg)
        old_seam.append(seam)
        out_img = seam_delete(out_img, seam)

    dummy, wrk_mask = red_seam(image, old_seam)
    out_img, mask = seam_reverse(image, wrk_mask, seams)

    if redSeams:
        out_img, dummy = red_seam(out_img, mask)
    
    return out_img.astype(np.uint8)
    

# __________________________________________________________________
""" COMPARISON METRICS 
    There are two functions here, one for visual comparison support and one 
    for a quantitative metric. 
"""

def difference_image(result_image, comparison_image):
    """ Take two images and produce a difference image that best visually
    indicates how and where the two images differ in pixel values.
    
    Parameters
    ----------
    result_image, comparison_image : numpy.ndarray (dtype=uint8)
        two BGR images of the same shape (r,c,ch) 
    
    Returns
    -------
    numpy.ndarray (dtype=uint8)
        A BGR image of shape (r, c, ch) representing the differences between two
        images. 
        
    NOTES: MANY ERRORS IN PRODUCING DIFFERENCE IMAGES RELATE TO THESE ISSUES
        1) Do your calculations in floats, so that data is not lost.
        2) Before converting back to uint8, complete any necessary scaling,
           rounding, or clipping. 
    """
    
    org = comparison_image.astype(np.float64)
    gen = result_image.astype(np.float64)
    
    out_img = np.abs(gen-org)

    out_img = np.sum(out_img, axis=2) / 3 #averaged difference into greyscale

    out_img = np.stack((out_img, out_img, out_img), axis = 2)

    return out_img.astype(np.uint8)


def numerical_comparison(result_image, comparison_image):
    """ Take two images and produce one or two single-value metrics that
    numerically best indicate(s) how different or similar two images are.
    Only one metric is required, you may submit two, but no more.
    
    If your metric produces a result indicating a total number of pixels, or values,
    formulate it as a ratio of the total pixels in the image. This supports use
    of your results to evaluate code performance on different images.

    ******************************************************************
    NOTE: You may not use functions that perform the whole function for you.
    Research methods, find an algorithm (equation) and implement it. You may
    use numpy array MATHEMATICAL functions such as abs, sqrt, min, max, dot, .T 
    and others that perform a single operation for you.
    
    FORBIDDEN: Library functions of error or similarity; e.g. SSIM, RMSE, etc.
    Use of these functions may result in zero for the assignment and review 
    for an honor code violation.
    ******************************************************************

    Parameters
    ----------
    result_image, comparison_image : numpy.ndarray (dtype=uint8)
        two BGR images of the same shape (r,c,ch) 

    Returns
    -------
    value(s) : float   
        One or two single_value metric comparisons
        Return a tuple of values if you are using two metrics.
        
    NOTE: you may return only one or two values; choose the best one(s) you tried. 
    """
    
    res = result_image.astype(np.float64)
    com = comparison_image.astype(np.float64)


    diffy = difference_image(result_image, comparison_image).astype(np.float64)

    rssd_num  = (np.sum( (diffy)**2 ))**0.5 #, axis=(0,1,2) 

    rssd_den = (np.sum( (comparison_image)**2 ))**0.5

    rssd = rssd_num / rssd_den
    # from video textures assignment
    
    percent_difference = (np.sum(diffy) / np.sum(comparison_image)) * 100


    #SSIM map
    
    k1 = 0.0255
    k2 = 0.02295

    c1 = (255 * k1) #6.5025
    c2 = (255 * k2) #58.5225
    
    sigma = 1.5
    kernel = (11,11)

    m1 = cv.GaussianBlur(res, kernel, sigma)
    m2 = cv.GaussianBlur(com, kernel, sigma)
    s1 = cv.GaussianBlur(res**2, kernel, sigma)
    s2 = cv.GaussianBlur(com**2, kernel, sigma)
    s3 = cv.GaussianBlur(res*com, kernel, sigma)
    
    m3 = m1*m2
    m1*=m1
    m2*=m2
    s1-=m1
    s2-=m2
    s3-=m3

    ssim = ((2*m3 + c1)*(2*s3 + c2)) / ((m1 + m2 + c1)*(s1 + s2 + c2)) 

    y,x,z = ssim.shape
    #inter = np.sum(ssim, axis = 0)
    #overall = np.sum(inter, axis = 1)

    mssim = np.sum(ssim) / (y*x*z)

    return round(rssd,2), mssim



if __name__ == "__main__":
    
    # You may use this area for code that allows you to test your functions.
    # This section will not be graded, and is optional. 
    
    # Comment out this section when you submit to the autograder to avoid the chance 
    # of wasting time and attempts.

    import time

    beach = cv.imread("images/base/beach.png")
    dolphin = cv.imread("images/base/dolphin.png")
    bench = cv.imread("images/base/bench.png")
    car = cv.imread("images/base/car.png")

    comp_beach = cv.imread("images/comparison/comp_beach_back_rem.png") # `res_beach_back_rem` `diff_beach_back_rem`
    comp_dolphin_red = cv.imread("images/comparison/comp_dolphin_back_ins_red.png") # `res_dolphin_back_ins_red`
    comp_dolphin = cv.imread("images/comparison/comp_dolphin_back_ins.png") # `res_dolphin_back_ins` `diff_dolphin_back_ins`
    comp_dolphin_double = cv.imread("images/comparison/comp_dolphin_back_double.png") # `res_dolphin_back_double` `diff_dolphin_back_double`
    comp_bench_back = cv.imread("images/comparison/comp_bench_back_rem.png") # `res_bench_back_rem` `diff_bench_back_rem`
    comp_bench_back_red = cv.imread("images/comparison/comp_bench_back_rem_red.png") # `res_bench_back_rem_red`
    comp_bench_for = cv.imread("images/comparison/comp_bench_for_rem.png") # `res_bench_for_rem` `diff_bench_for_rem`
    comp_bench_for_red = cv.imread("images/comparison/comp_bench_for_rem_red.png") # `res_bench_for_rem_red`
    comp_car_back = cv.imread("images/comparison/comp_car_back_ins.png") # `res_car_back_ins` `diff_car_back_ins`
    comp_car_for = cv.imread("images/comparison/comp_car_for_ins.png") # `res_car_for_ins` `diff_car_for_ins`

    timer1 = time.time()

    ### BEACH ###
    
    # print('BEACH          ', end=" ")
    # res_beach_back_rem = beach_back_removal(beach)
    # diff_beach_back_rem = difference_image(comp_beach, res_beach_back_rem)
    # cv.imwrite('images/output/res_beach_back_rem.png', res_beach_back_rem)
    # cv.imwrite('images/output/diff_beach_back_rem.png', diff_beach_back_rem)
    
    # beach_timer = time.time()

    # perc, ssim = numerical_comparison(res_beach_back_rem, comp_beach)
    # print('time: ',round((beach_timer-timer1)/60,2) )
    # print('                Percent difference: ', perc, '%')
    # print('                      SSIM average: ', ssim)
    # print('')



    ### DOLPHIN ###
    
    # print('DOLPHIN        ', end=" ")
    # res_dolphin_back_ins_red = dolphin_back_insert(dolphin, redSeams=True)
    # res_dolphin_back_ins = dolphin_back_insert(dolphin)
    # diff_dolphin_back_ins = difference_image(comp_dolphin, res_dolphin_back_ins)
    # cv.imwrite('images/output/res_dolphin_back_ins_red.png', res_dolphin_back_ins_red)
    # cv.imwrite('images/output/res_dolphin_back_ins.png', res_dolphin_back_ins)
    # cv.imwrite('images/output/diff_dolphin_back_ins.png', diff_dolphin_back_ins)
    
    dolphin_timer = time.time()

    # perc, ssim = numerical_comparison(res_dolphin_back_ins, comp_dolphin)
    # print('time: ',round((dolphin_timer - beach_timer)/60,2) )
    # print('                Percent difference: ', perc, '%')
    # print('                      SSIM average: ', ssim)
    # print('')
    

    
    ### DOUBLE DOLPHIN ###
    
    # print('DOUBLE DOLPHIN ', end=" ")
    # res_dolphin_back_double = dolphin_back_double_insert(dolphin)
    # diff_dolphin_back_double = difference_image(comp_dolphin_double, res_dolphin_back_double)
    # cv.imwrite('images/output/res_dolphin_back_double.png', res_dolphin_back_double)
    # cv.imwrite('images/output/diff_dolphin_back_double.png', diff_dolphin_back_double)
    
    double_dolphin_timer = time.time()

    # perc, ssim = numerical_comparison(res_dolphin_back_double, comp_dolphin_double)
    # print('time: ',round((double_dolphin_timer - dolphin_timer)/60,2) )
    # print('                Percent difference: ', perc, '%')
    # print('                      SSIM average: ', ssim)
    # print('')
    

    
    ### BENCH BACKWARD ENERGY ###
    
    # print('BENCH BACKWARD ', end=" ")
    # res_bench_back_rem_red = bench_back_removal(bench, redSeams=True)
    # res_bench_back_rem = bench_back_removal(bench)
    # diff_bench_back_rem = difference_image(comp_bench_back, res_bench_back_rem)
    # cv.imwrite('images/output/res_bench_back_rem_red.png', res_bench_back_rem_red)
    # cv.imwrite('images/output/res_bench_back_rem.png', res_bench_back_rem)
    # cv.imwrite('images/output/diff_bench_back_rem.png', diff_bench_back_rem)
    
    bench_back_timer = time.time()
    
    # perc, ssim = numerical_comparison(res_bench_back_rem, comp_bench_back)
    # print('time: ',round((bench_back_timer - double_dolphin_timer)/60,2) )
    # print('                Percent difference: ', perc, '%')
    # print('                      SSIM average: ', ssim)
    # print('')



    ### BENCH FORWARD ENERGY ###
    
    print('BENCH FORWARD  ', end=" ")
    res_bench_for_rem = bench_for_removal(bench)
    res_bench_for_rem_red = bench_for_removal(bench, redSeams = True)
    diff_bench_for_rem = difference_image(comp_bench_for, res_bench_for_rem)
    cv.imwrite('images/output/res_bench_for_rem.png', res_bench_for_rem)
    cv.imwrite('images/output/res_bench_for_rem_red.png', res_bench_for_rem_red)
    cv.imwrite('images/output/diff_bench_for_rem.png', diff_bench_for_rem)

    bench_fwd_timer = time.time()

    perc, ssim = numerical_comparison(res_bench_for_rem, comp_bench_for)
    print('time: ',round((bench_fwd_timer-bench_back_timer)/60,2) )
    print('                Percent difference: ', perc, '%')
    print('                      SSIM average: ', ssim)
    print('')



    ## CAR BACKWARD ENERGY ###
    
    # print('CAR BACKWARD   ', end=" ")
    # res_car_back_ins = car_back_insert(car)
    # diff_car_back_ins = difference_image(comp_car_back, res_car_back_ins)
    # cv.imwrite('images/output/res_car_back_ins.png', res_car_back_ins)
    # cv.imwrite('images/output/diff_car_back_ins.png', diff_car_back_ins)

    car_back_timer = time.time()

    # perc, ssim = numerical_comparison(res_car_back_ins, comp_car_back)
    # print('time: ',round((car_back_timer - bench_fwd_timer)/60,2) )
    # print('                Percent difference: ', perc, '%')
    # print('                      SSIM average: ', ssim)
    # print('')



    ### CAR FORWARD ENERGY ###
    
    print('CAR FORWARD    ', end=" ")
    res_car_for_ins = car_for_insert(car, redSeams=True)
    diff_car_for_ins = difference_image(comp_car_for, res_car_for_ins)
    cv.imwrite('images/output/res_car_for_ins_red.png', res_car_for_ins)
    cv.imwrite('images/output/diff_car_for_ins.png', diff_car_for_ins)

    car_fwd_timer = time.time()

    perc, ssim = numerical_comparison(res_car_for_ins, comp_car_for)
    print('time: ',round((car_fwd_timer-car_back_timer)/60,2) )
    print('                Percent difference: ', perc, '%')
    print('                      SSIM average: ', ssim)
    print('')



    timer2 = time.time()

    print('Total execution time in minutes: ',round((timer2-timer1)/60,2))



