from math import isfinite
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
from matplotlib.widgets import RectangleSelector
#import fit

# Specific command for Spyder to plot figures and interact with them
# try:
#     import IPython
#     shell = IPython.get_ipython()
#     shell.enable_matplotlib(gui='qt')
# except Exception as inst:
#     print(inst)
#     print("Backend problem. Check your Spyder backend")


def read_image(path):
    try:  
        img=cv.imread(path, cv.IMREAD_UNCHANGED) #we load the images
    except cv.error as e:
        # inspect error object
        print(e)
        for k in dir(e):
            if k[0:2] != "__":
                print("e.%s = %s" % (k, getattr(e, k)))
    return img


UseBestOffset = 0
ROIBestOffset = [600,750,700,850]

#takes as argument the path of the folder of the shot (full path) and the camera name
#return the computed absorption picture

def select_roi(frame_path, path_analyzed_data, frame_name, camera_name, name=None):
    """
    Allows to select the Region Of Interest of a picture 
    """
    im=get_absorption_picture(frame_path, path_analyzed_data, frame_name, camera_name)
    ROI=plot_ROI_selection(im, name)
    print("ROI : ", ROI)
    return(ROI)

def open_picture(frame_path, path_analyzed_data, frame_name, camera_name, roi_background=None):
    """
    Open a multiple tiff file to extract a plot 
    """
    #Open the multiple tiff file and split it into 3 images
    img_pil = Image.open(frame_path)
    img=[]
    
    img_path = path_analyzed_data + frame_name + ".tiff"
    
    for i in range(3):
        try:
            img_pil.seek(i)
            img.append(np.asarray(img_pil))
            # img_pil.save('page_%s.tiff'%(i,))
            if i==0:
                img_pil.save(img_path)
        except EOFError:
            break
        
    # path_analyzed_data_with_at = path_analyzed_data + 
    img_with_atoms=read_image(img_path)
    img_no_atoms=img[1]
    #img_no_atoms=read_image(img_pil)
    img_background=img[2]

    return {"atoms": img_with_atoms, 'no atoms': img_no_atoms, 'background': img_background}
    
    
def open_picture2(folder_path,camera_name, roi_background=None):
    #print(self.background)
    #The suffixes we have to add to the folder name to obtain the corresponding pictures 

    if (folder_path.split("#")[1] == "Princeton") or (folder_path.split("#")[2] == "Princeton"):
        camera_name="Princeton"
        with_atoms_suffix="frames_0001.tiff"
        no_atoms_suffix="frames_0002.tiff"
        background_suffix="frames_0000.tiff"
        img_with_atoms=read_image(os.path.join(folder_path, with_atoms_suffix)) #we load the images
        img_no_atoms=read_image(os.path.join(folder_path, no_atoms_suffix))
        img_background=cv.imread("Z:\\Rubidium 4\\Python Programs\\GUI_RbIV_newest\\bg.tiff", cv.IMREAD_UNCHANGED)

    else:
        with_atoms_suffix="_With.png"
        no_atoms_suffix="_NoAt.png"
        background_suffix="_Bgd.png"
        for file in os.listdir(folder_path):
            if file.endswith(with_atoms_suffix):
                camera_name=file[:-len(with_atoms_suffix)] #we get the camera name from the image file

        img_with_atoms=read_image(os.path.join(folder_path, camera_name+with_atoms_suffix)) #we load the images
        img_no_atoms=read_image(os.path.join(folder_path, camera_name+no_atoms_suffix))
        img_background=read_image(os.path.join(folder_path, camera_name+background_suffix))

    return {"atoms": img_with_atoms, 'no atoms': img_no_atoms, 'background':img_background}

def get_absorption_picture(frame_path, path_analyzed_data, frame_name, camera_name, roi_background=None, C_sat=np.inf):
    """
    Extract the absorption picture from a file
    """
    dict_pictures=open_picture(frame_path, path_analyzed_data, frame_name, camera_name, roi_background)
    transmission=np.nan_to_num((dict_pictures['no atoms']-dict_pictures['background'])/(dict_pictures['atoms']-dict_pictures['background']), nan=float('NaN'))
    #transmission=np.nan_to_num((dict_pictures['no atoms'])/(dict_pictures['atoms']), nan=float('NaN'))
    
    # transmission[transmission<0]=np.nan
    abs=np.nan_to_num(np.log(transmission) - (dict_pictures['atoms']-dict_pictures['no atoms'])/C_sat, nan=np.nan, posinf=np.nan, neginf=np.nan)
    abs[abs>10]=np.nan
    # exclude=dict_pictures['no atoms']/np.nanmax(dict_pictures['no atoms'])<0.02
    # abs[exclude]=np.nan
    print("Max {}".format(np.nanmax(abs)))
    print("End of absorption picture")
    return(abs)

def plot_ROI_selection(img,name=None):
    
    def selection_callback(eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        ROI[2], ROI[0] = int(eclick.xdata), int(eclick.ydata)
        ROI[3], ROI[1] = int(erelease.xdata), int(erelease.ydata)
        

    fig=plt.figure(constrained_layout=True)
    ax=fig.add_subplot(111)
    myplot=ax.imshow(img, origin="lower", cmap="rainbow")  # plot something
    if name is not None:
        ax.set_title(name)
    fig.colorbar(myplot)
    ROI=[0,-1,0,-1]
    print("pass1")
    selector= RectangleSelector(
        ax, selection_callback,
        useblit=True,
        button=[1, 3],  # disable middle button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True)
    print("pass2")
    selector.set_active(True)
    print("pass3")

    plt.show(block=True)
    print("pass4")
    return np.s_[ROI[0]:ROI[1], ROI[2]:ROI[3]]


def open_picture_old(folder_path,camera_name, roi_background=None):
    """
    Old Open picture function
    """
    #The suffixes we have to add to the folder name to obtain the corresponding pictures 
    with_atoms_suffix="_With.png"
    no_atoms_suffix="_NoAt.png"
    background_suffix="_Bgd.png"
    img_with_atoms=cv.imread(os.path.join(folder_path, camera_name+with_atoms_suffix), cv.IMREAD_UNCHANGED)
    img_no_atoms=cv.imread(os.path.join(folder_path, camera_name+no_atoms_suffix), cv.IMREAD_UNCHANGED)
    ### Remove the offest
    if  roi_background is not None:
        ratio = np.nanmean(img_no_atoms[roi_background]/img_with_atoms[roi_background])
        img_no_atoms = img_no_atoms/ratio
        print("Open_pic_ratio : ", ratio)
    ###
    img_background=cv.imread(os.path.join(folder_path, camera_name+background_suffix), cv.IMREAD_UNCHANGED)
    return {"atoms": img_with_atoms, 'no atoms': img_no_atoms, 'background':img_background}

if __name__=='__main__':

    path = "Z:\\Rubidium 4\\Data\\Absorption\\2022\\03\\03\\Left\\"

    # import os 
    # name_tot=[]
    # for file in os.scandir(path):
    #     if file.name.startswith('181258'):
    #         name_tot.append(file.name)
    # print(name_tot)


    # camera_name="Lumeneracamera1"
    # for folder in name_tot:
    #     try:
    #         #ROI= np.s_[400:600,500:800]
    #         plt.figure()
    #         im=get_absorption_picture(path+folder,camera_name)
    #         # im=im[ROI]
    #         plt.imshow(im)
    #         plt.colorbar()
    #     except:
    #         pass

    # plt.show()

    # image=im-np.nanmin(im)
    # image=(image/np.nanmax(image)*255)
    # image=image.astype('uint8')
    # cleaning=np.isfinite(im)
    # y,x=np.indices(im.shape)
    # x_clean=x[cleaning]
    # y_clean=y[cleaning]
    # im_clean=im[cleaning]
    # image_denoised=cv.fastNlMeansDenoising(image)
    # # diff=image.astype('float64')-image_denoised.astype('float64')
    # # print(np.max(np.abs(diff)))
    # plt.imshow(image_denoised)
    # plt.title("denoised")
    # # initial_guess=fit.find_initial_guess(x_clean, y_clean, im_clean)
    # # print(initial_guess)
    # y,x =np.indices(im.shape)
    # nozero=(image_denoised != 0)
    # x_clean=x[nozero]
    # y_clean=y[nozero]
    # im_clean=image_denoised[nozero]
    # initial_guess=fit.find_initial_guess(x_clean, y_clean, im_clean)
    # print(initial_guess)
    # print(np.max(im_clean), np.min(im_clean))
    # plt.figure()
    # plt.imshow(fit.rot_gaussian(x,y, *initial_guess))
    # popt=fit.fit_gaussian_2D(x_clean,y_clean, im_clean)
    # plt.figure()
    # plt.imshow(fit.rot_gaussian(x,y, *popt))
    # plt.show()
    # plt.figure()
    # plt.title("6")
    # plt.imshow(get_absorption_picture(path_6, os.path.basename(path_6),camera_name))
    # plt.show()