import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.signal

###Operation on images###
def rebin(a, bin):
    a=a[:bin*(a.shape[0]//bin), :bin*(a.shape[1]//bin)]
    return np.nanmean(np.nanmean(a.reshape(a.shape[0]//bin, bin, a.shape[1]//bin,bin),axis=-1), axis=-2)

#Rotate the 2d arrays (x,y) by an angle theta
def rotation(x, y, theta):
    angle=np.deg2rad(theta)
    return (np.cos(angle)*x+np.sin(angle)*y, -np.sin(angle)*x + np.cos(angle)*y)

###Mathematical functions###
#Simple 2d gaussian
def gaussian(x, y, A, sigma_x, sigma_y, offset):
    return A*np.exp(-((x)**2/(2*sigma_x**2) + (y)**2/(2*sigma_y**2))) + offset

def gaussian1d(x, A, center, waist, offset):
    return A*np.exp(-2*(x-center)**2/waist**2)+offset

#2d gaussian with rotation
def rot_gaussian(x, y, A, x_0, y_0, sigma_x, sigma_y, offset, theta):
    x_rot, y_rot=rotation(x-x_0, y-y_0, theta)
    return gaussian(x_rot, y_rot, A, sigma_x, sigma_y, offset)

#Two 2d gaussian (thermal + BEC)
def gaussian2(x, y, A, A2, sigma_x, sigma_y, sigma2_x, sigma2_y, offset):
    return A*np.exp(-(x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2))) + A2*np.exp(-(x**2/(2*sigma2_x**2) + y**2/(2*sigma2_y**2))) + offset

#2d gaussian with rotation
def rot_gaussian2(x, y, A, A2, x_0, y_0, sigma_x, sigma_y, sigma2_x, sigma2_y, offset, theta):
    x_rot, y_rot=rotation(x-x_0, y-y_0, theta)
    return gaussian2(x_rot, y_rot, A, A2, sigma_x, sigma_y, sigma2_x, sigma2_y, offset)

#Lorentzian for gaussian beam propagation
def max_waist(x, A, x_0,w_0,offest,wl):
    x=x
    x_0=x_0
    z_r=np.pi*w_0**2/(wl)
    waist=w_0*np.sqrt(1+((x-x_0)/z_r)**2)
    return (A/waist**2)+offest

###2D gaussian fit###
#finds an initial guess for a 2d gaussian fit
def find_initial_guess(x,y,z):
    offset = np.min(z)
    z = z - offset
    A = np.max(z)

    #calculate the centroid
    centroid_x, centroid_y = np.average(x, weights=z), np.average(y, weights=z)
    cov = np.zeros((2,2))

    #calculate the covariance matrix to find the sigma_x, sigma_y and the angle
    cov[0,0] = np.average((x)**2, weights=z) - centroid_x**2
    cov[1,1] = np.average((y)**2, weights=z) - centroid_y**2
    cov[0,1] = np.average(x*y, weights=z) - centroid_x*centroid_y
    cov[1,0] = cov[0,1]

    #We diagonalize the covariance matrix
    eig = np.linalg.eigh(cov)
    sigma_x, sigma_y = np.sqrt(eig[0])

    ##The index tells us the eigenvector that has 2 values of the same sign i.e (cos, sin)
    index = 0
    if eig[1][0, 0]*eig[1][0, 1]>0:
        index = 0
    else:
        index = 1

    #We take the arctan of the ratio of the compenents of the eigenvector (cos, sin) to get the angle
    theta = np.rad2deg(np.arctan(eig[1][index, 1] / eig[1][index, 0]))
    return (A, centroid_x, centroid_y, sigma_x, sigma_y, offset, theta)

def fit_gaussian_2D(x, y, z,  bin=1, angle=None, initial_guess=None):
    print("Max z {}".format(np.nanmax(z)))

    #Get initial guess
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    if not bin==1:
        x=rebin(x, bin)
        y=rebin(y, bin)
        z=rebin(z, bin) 

    #We clean the images by removing the NAN 
    cleaning = np.isfinite(z)
    x = x[cleaning]
    y = y[cleaning]
    z = z[cleaning]

    if initial_guess is None:
        initial_guess=find_initial_guess(x,y,z)
    print("Initial guess: {}".format(initial_guess))

    if angle is None:
        func_to_fit = lambda x, A, x_0, y_0, sigma_x, sigma_y, offset, theta: rot_gaussian(*x, A, x_0, y_0, sigma_x, sigma_y, offset, theta)
    else:
        func_to_fit = lambda x, A, x_0, y_0, sigma_x, sigma_y, offset : rot_gaussian(*x, A, x_0, y_0, sigma_x, sigma_y,  offset, angle)
    min_z, max_z = np.min(z), np.max(z)

    try :
        if angle is None:
            popt, _ = curve_fit(func_to_fit, [x.ravel(), y.ravel()], z.ravel(), p0=initial_guess, bounds=([0, x_min, y_min, 0, 0, min_z, 0], [2*(max_z-min_z), x_max, y_max, x_max, y_max, max_z, 89.99]), maxfev=50)
            return popt
        else: 
            popt, _ = curve_fit(func_to_fit, [x.ravel(), y.ravel()], z.ravel(), p0=initial_guess[:-1], bounds=([0, x_min, y_min, 0, 0, min_z], [2*(max_z-min_z), x_max, y_max, x_max, y_max, max_z]), maxfev=50)
            return np.append(popt, angle)

    except Exception as e:
        print("Could not fit : {}".format(e))
        empty_array = np.empty(len(initial_guess))
        empty_array[:] = np.NAN
        return empty_array

###2D gaussian2 fit###

def fit_gaussian2_2D(x, y, z,  bin=1, angle=None, initial_guess=None):
    print("Max z {}".format(np.nanmax(z)))

    #Get initial guess
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    if not bin==1:
        x=rebin(x, bin)
        y=rebin(y, bin)
        z=rebin(z, bin) 

    #We clean the images by removing the NAN 
    cleaning = np.isfinite(z)
    x = x[cleaning]
    y = y[cleaning]
    z = z[cleaning]

    if initial_guess is None:
        initial_guess=find_initial_guess(x,y,z)
    initial_guess = np.concatenate((initial_guess[:1], initial_guess[:3], np.array(initial_guess[3:5])*np.sqrt(2), np.array(initial_guess[3:5])/np.sqrt(2), initial_guess[5:]))
    print("Initial guess: {}".format(initial_guess))

    if angle is None:
        func_to_fit = lambda x, A, A2, x_0, y_0, sigma_x, sigma_y, sigma2_x, sigma2_y, offset, theta : rot_gaussian2(*x, A, A2, x_0, y_0, sigma_x, sigma_y, sigma2_x, sigma2_y, offset,theta)
    else:
        func_to_fit = lambda x, A, A2, x_0, y_0, sigma_x, sigma_y, sigma2_x, sigma2_y, offset : rot_gaussian2(*x, A, A2, x_0, y_0, sigma_x, sigma_y, sigma2_x, sigma2_y, offset, angle)
    min_z, max_z = np.min(z), np.max(z)

    try :
        if angle is None:
            popt, _ = curve_fit(func_to_fit, [x.ravel(), y.ravel()], z.ravel(), p0=initial_guess, bounds=([0, 0, x_min, y_min, 0, 0, 0, 0, min_z, 0], [2*(max_z-min_z), 2*(max_z-min_z), x_max, y_max, x_max, y_max, x_max, y_max, max_z, 89.99]), maxfev=400)
            return popt
        else: 
            popt, _ = curve_fit(func_to_fit, [x.ravel(), y.ravel()], z.ravel(), p0=initial_guess[:-1], bounds=([0, 0, x_min, y_min, 0, 0, 0, 0, min_z], [2*(max_z-min_z), 2*(max_z-min_z), x_max, y_max, x_max, y_max, x_max, y_max, max_z]), maxfev=400)
            return np.append(popt, angle)

    except Exception as e:
        print("Could not fit : {}".format(e))
        empty_array = np.empty(len(initial_guess))
        empty_array[:] = np.NAN
        return empty_array



###Propagation fit###
def fit_gauss1D(x, y):
    x_min, x_max = np.min(x), np.max(x)
    blurred = scipy.signal.convolve(y, 1/20*np.ones(20), mode="same")
    y_min, y_max = np.min(blurred), np.max(blurred)
    initial_guess = [(y_max-y_min), x[np.argmax(blurred)], 40/1.86, y_min]
    bounds = ([initial_guess[0]/5, x_min, 0, -np.inf], [10*initial_guess[0], x_max, 500, np.inf])
    popt, _ = curve_fit(gaussian1d, x, y, p0=initial_guess, bounds=bounds)
    return popt

#Fit rows by 1D gaussian by a pixel step of "step"
def get_row_fit(img, step):
    x_cut=np.arange(img.shape[1])
    x_max=np.zeros(img.shape[0]//step)
    z_max=np.zeros(img.shape[0]//step)
    y_max=np.arange(img.shape[0],step=step)
    for i,_ in enumerate(x_max):
        params=fit_gauss1D(x_cut, np.mean(img[step*i: step*(i+1)], axis=0))
        z_max[i], x_max[i]= params[0:2]
    return x_max, y_max,z_max

#Fit the obtained max intensity for each row using a Lorentzian to find focus
def fit_waist(x, y, wl,pixel_size):
    x=x
    y_min, y_max= np.min(y), np.max(y)
    initial_guess=[(y_max-y_min)*(30e-6)**2, 650*pixel_size, 30e-6,y_min]
    boundaries=([initial_guess[0]/10,0,0,0],[10*initial_guess[0], 4000, 100e-6,y_max])
    fit_func= lambda x,A,x_0,w_0, offset : max_waist(x,A,x_0,w_0, offset, wl)
    popt,_=curve_fit(fit_func, x, y, p0=initial_guess, bounds=boundaries)
    popt=[*popt, wl]
    return popt

#Fits propagation along optical axis
def fit_propagation(img,step, wl,pixel_size):
    img=np.nan_to_num(img,nan=0, neginf=0, posinf=0)
    x_max, y_max,z_max=get_row_fit(img, step)
    # axs[0].plot(x_max, y_max,".")
    optical_axis=np.sqrt((x_max-x_max[0])**2+y_max**2)*pixel_size
    params=fit_waist(optical_axis, z_max, wl,pixel_size)
    return(optical_axis, z_max, params)

if __name__ == "__main__":
    #####Check if the gaussians work correctly#####
    x=np.linspace(-10,20,1000)
    y=np.linspace(-10,20,1000)
    x,y=np.meshgrid(x,y)
    z=rot_gaussian(x,y,10,5,5,5,0.5,30,10)
    
    #####Check if the initial guess method works correctly#####
    #Generate dataset
    z_data=rot_gaussian(x,y,10,5,5,5,5,25,10)
    plt.imshow(z_data)
    plt.title("Perfect gaussian")
    plt.colorbar()
    plt.figure()
    z_noise=2*np.random.normal(size=z_data.shape)
    z_data+=z_noise
    #Generate an initial guess
    initial_guess=find_initial_guess(x,y,z_data)
    print(initial_guess)
    plt.imshow(z_data)
    plt.title("data")
    plt.colorbar()
    plt.figure()
    plt.imshow(rot_gaussian(x,y,*initial_guess))
    plt.title("initial guess")
    plt.colorbar()
    plt.figure()
    popt= fit_gaussian_2D(x,y, z_data, bin=2)
    print(popt)
    plt.imshow(rot_gaussian(x,y,*popt))
    plt.title("fit")
    plt.colorbar()
    plt.show()