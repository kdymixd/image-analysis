import abc
import plotting
import dy
import numpy as np
import fit

class AbstractAnalyis(abc.ABC):
    plot: plotting.AbstractPlot
    parameters_to_save: list
    
    @abc.abstractmethod
    def __init__(self, pixel_size):
        self.first_time=True
        self.pixel_size=pixel_size

    @abc.abstractmethod
    def calculate_atom_number(self):
        pass

    @abc.abstractmethod #defines the values
    def analyze(self, img):
        self.analysis_params=[]
        pass
    
    @abc.abstractmethod
    def first_plot(self, img,timestamp):
        self.plot.first_plot(img, self.analysis_params, timestamp)

    @abc.abstractmethod
    def update_plot(self, img, timestamp):
        self.plot.update_plot(img, self.analysis_params, timestamp)

    @abc.abstractmethod
    def create_plot(self):
        pass

    #return a dict of parameters we want to save to the database:
    def save_params(self, img, file_name, thermal=True):
        params=vars(self)
        dict_parameters_to_save={k:v for k,v in params.items() if k in self.parameters_to_save}
        # dict_parameters_to_save["atom_number"]=self.calculate_atom_number(img, thermal=thermal)
        if thermal==False:
            dict_parameters_to_save["atom_number"]=self.calculate_atom_number(img, thermal=True)
            dict_parameters_to_save["atom_number_condensed"]=self.calculate_atom_number(img, thermal=False)
        elif thermal==True:
            dict_parameters_to_save["atom_number"]=self.calculate_atom_number(img, thermal=True)
        dict_parameters_to_save["file_name"]=file_name
        return dict_parameters_to_save

    def plot_and_save(self, img, timestamp, file_name, thermal=True):
        self.analyze(img)
        if self.first_time:
            self.first_plot(img, timestamp)
        else :
            self.update_plot(img, timestamp)
        self.first_time=False
        return self.save_params(img, file_name, thermal=thermal)
    
    

class AnalysisSigma(AbstractAnalyis):
    parameters_to_save=["center_x", "center_y", "sigma_x", "sigma_y", "theta"]
    
    def __init__(self,pixel_size, bin=1, angle=None):
        super().__init__(pixel_size)
        self.bin=bin
        self.angle=angle
        self.pixel_size

    def calculate_atom_number(self, img, thermal=True):
        return self.A*2*np.pi*self.sigma_x*self.sigma_y/dy.sigma0
    
    def create_plot(self):
        self.plot=plotting.PlotSigma(3,1, cut='max')
        
    def first_plot(self, img, timestamp):
        return super().first_plot(img, timestamp)
    
    def update_plot(self, img, timestamp):
        return super().update_plot(img, timestamp)

    def analyze(self, img):
        y, x=np.indices(img.shape)
        popt=fit.fit_gaussian_2D(x,y,img, bin=self.bin, angle=self.angle)
        self.analysis_params=popt
        self.A=popt[0]
        self.sigma_x=popt[3]*self.pixel_size
        self.sigma_y=popt[4]*self.pixel_size
        self.center_x=popt[1]*self.pixel_size
        self.center_y=popt[2]*self.pixel_size
        self.theta=popt[-1]

class AnalysisSigma2(AbstractAnalyis):
    
    parameters_to_save=["center_x", "center_y", "sigma_x", "sigma_y", "sigma2_x", "sigma2_y", "theta"]
    
    def __init__(self,pixel_size, bin=1, angle=None):
        super().__init__(pixel_size)
        self.bin=bin
        self.angle=angle
        self.pixel_size

    def calculate_atom_number(self, img, thermal=True):
        if thermal==True:
            return self.A*2*np.pi*self.sigma_x*self.sigma_y/dy.sigma0
        return self.A2*2*np.pi*self.sigma2_x*self.sigma2_y/dy.sigma0   
    
    def create_plot(self):
        self.plot=plotting.PlotSigma2(3,1, cut='max')
        
    def first_plot(self, img, timestamp):
        return super().first_plot(img, timestamp)
    
    def update_plot(self, img, timestamp):
        return super().update_plot(img, timestamp)

    def analyze(self, img):
        y, x=np.indices(img.shape)
        popt=fit.fit_gaussian2_2D(x,y,img, bin=self.bin, angle=self.angle)
        self.analysis_params=popt
        self.A, self.A2, self.center_x, self.center_y, self.sigma_x, self.sigma_y, self.sigma2_x, self.sigma2_y, self.offset, self.theta = popt
        self.sigma_x, self.sigma_y, self.sigma2_x, self.sigma2_y = self.sigma_x*self.pixel_size, self.sigma_y*self.pixel_size, self.sigma2_x*self.pixel_size, self.sigma2_y*self.pixel_size
        
        # verify that the condensed / thermal fraction were not inverted during curve_fit (sigma_thermal should be > sigma_condensed)
        if self.sigma2_x > self.sigma_x:
            self.sigma_x, self.sigma2_x = self.sigma2_x, self.sigma_x
            self.sigma_y, self.sigma2_y = self.sigma2_y, self.sigma_y
            self.A, self.A2 = self.A2, self.A        
    
class AnalysisNoFit(AbstractAnalyis):
    parameters_to_save=[]
    def __init__(self,pixel_size):
        super().__init__(pixel_size)
    
    def create_plot(self):
        self.plot=plotting.NoPlot()
    def first_plot(self, img, timestamp):
        return 
     
    def update_plot(self, img, timestamp):
        return

    def analyze(self, img):
        return super().analyze(img)

    def calculate_atom_number(self,img, thermal=True):
        return self.pixel_size**2/dy.sigma0*np.nansum(img)
    

class AnalysisPropagation(AbstractAnalyis):
    parameters_to_save=["center_z","w_center"]
    def __init__(self, pixel_size, wl, step=10):
        super().__init__(pixel_size)
        self.wl=wl
        self.step=step

    def create_plot(self):
        self.plot=plotting.PlotPropagation()
    def analyze(self, img):
        self.optical_axis, self.z_max, self.analysis_params = fit.fit_propagation(img,self.step,self.wl,self.pixel_size)
        self.A=self.analysis_params[0]
        self.center_z=self.analysis_params[1]
        self.waist_center=self.analysis_params[2]
    def calculate_atom_number(self, img, thermal=True):
        return np.NaN

    def save_params(self, file_name):
        return super().save_params(file_name)
    
    def first_plot(self, img, timestamp):
        self.plot.first_plot(img, self.optical_axis, self.z_max, self.analysis_params, timestamp=None)
    
    def update_plot(self, img, timestamp):
        self.plot.update_plot(img, self.optical_axis, self.z_max, self.analysis_params, timestamp=None)