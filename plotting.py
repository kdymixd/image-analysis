from cgi import parse_multipart
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import abc
import fit

class AbstractPlot(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def first_plot(self):
        pass

    @abc.abstractmethod    
    def update_plot(self):
        pass


class NoPlot(AbstractPlot):
    def __init__(self):
        pass
    def first_plot(self):
        return super().first_plot()
    
    def update_plot(self):
        return super().update_plot()

class PlotSigma(AbstractPlot):
    def __init__(self,ratio_image, factor=0.5, cut="mean"):
        # mpl.use('TkAgg')
        # plt.ion()
        self.fig=plt.figure(constrained_layout=True, figsize=(factor*(ratio_image*2+1),factor*(ratio_image+1)))
        self.spec=self.fig.add_gridspec(ncols=3, nrows=2, width_ratios=[ratio_image,ratio_image,1], height_ratios=[ratio_image,1])
        self.ax_im_raw=self.fig.add_subplot(self.spec[0,0])
        self.ax_im_fit=self.fig.add_subplot(self.spec[0,1])
        self.ax_plot_y=self.fig.add_subplot(self.spec[0,2]) 
        self.ax_plot_x=self.fig.add_subplot(self.spec[1,1])
        if cut == "mean":
            self.cut=0
        elif cut== "max":
            self.cut=1
            self.vline=None
            self.hline=None
        else:
            self.cut=0
    
    def first_plot(self,im_raw, params, timestamp=None):
        
        ratio_im=im_raw.shape[1]/im_raw.shape[0]
        y,x= np.indices(im_raw.shape)
        im_fit=fit.rot_gaussian(x,y,*params)
        fig_size=self.fig.get_size_inches()
        
        if timestamp is not None:
            self.fig.canvas.set_window_title(timestamp)
        self.fig.set_size_inches(ratio_im*fig_size[0],fig_size[1])
        self.plot_im_raw=self.ax_im_raw.imshow(im_raw, origin="lower")
        self.plot_im_fit=self.ax_im_fit.imshow(im_fit, origin="lower")
        self.plot_im_raw.set_clim(np.min(im_fit), np.max(im_fit))
        self.ax_im_raw.set_xticks([]), self.ax_im_raw.set_yticks([])
        self.ax_im_fit.set_xticks([]), self.ax_im_fit.set_yticks([])
        
        if self.cut==0:
            self.plot_x_raw,=self.ax_plot_x.plot(np.nanmean(im_raw, axis=0),'b--')
            self.plot_y_raw,=self.ax_plot_y.plot(np.nanmean(im_raw, axis=1), np.arange(im_raw.shape[0]),'b--')
            self.plot_x_fit,=self.ax_plot_x.plot(np.nanmean(im_fit, axis=0),'r')
            self.plot_y_fit,=self.ax_plot_y.plot(np.nanmean(im_fit, axis=1),np.arange(im_raw.shape[0]), 'r')
        else :
            x_max, y_max= self.find_argmax(im_raw)
            self.plot_x_raw,=self.ax_plot_x.plot(im_raw[y_max,:],'b--')
            self.plot_y_raw,=self.ax_plot_y.plot(im_raw[:,x_max], np.arange(im_raw.shape[0]),'b--')
            self.plot_x_fit,=self.ax_plot_x.plot(im_fit[y_max,:],'r')
            self.plot_y_fit,=self.ax_plot_y.plot(im_fit[:,x_max],np.arange(im_raw.shape[0]), 'r')
            self.vline=self.ax_im_raw.axvline(x=x_max, color="black")
            self.hline=self.ax_im_raw.axhline(y=y_max, color="black")

        
        self.fig.canvas.draw()
        plt.pause(0.1)
        # self.fig.canvas.flush_events()
    
    def update_plot(self, im_raw,params, timestamp=None):
        y,x= np.indices(im_raw.shape)
        im_fit=fit.rot_gaussian(x,y,*params)
        self.plot_im_raw.set_data(im_raw)
        self.plot_im_fit.set_data(im_fit)
        self.plot_im_raw.set_clim(np.min(im_fit), np.max(im_fit))
        # self.plot_im_raw.autoscale()
        self.plot_im_fit.autoscale()
        if self.cut==0:
            self.plot_x_raw.set_ydata(np.nanmean(im_raw, axis=0))
            self.plot_y_raw.set_xdata(np.nanmean(im_raw, axis=1))
            self.plot_x_fit.set_ydata(np.nanmean(im_fit, axis=0))
            self.plot_y_fit.set_xdata(np.nanmean(im_fit, axis=1))
        else: 
            x_max, y_max= self.find_argmax(im_raw)
            self.plot_x_raw.set_ydata(im_raw[y_max,:])
            self.plot_y_raw.set_xdata(im_raw[:,x_max])
            self.plot_x_fit.set_ydata(im_fit[y_max,:])
            self.plot_y_fit.set_xdata(im_fit[:,x_max])
            self.vline.set_xdata(x_max)
            self.hline.set_ydata(y=y_max)
        if timestamp is not None:
            self.fig.canvas.set_window_title(timestamp)
        self.ax_plot_x.relim(), self.ax_plot_y.relim()
        self.ax_plot_x.autoscale_view(True,True,True), self.ax_plot_y.autoscale_view(True,True,True)
        self.fig.canvas.draw()
        plt.pause(0.1)
        # self.fig.canvas.flush_events()

    def find_argmax(self, im):
        y,x =np.indices(im.shape)
        mask=np.invert(np.isnan(im))
        im=im-np.nanmin(im)
        norm=im/np.nanmax(im)
        centroid_x=int(np.average(x[mask], weights=norm[mask]**10))
        centroid_y=int(np.average(y[mask], weights=norm[mask]**10))
        return (centroid_x, centroid_y)


class PlotSigma2(AbstractPlot):
    def __init__(self,ratio_image, factor=0.5, cut="mean"):
        # mpl.use('TkAgg')
        # plt.ion()
        self.fig=plt.figure(constrained_layout=True, figsize=(factor*(ratio_image*2+1),factor*(ratio_image+1)))
        self.spec=self.fig.add_gridspec(ncols=3, nrows=2, width_ratios=[ratio_image,ratio_image,1], height_ratios=[ratio_image,1])
        self.ax_im_raw=self.fig.add_subplot(self.spec[0,0])
        self.ax_im_fit=self.fig.add_subplot(self.spec[0,1])
        self.ax_plot_y=self.fig.add_subplot(self.spec[0,2]) 
        self.ax_plot_x=self.fig.add_subplot(self.spec[1,1])
        if cut == "mean":
            self.cut=0
        elif cut== "max":
            self.cut=1
            self.vline=None
            self.hline=None
        else:
            self.cut=0
    
    def first_plot(self,im_raw, params, timestamp=None):
        ratio_im=im_raw.shape[1]/im_raw.shape[0]
        y,x= np.indices(im_raw.shape)
        im_fit=fit.rot_gaussian2(x,y,*params)

        params_thermal = np.copy(params)
        params_thermal[0], params_thermal[1] = np.min([params_thermal[0], params_thermal[1]]), np.max([params_thermal[0], params_thermal[1]])
        params_thermal[1] = 0
        im_fit_thermal=fit.rot_gaussian2(x,y,*params_thermal)

        fig_size=self.fig.get_size_inches()
        if timestamp is not None:
            self.fig.canvas.set_window_title(timestamp)
        self.fig.set_size_inches(ratio_im*fig_size[0],fig_size[1])
        self.plot_im_raw=self.ax_im_raw.imshow(im_raw, origin="lower")
        self.plot_im_fit=self.ax_im_fit.imshow(im_fit, origin="lower")
        self.plot_im_raw.set_clim(np.min(im_fit), np.max(im_fit))
        self.ax_im_raw.set_xticks([]), self.ax_im_raw.set_yticks([])
        self.ax_im_fit.set_xticks([]), self.ax_im_fit.set_yticks([])
        if self.cut==0:
            self.plot_x_raw,=self.ax_plot_x.plot(np.nanmean(im_raw, axis=0),'b--')
            self.plot_y_raw,=self.ax_plot_y.plot(np.nanmean(im_raw, axis=1), np.arange(im_raw.shape[0]),'b--')
            self.plot_x_fit,=self.ax_plot_x.plot(np.nanmean(im_fit, axis=0),'r')
            self.plot_x_fit_thermal,=self.ax_plot_x.plot(np.nanmean(im_fit_thermal, axis=0),'k--')
            self.plot_y_fit,=self.ax_plot_y.plot(np.nanmean(im_fit, axis=1),np.arange(im_raw.shape[0]), 'r')
            self.plot_y_fit_thermal,=self.ax_plot_y.plot(np.nanmean(im_fit_thermal, axis=1),np.arange(im_raw.shape[0]), 'k--')

        else :
            x_max, y_max= self.find_argmax(im_raw)
            self.plot_x_raw,=self.ax_plot_x.plot(im_raw[y_max,:],'b--')
            self.plot_y_raw,=self.ax_plot_y.plot(im_raw[:,x_max], np.arange(im_raw.shape[0]),'b--')
            self.plot_x_fit,=self.ax_plot_x.plot(im_fit[y_max,:],'r')
            self.plot_y_fit,=self.ax_plot_y.plot(im_fit[:,x_max],np.arange(im_raw.shape[0]), 'r')
            self.plot_x_fit_thermal,=self.ax_plot_x.plot(im_fit_thermal[y_max,:],'k--')
            self.plot_y_fit_thermal,=self.ax_plot_y.plot(im_fit_thermal[:,x_max],np.arange(im_raw.shape[0]), 'k--')
            self.vline=self.ax_im_raw.axvline(x=x_max, color="black")
            self.hline=self.ax_im_raw.axhline(y=y_max, color="black")

        
        self.fig.canvas.draw()
        plt.pause(0.1)
        # self.fig.canvas.flush_events()
    
    def update_plot(self, im_raw,params, timestamp=None):
        y,x= np.indices(im_raw.shape)
        im_fit=fit.rot_gaussian2(x,y,*params)

        params_thermal = np.copy(params)
        params_thermal[0], params_thermal[1] = np.min([params_thermal[0], params_thermal[1]]), np.max([params_thermal[0], params_thermal[1]])
        params_thermal[1] = 0
        im_fit_thermal=fit.rot_gaussian2(x,y,*params_thermal)

        self.plot_im_raw.set_data(im_raw)
        self.plot_im_fit.set_data(im_fit)
        self.plot_im_raw.set_clim(np.min(im_fit), np.max(im_fit))
        # self.plot_im_raw.autoscale()
        self.plot_im_fit.autoscale()
        if self.cut==0:
            self.plot_x_raw.set_ydata(np.nanmean(im_raw, axis=0))
            self.plot_y_raw.set_xdata(np.nanmean(im_raw, axis=1))
            self.plot_x_fit.set_ydata(np.nanmean(im_fit, axis=0))
            self.plot_y_fit.set_xdata(np.nanmean(im_fit, axis=1))
            self.plot_x_fit_thermal.set_ydata(np.nanmean(im_fit_thermal, axis=0))
            self.plot_y_fit_thermal.set_xdata(np.nanmean(im_fit_thermal, axis=1))

        else: 
            x_max, y_max= self.find_argmax(im_raw)
            self.plot_x_raw.set_ydata(im_raw[y_max,:])
            self.plot_y_raw.set_xdata(im_raw[:,x_max])
            self.plot_x_fit.set_ydata(im_fit[y_max,:])
            self.plot_y_fit.set_xdata(im_fit[:,x_max])
            self.plot_x_fit_thermal.set_ydata(im_fit_thermal[y_max,:])
            self.plot_y_fit_thermal.set_xdata(im_fit_thermal[:,x_max])
            self.vline.set_xdata(x_max)
            self.hline.set_ydata(y=y_max)
        if timestamp is not None:
            self.fig.canvas.set_window_title(timestamp)
        self.ax_plot_x.relim(), self.ax_plot_y.relim()
        self.ax_plot_x.autoscale_view(True,True,True), self.ax_plot_y.autoscale_view(True,True,True)
        self.fig.canvas.draw()
        plt.pause(0.1)
        # self.fig.canvas.flush_events()

    def find_argmax(self, im):
        y,x =np.indices(im.shape)
        mask=np.invert(np.isnan(im))
        im=im-np.nanmin(im)
        norm=im/np.nanmax(im)
        centroid_x=int(np.average(x[mask], weights=norm[mask]**10))
        centroid_y=int(np.average(y[mask], weights=norm[mask]**10))
        return (centroid_x, centroid_y)


class PlotPropagation(AbstractPlot):
    def __init__(self) :
        # mpl.use('TkAgg')
        # plt.ion()
        self.fig, (self.ax_im, self.ax_plot) = plt.subplots(1, 2, figsize=(10, 7), constrained_layout=True)
    
    def first_plot(self,im, optical_axis, raw_curve, params, timestamp=None):
        fit_curve=fit.max_waist(optical_axis, *params)
        ratio_im=im.shape[1]/im.shape[0]
        fig_size=self.fig.get_size_inches()
        if timestamp is not None:
            self.fig.canvas.set_window_title(timestamp)
        self.fig.set_size_inches(ratio_im*fig_size[0],fig_size[1])
        self.plot_im=self.ax_im.imshow(im, origin="lower")
        self.ax_im.set_xticks([]), self.ax_im.set_yticks([])
        self.plot_raw,=self.ax_plot.plot(optical_axis, raw_curve, 'b.')
        self.plot_fit,=self.ax_plot.plot(optical_axis, fit_curve,'r--')
    
    def update_plot(self, im, optical_axis, raw_curve, params, timestamp=None):
        fit_curve=fit.max_waist(optical_axis, *params)
        self.plot_im.set_data(im)
        self.plot_im.autoscale()
        self.plot_raw.set_ydata(raw_curve)
        self.plot_raw.set_xdata(optical_axis)
        self.plot_fit.set_ydata(fit_curve)
        self.plot_fit.set_xdata(optical_axis)


        if timestamp is not None:
            self.fig.canvas.set_window_title(timestamp)

        self.ax_plot.relim(), self.ax_plot.relim()
        self.ax_plot.autoscale_view(True,True,True)
        self.fig.canvas.draw()
        plt.pause(0.1)


# if __name__ == "__main__":
    # path="/home/francor/passerelle_na/Rubidium 4/Data/Absorption/2022/01/13/Dip_Repump3/182812#Dip_Repump3#CMOT_Detuning = -20#CMOT_MOTCoilsI = 12#General_Iter = 1#Imaging_Freq = 0#Molasses_Duration = 0,02#Molasses_VVA = 10#TOF_Duration = 1"
    # cam_name="Lumeneracamera1"
    # img=get_absorption_picture(path, cam_name)
    # a=plot_ROI_selection(img)
    # print(a)