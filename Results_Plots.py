import numpy as np
from matplotlib.pylab import *
from PyAstronomy import pyasl
from astropy.io import fits
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import constants as consts
from astropy import wcs
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.stats import chisquare
from scipy import ndimage
from lmfit import Model
import math
import statistics
import cmasher as cmr
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import matplotlib.gridspec as gridspec
import threadcount as tc
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} km/s"

def gauss(x, a1, m1, sd1, c):
    curve = a1 * np.exp(-1.0 * ((x - m1) ** 2.0) / (2.0 * sd1 ** 2.0)) + c
    return curve


def twogauss(x, a1, m1, sd1, a2, m2, sd2, c):
    curve = a1 * np.exp(-1.0 * ((x - m1) ** 2.0) / (2.0 * sd1 ** 2.0)) + \
            a2 * np.exp(-1.0 * ((x - m2) ** 2.0) / (2.0 * sd2 ** 2.0)) + c
    return curve


def threegauss(x, a1, m1, sd1, a2, m2, sd2, a3, m3, sd3, c):
    curve = a1 * np.exp(-1.0 * ((x - m1) ** 2.0) / (2.0 * sd1 ** 2.0)) + \
            a2 * np.exp(-1.0 * ((x - m2) ** 2.0) / (2.0 * sd2 ** 2.0)) + \
            a3 * np.exp(-1.0 * ((x - m3) ** 2.0) / (2.0 * sd3 ** 2.0)) + c
    return curve

def slope(x, m, c):
    y = m * x + c
    return y

def slope_c0(x, m):
    y = m * x
    return y

def montecarloerror(function, x, slope, mes_error, ntrials):

    ntrials = 11
    bestfitparams = np.full([ntrials], np.nan)
    for trial in range(ntrials):
        trial_nw_dy_arcsec = np.random.uniform(low=x[0], high=x[-1], size = len(x))
        trial_nw_halfwidth_50 = function(trial_nw_dy_arcsec, slope) + np.random.normal(scale=mes_error, size=len(x))
        trial_fit_conical = curve_fit(function, trial_nw_dy_arcsec, trial_nw_halfwidth_50, [2])[0]
        bestfitparams[trial] = trial_fit_conical[0]
    error = np.std(bestfitparams)
    return error

def neprofile(alpha, r):
    nescaleheight = 0.8
    nespaxel = ne * (r/nescaleheight) ** (alpha)
    if nespaxel > ne:
        nespaxel=ne
    return nespaxel
    
def vprofile(vscaleheight, r, voutmax):
        if vscaleheight == 0:
            v = voutmax
        else:
            v = voutmax * (1 - exp(-r/vscaleheight))
        return v

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 20
EVEN_BIGGER_SIZE = 24

plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=EVEN_BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=EVEN_BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

cval = consts.c.to('km/s').value

sparcsec = 1/0.2915
spkpc = 1/0.1976
center = (35, 61)
z = 0.03386643885613516

gal = 'MRK1486'
filename_preamble = 'nelder_extinction_corrected'

cube0 = fits.open(gal + '/' + gal + '_red_metacube_extinction_corrected.fits')
reddata = cube0[0].data
redhdr = cube0[0].header

cube1 = fits.open(gal + '/' + gal + '_red_varcube_extinction_corrected.fits')
redvar = cube1[0].data

cube2 = fits.open(gal + '/' + 'Red_Cont_PPXF_extinction_corrected.fits')
contdata = cube2[0].data

linefits = tc.fit.ResultDict.loadtxt(gal + '/' + filename_preamble + '_5007_mc_best_fit.txt')
singlelinefits = tc.fit.ResultDict.loadtxt(gal + '/' + filename_preamble + '_5007_simple_model.txt')
hbetalinefits = tc.fit.ResultDict.loadtxt(gal + '/' + filename_preamble + '_Hbeta_mc_best_fit.txt')
hbetasinglelinefits = tc.fit.ResultDict.loadtxt(gal + '/' + filename_preamble + '_Hbeta_simple_model.txt')
outflow_prof = np.loadtxt(gal + '/' + filename_preamble + '_outflow_width_prof.txt', delimiter=',')

img = np.asarray(Image.open(gal + '/' + gal + '_three_colour.png'))
img = ndimage.rotate(img, 45)

outflow_prof_dx_spaxels = outflow_prof[:, 0] * sparcsec + center[1] - 27
outflow_prof_dy_spaxels = outflow_prof[:, 1] * sparcsec + center[0]
outflow_prof_dy_spaxels_hst = outflow_prof[:, 1] * sparcsec * (-1.) + center[0]
outflow_prof_50pc_halfwidth = outflow_prof[:, 2] * sparcsec
outflow_prof_90pc_halfwidth = outflow_prof[:, 3] * sparcsec

se_minor_axis_outflow_dx = int(outflow_prof[0, 0] * sparcsec + center[1])
nw_minor_axis_outflow_dx = int(outflow_prof[-1, 0] * sparcsec + center[1])
se_minor_axis_outflow_dy = (np.rint(outflow_prof[:29, 1] * sparcsec + center[0])).astype(int)
nw_minor_axis_outflow_dy = (np.rint(outflow_prof[29:, 1] * sparcsec + center[0])).astype(int)
se_minor_axis_outflow_50pc_halfwidth = (np.rint(outflow_prof[:29, 2] * sparcsec)).astype(int)
nw_minor_axis_outflow_50pc_halfwidth = (np.rint(outflow_prof[29:, 2] * sparcsec)).astype(int)
se_minor_axis_outflow_90pc_halfwidth = (np.rint(outflow_prof[:29, 3] * sparcsec)).astype(int)
nw_minor_axis_outflow_90pc_halfwidth = (np.rint(outflow_prof[29:, 3] * sparcsec)).astype(int)

reddata = reddata - contdata

lamstart = redhdr['CRVAL3']
deltalam = redhdr['CDELT3']
lamlength = redhdr['NAXIS3']
racenter = redhdr['CRVAL1']  # degrees
decenter = redhdr['CRVAL2']  # degrees
expstart = redhdr['DATE-BEG']

redwave = np.arange(lamstart, lamstart + (lamlength * deltalam), deltalam)
redvacwave = pyasl.airtovac2(redwave)
keck = EarthLocation.of_site('Keck Observatory')
redsc = SkyCoord(ra=racenter * u.deg, dec=decenter * u.deg)
redbarycorr = redsc.radial_velocity_correction(obstime=Time(expstart), location=keck)
redbcorr = redbarycorr.to(u.km / u.s)
redbarywave = redvacwave * (1.0 + (redbcorr.value / cval))
deredshiftbarywave = redbarywave / (1+z)


shape = np.shape(reddata)
xpix = np.arange(0, shape[1], 1)
ypix = np.arange(0, shape[2], 1)

# Kinematic galaxy center, determined from kinematic_center.py

center = (35, 61)
diskleft = 37
diskright = 85
snrcut = 5
oiii5007lambda = 5006.843
hbetalambda = 4861.333

hbetaderedshiftbarywave = cval * (redbarywave - (hbetalambda * (1+z) + 1.3)) / hbetalambda

oiii5007singlesnr = singlelinefits["snr"]
oiii5007singlesigma = singlelinefits["g1_sigma"] * cval / oiii5007lambda         
oiii5007singlemean = cval * (singlelinefits["g1_center"] - oiii5007lambda) / oiii5007lambda
oiii5007singleamp = singlelinefits["g1_height"]
oiii5007singleflux = singlelinefits["g1_flux"]
oiii5007singlec = singlelinefits["c"]

hbetasinglesnr = hbetasinglelinefits["snr"]
hbetasinglesigma = hbetasinglelinefits["g1_sigma"] * cval / hbetalambda         
hbetasinglemean = cval * (hbetasinglelinefits["g1_center"] - hbetalambda) / hbetalambda
hbetasingleamp = hbetasinglelinefits["g1_height"]
hbetasingleflux = hbetasinglelinefits["g1_flux"]
hbetasinglec = hbetasinglelinefits["c"]

oiii5007snr = linefits["avg_g1_center"]
oiii5007choice = linefits["choice"]
oiii5007c = linefits["avg_c"]
oiii5007success = linefits["success"]

oiii5007flux1 = np.full([shape[1], shape[2]], np.nan)
oiii5007amp1 = np.full([shape[1], shape[2]], np.nan)
oiii5007sig1 = np.full([shape[1], shape[2]], np.nan)
oiii5007mean1 = np.full([shape[1], shape[2]], np.nan)
oiii5007mean1error = np.full([shape[1], shape[2]], np.nan)

oiii5007flux2 = np.full([shape[1], shape[2]], np.nan)
oiii5007amp2 = np.full([shape[1], shape[2]], np.nan)
oiii5007sig2 = np.full([shape[1], shape[2]], np.nan)
oiii5007mean2 = np.full([shape[1], shape[2]], np.nan)
oiii5007mean2error = np.full([shape[1], shape[2]], np.nan)

oiii5007flux3 = np.full([shape[1], shape[2]], np.nan)
oiii5007amp3 = np.full([shape[1], shape[2]], np.nan)
oiii5007sig3 = np.full([shape[1], shape[2]], np.nan)
oiii5007mean3 = np.full([shape[1], shape[2]], np.nan)
oiii5007mean3error = np.full([shape[1], shape[2]], np.nan)

oiii5007totalflux = np.full([shape[1], shape[2]], np.nan)
c23fluxrat = np.full([shape[1], shape[2]], np.nan)

meanoiii5007totalflux = np.full([shape[1], shape[2]], np.nan)
c23meanfluxrat = np.full([shape[1], shape[2]], np.nan)
meansingsig = np.full([shape[1], shape[2]], np.nan)
lineseparation = np.full([shape[1], shape[2]], np.nan)
lineseparationerror = np.full([shape[1], shape[2]], np.nan)
meanlineseparation = np.full([shape[1], shape[2]], np.nan)
meanlineseparationerror = np.full([shape[1], shape[2]], np.nan)
oiii5007peak = np.full([shape[1], shape[2]], np.nan)
oiii5007medianave = np.full([shape[1], shape[2]], np.nan)
oiii5007singlemeanave = np.full([shape[1], shape[2]], np.nan)

outflux = np.full([shape[1], shape[2]], np.nan)
outvel = np.full([shape[1], shape[2]], np.nan)
influx = np.full([shape[1], shape[2]], np.nan)
invel = np.full([shape[1], shape[2]], np.nan)
systflux = np.full([shape[1], shape[2]], np.nan)
outcomponent = np.full([shape[1], shape[2]], 0)
incomponent = np.full([shape[1], shape[2]], 0)
systcomponent = np.full([shape[1], shape[2]], 0)

galaxyrow = (30, 39)
galaxycol = (41, 80)

for x in xpix:
    for y in ypix:
        oiii5007peakind = np.argmax(reddata[1523:1551, x, y])
        oiii5007peak[x, y] = redbarywave[oiii5007peakind + 1523]

        if oiii5007snr[x, y] >= 5 and oiii5007success[x, y] == 1:
            if oiii5007choice[x, y] == 1:
                oiii5007flux1[x, y] = linefits["avg_g1_flux"][x, y]
                oiii5007amp1[x, y] = linefits["avg_g1_height"][x, y]
                oiii5007sig1[x, y] = linefits["avg_g1_sigma"][x, y] * cval / oiii5007lambda
                oiii5007mean1[x, y] = cval * (linefits["avg_g1_center"][x, y] - oiii5007lambda) / oiii5007lambda
                oiii5007mean1error[x, y] = (cval ** 2) * (linefits["avg_g1_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                oiii5007flux2[x, y] = 0.
                oiii5007flux3[x, y] = 0.
                lineseparation[x, y] = 0.
                lineseparationerror[x, y] = 0.
            if oiii5007choice[x, y] == 2:
                oiii5007flux3[x, y] = 0.
                if linefits["avg_g1_flux"][x, y] >= linefits["avg_g2_flux"][x, y]:
                    oiii5007flux1[x, y] = linefits["avg_g1_flux"][x, y]
                    oiii5007amp1[x, y] = linefits["avg_g1_height"][x, y]
                    oiii5007sig1[x, y] = linefits["avg_g1_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean1[x, y] = cval * (linefits["avg_g1_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean1error[x, y] = (cval ** 2) * (linefits["avg_g1_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux2[x, y] = linefits["avg_g2_flux"][x, y]
                    oiii5007amp2[x, y] = linefits["avg_g2_height"][x, y]
                    oiii5007sig2[x, y] = linefits["avg_g2_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean2[x, y] = cval * (linefits["avg_g2_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean2error[x, y] = (cval ** 2) * (linefits["avg_g2_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                else:
                    oiii5007flux1[x, y] = linefits["avg_g2_flux"][x, y]
                    oiii5007amp1[x, y] = linefits["avg_g2_height"][x, y]
                    oiii5007sig1[x, y] = linefits["avg_g2_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean1[x, y] = cval * (linefits["avg_g2_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean1error[x, y] = (cval ** 2) * (linefits["avg_g2_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux2[x, y] = linefits["avg_g1_flux"][x, y]
                    oiii5007amp2[x, y] = linefits["avg_g1_height"][x, y]
                    oiii5007sig2[x, y] = linefits["avg_g1_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean2[x, y] = cval * (linefits["avg_g1_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean2error[x, y] = (cval ** 2) * (linefits["avg_g1_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                lineseparation[x, y] = abs(oiii5007mean1[x, y] - oiii5007mean2[x, y])
                lineseparationerror[x, y] = oiii5007mean1error[x, y] + oiii5007mean2error[x, y]
            if oiii5007choice[x, y] == 3:
                if linefits["avg_g1_flux"][x, y] >= linefits["avg_g2_flux"][x, y] >= linefits["avg_g3_flux"][x, y]:
                    oiii5007flux1[x, y] = linefits["avg_g1_flux"][x, y]
                    oiii5007amp1[x, y] = linefits["avg_g1_height"][x, y]
                    oiii5007sig1[x, y] = linefits["avg_g1_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean1[x, y] = cval * (linefits["avg_g1_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean1error[x, y] = (cval ** 2) * (linefits["avg_g1_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux2[x, y] = linefits["avg_g2_flux"][x, y]
                    oiii5007amp2[x, y] = linefits["avg_g2_height"][x, y]
                    oiii5007sig2[x, y] = linefits["avg_g2_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean2[x, y] = cval * (linefits["avg_g2_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean2error[x, y] = (cval ** 2) * (linefits["avg_g2_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux3[x, y] = linefits["avg_g3_flux"][x, y]
                    oiii5007amp3[x, y] = linefits["avg_g3_height"][x, y]
                    oiii5007sig3[x, y] = linefits["avg_g3_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean3[x, y] = cval * (linefits["avg_g3_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean3error[x, y] = (cval ** 2) * (linefits["avg_g3_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                if linefits["avg_g1_flux"][x, y] >= linefits["avg_g3_flux"][x, y] >= linefits["avg_g2_flux"][x, y]:
                    oiii5007flux1[x, y] = linefits["avg_g1_flux"][x, y]
                    oiii5007amp1[x, y] = linefits["avg_g1_height"][x, y]
                    oiii5007sig1[x, y] = linefits["avg_g1_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean1[x, y] = cval * (linefits["avg_g1_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean1error[x, y] = (cval ** 2) * (linefits["avg_g1_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux2[x, y] = linefits["avg_g3_flux"][x, y]
                    oiii5007amp2[x, y] = linefits["avg_g3_height"][x, y]
                    oiii5007sig2[x, y] = linefits["avg_g3_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean2[x, y] = cval * (linefits["avg_g3_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean2error[x, y] = (cval ** 2) * (linefits["avg_g3_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux3[x, y] = linefits["avg_g2_flux"][x, y]
                    oiii5007amp3[x, y] = linefits["avg_g2_height"][x, y]
                    oiii5007sig3[x, y] = linefits["avg_g2_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean3[x, y] = cval * (linefits["avg_g2_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean3error[x, y] = (cval ** 2) * (linefits["avg_g2_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                if linefits["avg_g2_flux"][x, y] >= linefits["avg_g1_flux"][x, y] >= linefits["avg_g3_flux"][x, y]:
                    oiii5007flux1[x, y] = linefits["avg_g2_flux"][x, y]
                    oiii5007amp1[x, y] = linefits["avg_g2_height"][x, y]
                    oiii5007sig1[x, y] = linefits["avg_g2_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean1[x, y] = cval * (linefits["avg_g2_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean1error[x, y] = (cval ** 2) * (linefits["avg_g2_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux2[x, y] = linefits["avg_g1_flux"][x, y]
                    oiii5007amp2[x, y] = linefits["avg_g1_height"][x, y]
                    oiii5007sig2[x, y] = linefits["avg_g1_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean2[x, y] = cval * (linefits["avg_g1_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean2error[x, y] = (cval ** 2) * (linefits["avg_g1_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux3[x, y] = linefits["avg_g3_flux"][x, y]
                    oiii5007amp3[x, y] = linefits["avg_g3_height"][x, y]
                    oiii5007sig3[x, y] = linefits["avg_g3_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean3[x, y] = cval * (linefits["avg_g3_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean3error[x, y] = (cval ** 2) * (linefits["avg_g3_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                if linefits["avg_g2_flux"][x, y] >= linefits["avg_g3_flux"][x, y] >= linefits["avg_g1_flux"][x, y]:
                    oiii5007flux1[x, y] = linefits["avg_g2_flux"][x, y]
                    oiii5007amp1[x, y] = linefits["avg_g2_height"][x, y]
                    oiii5007sig1[x, y] = linefits["avg_g2_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean1[x, y] = cval * (linefits["avg_g2_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean1error[x, y] = (cval ** 2) * (linefits["avg_g2_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux2[x, y] = linefits["avg_g3_flux"][x, y]
                    oiii5007amp2[x, y] = linefits["avg_g3_height"][x, y]
                    oiii5007sig2[x, y] = linefits["avg_g3_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean2[x, y] = cval * (linefits["avg_g3_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean2error[x, y] = (cval ** 2) * (linefits["avg_g3_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux3[x, y] = linefits["avg_g1_flux"][x, y]
                    oiii5007amp3[x, y] = linefits["avg_g1_height"][x, y]
                    oiii5007sig3[x, y] = linefits["avg_g1_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean3[x, y] = cval * (linefits["avg_g1_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean3error[x, y] = (cval ** 2) * (linefits["avg_g1_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                if linefits["avg_g3_flux"][x, y] >= linefits["avg_g1_flux"][x, y] >= linefits["avg_g2_flux"][x, y]:
                    oiii5007flux1[x, y] = linefits["avg_g3_flux"][x, y]
                    oiii5007amp1[x, y] = linefits["avg_g3_height"][x, y]
                    oiii5007sig1[x, y] = linefits["avg_g3_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean1[x, y] = cval * (linefits["avg_g3_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean1error[x, y] = (cval ** 2) * (linefits["avg_g3_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux2[x, y] = linefits["avg_g1_flux"][x, y]
                    oiii5007amp2[x, y] = linefits["avg_g1_height"][x, y]
                    oiii5007sig2[x, y] = linefits["avg_g1_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean2[x, y] = cval * (linefits["avg_g1_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean2error[x, y] = (cval ** 2) * (linefits["avg_g1_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux3[x, y] = linefits["avg_g2_flux"][x, y]
                    oiii5007amp3[x, y] = linefits["avg_g2_height"][x, y]
                    oiii5007sig3[x, y] = linefits["avg_g2_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean3[x, y] = cval * (linefits["avg_g2_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean3error[x, y] = (cval ** 2) * (linefits["avg_g2_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                if linefits["avg_g3_flux"][x, y] >= linefits["avg_g2_flux"][x, y] >= linefits["avg_g1_flux"][x, y]:
                    oiii5007flux1[x, y] = linefits["avg_g3_flux"][x, y]
                    oiii5007amp1[x, y] = linefits["avg_g3_height"][x, y]
                    oiii5007sig1[x, y] = linefits["avg_g3_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean1[x, y] = cval * (linefits["avg_g3_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean1error[x, y] = (cval ** 2) * (linefits["avg_g3_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux2[x, y] = linefits["avg_g2_flux"][x, y]
                    oiii5007amp2[x, y] = linefits["avg_g2_height"][x, y]
                    oiii5007sig2[x, y] = linefits["avg_g2_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean2[x, y] = cval * (linefits["avg_g2_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean2error[x, y] = (cval ** 2) * (linefits["avg_g2_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                    oiii5007flux3[x, y] = linefits["avg_g1_flux"][x, y]
                    oiii5007amp3[x, y] = linefits["avg_g1_height"][x, y]
                    oiii5007sig3[x, y] = linefits["avg_g1_sigma"][x, y] * cval / oiii5007lambda
                    oiii5007mean3[x, y] = cval * (linefits["avg_g1_center"][x, y] - oiii5007lambda) / oiii5007lambda
                    oiii5007mean3error[x, y] = (cval ** 2) * (linefits["avg_g1_center_err"][x, y] ** 2) / (oiii5007lambda ** 2)
                lineseparation[x, y] = max(abs(oiii5007mean1[x, y] - oiii5007mean2[x, y]),
                                           abs(oiii5007mean2[x, y] - oiii5007mean3[x, y]),
                                           abs(oiii5007mean3[x, y] - oiii5007mean1[x, y]))
                if abs(oiii5007mean1[x, y] - oiii5007mean2[x, y]) >= abs(oiii5007mean2[x, y] - oiii5007mean3[x, y]) and abs(oiii5007mean1[x, y] - oiii5007mean2[x, y]) >= abs(oiii5007mean1[x, y] - oiii5007mean3[x, y]):
                    lineseparationerror[x, y] = oiii5007mean1error[x, y] + oiii5007mean2error[x, y]
                elif abs(oiii5007mean2[x, y] - oiii5007mean3[x, y]) >= abs(oiii5007mean1[x, y] - oiii5007mean2[x, y]) and abs(oiii5007mean2[x, y] - oiii5007mean3[x, y]) >= abs(oiii5007mean1[x, y] - oiii5007mean3[x, y]):
                    lineseparationerror[x, y] = oiii5007mean2error[x, y] + oiii5007mean3error[x, y]
                else:
                    lineseparationerror[x, y] = oiii5007mean1error[x, y] + oiii5007mean3error[x, y]
            if lineseparation[x, y] > 1000:
                lineseparation[x, y] = np.nan
        c23fluxrat[x, y] = (oiii5007flux2[x, y] + oiii5007flux3[x, y])/(oiii5007flux1[x, y] + oiii5007flux2[x, y] + oiii5007flux3[x, y])
        oiii5007totalflux[x, y] = oiii5007flux1[x, y] + oiii5007flux2[x, y] + oiii5007flux3[x, y]
        meanoiii5007totalflux = oiii5007totalflux

g1_center = hbetalinefits["avg_g1_center"]
g2_center = hbetalinefits["avg_g2_center"]
g3_center = hbetalinefits["avg_g3_center"]
g1_sigma = hbetalinefits["avg_g1_sigma"]
g2_sigma = hbetalinefits["avg_g2_sigma"]
g3_sigma = hbetalinefits["avg_g3_sigma"]

hbetalinefits["avg_g1_center"] = cval * (hbetalinefits["avg_g1_center"] - hbetalambda) / hbetalambda
hbetalinefits["avg_g2_center"] = cval * (hbetalinefits["avg_g2_center"] - hbetalambda) / hbetalambda
hbetalinefits["avg_g3_center"] = cval * (hbetalinefits["avg_g3_center"] - hbetalambda) / hbetalambda
hbetalinefits["avg_g1_sigma"] = hbetalinefits["avg_g1_sigma"] * cval / hbetalambda
hbetalinefits["avg_g2_sigma"] = hbetalinefits["avg_g2_sigma"] * cval / hbetalambda
hbetalinefits["avg_g3_sigma"] = hbetalinefits["avg_g3_sigma"] * cval / hbetalambda

onefits = 0
twofits = 0
threefits = 0
disk_outnumber_spaxels = 0
leftpix = 1220
rightpix = 1255
xdata = np.linspace(hbetaderedshiftbarywave[leftpix], hbetaderedshiftbarywave[rightpix], 10000)
pixels = []
for x in np.arange(galaxyrow[0], galaxyrow[1] + 1, 1):
    for y in np.arange(galaxycol[0], galaxycol[1] + 1, 1):
        if hbetalinefits["choice"][x, y] == 1:
            systflux[x, y] = hbetalinefits["avg_g1_flux"][x, y]
            systcomponent[x, y] = 1
            onefits += 1
        if hbetalinefits["choice"][x, y] == 2:
            twofits += 1
            if hbetalinefits["avg_g1_height"][x, y]  > hbetalinefits["avg_g2_height"][x, y]:
                outcomponent[x, y] = 2
                systcomponent[x, y] = 1
                outflux[x, y] = hbetalinefits["avg_g2_flux"][x, y]
                outvel[x, y] = abs(hbetalinefits["avg_g2_center"][x, y] - hbetalinefits["avg_g1_center"][x, y]) + 2 * hbetalinefits["avg_g2_sigma"][x, y]
                systflux[x, y] = hbetalinefits["avg_g1_flux"][x, y]
                disk_outnumber_spaxels += 1
            else:
                outcomponent[x, y] = 1
                systcomponent[x, y] = 2
                outflux[x, y] = hbetalinefits["avg_g1_flux"][x, y]
                outvel[x, y] = abs(hbetalinefits["avg_g1_center"][x, y] - hbetalinefits["avg_g2_center"][x, y]) + 2 * hbetalinefits["avg_g1_sigma"][x, y]
                systflux[x, y] = hbetalinefits["avg_g2_flux"][x, y]
                disk_outnumber_spaxels += 1
            if outvel[x, y] > 600:
                plt.plot(xdata, gauss(xdata, hbetalinefits["avg_g1_height"][x, y], hbetalinefits["avg_g1_center"][x, y], hbetalinefits["avg_g1_sigma"][x, y], hbetalinefits["avg_c"][x, y]), color='r')
                plt.plot(xdata, gauss(xdata, hbetalinefits["avg_g2_height"][x, y], hbetalinefits["avg_g2_center"][x, y], hbetalinefits["avg_g2_sigma"][x, y], hbetalinefits["avg_c"][x, y]), color='b')
                plt.scatter(hbetaderedshiftbarywave[leftpix:rightpix], reddata[leftpix:rightpix, x, y], color='k')
                plt.title(f'Outflow Component: {outcomponent[x, y]}. Outflow Flux: {outflux[x, y]}. Outvel: {outvel[x, y]}', fontsize=15)
                plt.savefig(gal + '/diskfits/' + filename_preamble + '_' + str(x) + '_' + str(y) + '_diskfit.png', bbox_inches='tight')
                plt.close()  

        elif hbetalinefits["choice"][x, y] == 3:
            threefits += 1
            if hbetalinefits["avg_g2_center"][x, y] < hbetalinefits["avg_g1_center"][x, y] and hbetalinefits["avg_g2_center"][x, y] < hbetalinefits["avg_g3_center"][x, y] and hbetalinefits["avg_g2_height"][x, y] < 0.5 * hbetasinglelinefits["g1_height"][x, y]:
                outflux[x, y] = hbetalinefits["avg_g2_flux"][x, y]
                if hbetalinefits["avg_g3_height"][x, y]< hbetalinefits["avg_g1_height"][x, y]:
                    outcomponent[x, y] = 2
                    incomponent[x, y] = 3
                    systcomponent[x, y] = 1
                    influx[x, y] = hbetalinefits["avg_g3_flux"][x, y]
                    outvel[x, y] = abs(hbetalinefits["avg_g2_center"][x, y] - hbetalinefits["avg_g1_center"][x, y]) + 2 * hbetalinefits["avg_g2_sigma"][x, y]
                    invel[x, y] = abs(hbetalinefits["avg_g3_center"][x, y] - hbetalinefits["avg_g1_center"][x, y]) + 2 * hbetalinefits["avg_g3_sigma"][x, y]
                    systflux[x, y] = hbetalinefits["avg_g1_flux"][x, y]
                    disk_outnumber_spaxels += 1
                else:
                    outcomponent[x, y] = 2
                    incomponent[x, y] = 1
                    systcomponent[x, y] = 3
                    influx[x, y] = hbetalinefits["avg_g1_flux"][x, y]
                    outvel[x, y] = abs(hbetalinefits["avg_g2_center"][x, y] - hbetalinefits["avg_g3_center"][x, y]) + 2 * hbetalinefits["avg_g2_sigma"][x, y]
                    invel[x, y] = abs(hbetalinefits["avg_g1_center"][x, y] - hbetalinefits["avg_g3_center"][x, y]) + 2 * hbetalinefits["avg_g1_sigma"][x, y]
                    systflux[x, y] = hbetalinefits["avg_g3_flux"][x, y]
                    disk_outnumber_spaxels += 1
            elif hbetalinefits["avg_g3_center"][x, y] < hbetalinefits["avg_g1_center"][x, y] and hbetalinefits["avg_g3_center"][x, y] < hbetalinefits["avg_g2_center"][x, y] and hbetalinefits["avg_g3_height"][x, y] < 0.5 * hbetasinglelinefits["g1_height"][x, y]:
                outflux[x, y] = hbetalinefits["avg_g3_flux"][x, y]
                if hbetalinefits["avg_g2_height"][x, y] < hbetalinefits["avg_g1_height"][x, y]:
                    outcomponent[x, y] = 3
                    incomponent[x, y] = 2
                    systcomponent[x, y] = 1
                    influx[x, y] = hbetalinefits["avg_g2_flux"][x, y]
                    outvel[x, y] = abs(hbetalinefits["avg_g3_center"][x, y] - hbetalinefits["avg_g1_center"][x, y]) + 2 * hbetalinefits["avg_g3_sigma"][x, y]
                    invel[x, y] = abs(hbetalinefits["avg_g2_center"][x, y] - hbetalinefits["avg_g1_center"][x, y]) + 2 * hbetalinefits["avg_g2_sigma"][x, y]
                    systflux[x, y] = hbetalinefits["avg_g1_flux"][x, y]
                    disk_outnumber_spaxels += 1
                else:
                    outcomponent[x, y] = 3
                    incomponent[x, y] = 1
                    systcomponent[x, y] = 2
                    influx[x, y] = hbetalinefits["avg_g1_flux"][x, y]
                    outvel[x, y] = abs(hbetalinefits["avg_g3_center"][x, y] - hbetalinefits["avg_g2_center"][x, y]) + 2 * hbetalinefits["avg_g3_sigma"][x, y]
                    invel[x, y] = abs(hbetalinefits["avg_g1_center"][x, y] - hbetalinefits["avg_g2_center"][x, y]) + 2 * hbetalinefits["avg_g1_sigma"][x, y]
                    systflux[x, y] = hbetalinefits["avg_g2_flux"][x, y]
                    disk_outnumber_spaxels += 1
            elif hbetalinefits["avg_g1_center"][x, y] < hbetalinefits["avg_g2_center"][x, y] and hbetalinefits["avg_g1_center"][x, y] < hbetalinefits["avg_g3_center"][x, y] and hbetalinefits["avg_g1_height"][x, y] < 0.5 * hbetasinglelinefits["g1_height"][x, y]:
                outflux[x, y] = hbetalinefits["avg_g1_flux"][x, y]
                if hbetalinefits["avg_g2_height"][x, y] < hbetalinefits["avg_g3_height"][x, y]:
                    outcomponent[x, y] = 1
                    incomponent[x, y] = 2
                    systcomponent[x, y] = 3
                    influx[x, y] = hbetalinefits["avg_g1_flux"][x, y]
                    outvel[x, y] = abs(hbetalinefits["avg_g1_center"][x, y] - hbetalinefits["avg_g3_center"][x, y]) + 2 * hbetalinefits["avg_g1_sigma"][x, y]
                    invel[x, y] = abs(hbetalinefits["avg_g2_center"][x, y] - hbetalinefits["avg_g3_center"][x, y]) + 2 * hbetalinefits["avg_g2_sigma"][x, y]
                    systflux[x, y] = hbetalinefits["avg_g3_flux"][x, y]
                    disk_outnumber_spaxels += 1
                else:
                    outcomponent[x, y] = 1
                    incomponent[x, y] = 3
                    systcomponent[x, y] = 2
                    influx[x, y] = hbetalinefits["avg_g1_flux"][x, y]
                    outvel[x, y] = abs(hbetalinefits["avg_g1_center"][x, y] - hbetalinefits["avg_g2_center"][x, y]) + 2 * hbetalinefits["avg_g1_sigma"][x, y]
                    invel[x, y] = abs(hbetalinefits["avg_g3_center"][x, y] - hbetalinefits["avg_g2_center"][x, y]) + 2 * hbetalinefits["avg_g3_sigma"][x, y]
                    systflux[x, y] = hbetalinefits["avg_g2_flux"][x, y]
                    disk_outnumber_spaxels += 1
            if outvel[x, y] > 600:
                plt.plot(xdata, gauss(xdata, hbetalinefits["avg_g1_height"][x, y], hbetalinefits["avg_g1_center"][x, y], hbetalinefits["avg_g1_sigma"][x, y], hbetalinefits["avg_c"][x, y]), color='r')
                plt.plot(xdata, gauss(xdata, hbetalinefits["avg_g2_height"][x, y], hbetalinefits["avg_g2_center"][x, y], hbetalinefits["avg_g2_sigma"][x, y], hbetalinefits["avg_c"][x, y]), color='b')
                plt.plot(xdata, gauss(xdata, hbetalinefits["avg_g3_height"][x, y], hbetalinefits["avg_g3_center"][x, y], hbetalinefits["avg_g3_sigma"][x, y], hbetalinefits["avg_c"][x, y]), color='g')
                plt.scatter(hbetaderedshiftbarywave[leftpix:rightpix], reddata[leftpix:rightpix, x, y], color='k')
                plt.title(f'Outflow Component: {outcomponent[x, y]}. Outflow Flux: {outflux[x, y]}. Outvel: {outvel[x, y]}', fontsize=15)
                plt.savefig(gal + '/diskfits/' + filename_preamble + '_' + str(x) + '_' + str(y) + '_diskfit.png', bbox_inches='tight')
                plt.close()  
        else:
            outflux[x, y] = 0
            influx[x, y] = 0
            outvel[x, y] = 0
            invel[x, y] = 0

        if isnan(outvel[x, y]) or isnan(outflux[x, y]):
            outflux[x, y] = 0
            outvel[x, y] = 0
            
        if isnan(invel[x, y]) or isnan(influx[x, y]):
            influx[x, y] = 0
            invel[x, y] = 0

fratio = outflux/systflux

# for x in np.arange(galaxyrow[0], galaxyrow[1] + 1, 1):
#     for y in np.arange(galaxycol[0], galaxycol[1] + 1, 1):
#         if fratio[x, y] == 0:
#             fratio[x, y] = np.nan
#         if fratio[x, y] > 1:
#             print(f"fratio: {fratio[x, y]}")
#         if outvel[x, y] == 0:
#             outvel[x, y] = np.nan

bright_center = (31, 65)

bright_region_fbroad = np.nansum(outflux[bright_center[0]-2:bright_center[0]+2, bright_center[1]-2:bright_center[1]+2])
print(bright_region_fbroad)

print(f"One Component Disk fits: {onefits}")
print(f"Two Component Disk fits: {twofits}")
print(f"Three Component Disk fits: {threefits}")
print(f"Total Galaxy Flux: {np.nansum(systflux)}")
print(f"Total Disk Edge Outflow Flux: {np.nansum(outflux)}")
print(f"Total Disk Edge Inflow Flux: {np.nansum(influx)}")
print(f"Total FBroad/FNarrow: {(np.nansum(outflux) + np.nansum(influx))/np.nansum(systflux)}")

se_minor_axis_outflow_total_flux_50pc = 0.
se_minor_axis_outflow_total_flux_90pc = 0.
nw_minor_axis_outflow_total_flux_50pc = 0.
nw_minor_axis_outflow_total_flux_90pc = 0.

for x in range(len(nw_minor_axis_outflow_dy)):
    nw_minor_axis_outflow_total_flux_50pc += np.nansum(hbetasingleflux[nw_minor_axis_outflow_dy[x], nw_minor_axis_outflow_dx - nw_minor_axis_outflow_50pc_halfwidth[x]:nw_minor_axis_outflow_dx + nw_minor_axis_outflow_50pc_halfwidth[x] + 1])
    nw_minor_axis_outflow_total_flux_90pc += np.nansum(hbetasingleflux[nw_minor_axis_outflow_dy[x], nw_minor_axis_outflow_dx - nw_minor_axis_outflow_90pc_halfwidth[x]:nw_minor_axis_outflow_dx + nw_minor_axis_outflow_90pc_halfwidth[x] + 1])

for x in range(len(se_minor_axis_outflow_dy)):
    se_minor_axis_outflow_total_flux_50pc += np.nansum(hbetasingleflux[se_minor_axis_outflow_dy[x], se_minor_axis_outflow_dx - se_minor_axis_outflow_50pc_halfwidth[x]:se_minor_axis_outflow_dx + se_minor_axis_outflow_50pc_halfwidth[x] + 1])
    se_minor_axis_outflow_total_flux_90pc += np.nansum(hbetasingleflux[se_minor_axis_outflow_dy[x], se_minor_axis_outflow_dx - se_minor_axis_outflow_90pc_halfwidth[x]:se_minor_axis_outflow_dx + se_minor_axis_outflow_90pc_halfwidth[x] + 1])

nw_outflow_numspax_50pc = np.nansum(nw_minor_axis_outflow_50pc_halfwidth) * 2
se_outflow_numspax_50pc = np.nansum(se_minor_axis_outflow_50pc_halfwidth) * 2

print(f"Total NW Minor Axis Outflow Flux (50%): {nw_minor_axis_outflow_total_flux_50pc}")
print(f"Total NW Minor Axis Outflow Flux (90%): {nw_minor_axis_outflow_total_flux_90pc}")
print(f"Total SE Minor Axis Outflow Flux (50%): {se_minor_axis_outflow_total_flux_50pc}")
print(f"Total SE Minor Axis Outflow Flux (90%): {se_minor_axis_outflow_total_flux_90pc}")

ne = 32                         # Electron density in the outflow in cm^-3
mrk1486distance = 149.3 * 3.08568 * 10 ** 24 # Luminosity Distance to MRK1486 in cm
hbetaemissivity = 1.24 * 10 ** (-25)    # HBeta emissivity at 10^4 K in erg cm^3 s^-1
mh = 1.67 * 10 ** (-27)         # Mass of hydrogen in kg
kgpersolmass = 2 * 10 ** 30         # kg per solar mass
cmperkpc = 3.08568 * 10 ** 21 # cm per kpc
kpcperspaxel = 0.1976 # kpc per spaxel
cmperspaxel = cmperkpc * kpcperspaxel
vout_Heckman = 30000000 # cm s^-1 
rout_Heckman = 2 * cmperkpc # cm
spyear = 3.156 * 10 ** 7        # seconds per year

neprofvec_const = np.full(shape[1], np.nan)
neprofvec_neg1 = np.full(shape[1], np.nan)
neprofvec_neg2 = np.full(shape[1], np.nan)
rvec = np.full(shape[1], np.nan)

# hbetasingleflux

for x in xpix:
    rvec[x] = abs(x - center[0]) * kpcperspaxel
    neprofvec_const[x] = neprofile(0, rvec[x])
    neprofvec_neg1[x] = neprofile(-1, rvec[x])
    neprofvec_neg2[x] = neprofile(-2, rvec[x])

nw_outflow_50_mass_outflow_our_calculation = 0
se_outflow_50_mass_outflow_our_calculation = 0
nw_outflow_50_mass = 0
se_outflow_50_mass = 0

nw_outflow_90_mass_outflow_our_calculation_const_ne = 0
nw_outflow_90_mass_outflow_our_calculation_neg1_ne = 0
nw_outflow_90_mass_outflow_our_calculation_neg2_ne = 0
se_outflow_90_mass_outflow_our_calculation_const_ne = 0
se_outflow_90_mass_outflow_our_calculation_neg1_ne = 0
se_outflow_90_mass_outflow_our_calculation_neg2_ne = 0
nw_outflow_90_mass = 0
se_outflow_90_mass = 0

for x in range(len(nw_minor_axis_outflow_dy)):
    nw_minor_axis_outflow_50pc_flux_strip = np.nansum(hbetasingleflux[nw_minor_axis_outflow_dy[x], nw_minor_axis_outflow_dx - nw_minor_axis_outflow_50pc_halfwidth[x]:nw_minor_axis_outflow_dx + nw_minor_axis_outflow_50pc_halfwidth[x] + 1])
    nw_minor_axis_outflow_50pc_lum_strip = nw_minor_axis_outflow_50pc_flux_strip * (10 ** (-16)) * 4 * pi * mrk1486distance ** 2 # erg s^-1
    nw_minor_axis_outflow_50pc_mass_strip = (((1.36 * mh)/(hbetaemissivity * neprofvec_neg1[nw_minor_axis_outflow_dy[x]])) * nw_minor_axis_outflow_50pc_lum_strip) # kg

    nw_minor_axis_outflow_50pc_mass_outflow_strip = (vout_Heckman * nw_minor_axis_outflow_50pc_mass_strip / (abs(nw_minor_axis_outflow_dy[x] - center[0]) * cmperspaxel)) * (spyear / kgpersolmass) 
    nw_outflow_50_mass_outflow_our_calculation += nw_minor_axis_outflow_50pc_mass_outflow_strip 
    nw_outflow_50_mass += nw_minor_axis_outflow_50pc_mass_strip

for x in range(len(se_minor_axis_outflow_dy)):
    se_minor_axis_outflow_50pc_flux_strip = np.nansum(hbetasingleflux[se_minor_axis_outflow_dy[x], se_minor_axis_outflow_dx - se_minor_axis_outflow_50pc_halfwidth[x]:se_minor_axis_outflow_dx + se_minor_axis_outflow_50pc_halfwidth[x] + 1])
    se_minor_axis_outflow_50pc_lum_strip = se_minor_axis_outflow_50pc_flux_strip * (10 ** (-16)) * 4 * pi * mrk1486distance ** 2 # erg s^-1
    se_minor_axis_outflow_50pc_mass_strip = (((1.36 * mh)/(hbetaemissivity * neprofvec_neg1[se_minor_axis_outflow_dy[x]])) * se_minor_axis_outflow_50pc_lum_strip) # kg

    se_minor_axis_outflow_50pc_mass_outflow_strip = (vout_Heckman * se_minor_axis_outflow_50pc_mass_strip / (abs(se_minor_axis_outflow_dy[x] - center[0]) * cmperspaxel)) * (spyear / kgpersolmass) 
    se_outflow_50_mass_outflow_our_calculation += se_minor_axis_outflow_50pc_mass_outflow_strip 
    se_outflow_50_mass += se_minor_axis_outflow_50pc_mass_strip

for x in range(len(nw_minor_axis_outflow_dy)):
    nw_minor_axis_outflow_90pc_flux_strip = np.nansum(hbetasingleflux[nw_minor_axis_outflow_dy[x], nw_minor_axis_outflow_dx - nw_minor_axis_outflow_90pc_halfwidth[x]:nw_minor_axis_outflow_dx + nw_minor_axis_outflow_90pc_halfwidth[x] + 1])
    nw_minor_axis_outflow_90pc_lum_strip = nw_minor_axis_outflow_90pc_flux_strip * (10 ** (-16)) * 4 * pi * mrk1486distance ** 2 # erg s^-1
    nw_minor_axis_outflow_90pc_mass_strip_const_ne = (((1.36 * mh)/(hbetaemissivity * neprofvec_const[nw_minor_axis_outflow_dy[x]])) * nw_minor_axis_outflow_90pc_lum_strip) # kg
    nw_minor_axis_outflow_90pc_mass_strip_neg1_ne = (((1.36 * mh)/(hbetaemissivity * neprofvec_neg1[nw_minor_axis_outflow_dy[x]])) * nw_minor_axis_outflow_90pc_lum_strip) # kg
    nw_minor_axis_outflow_90pc_mass_strip_neg2_ne = (((1.36 * mh)/(hbetaemissivity * neprofvec_neg2[nw_minor_axis_outflow_dy[x]])) * nw_minor_axis_outflow_90pc_lum_strip) # kg

    nw_minor_axis_outflow_90pc_mass_outflow_strip_const_ne = (vout_Heckman * nw_minor_axis_outflow_90pc_mass_strip_const_ne / (abs(nw_minor_axis_outflow_dy[x] - center[0]) * cmperspaxel)) * (spyear / kgpersolmass) 
    nw_minor_axis_outflow_90pc_mass_outflow_strip_neg1_ne = (vout_Heckman * nw_minor_axis_outflow_90pc_mass_strip_neg1_ne / (abs(nw_minor_axis_outflow_dy[x] - center[0]) * cmperspaxel)) * (spyear / kgpersolmass) 
    nw_minor_axis_outflow_90pc_mass_outflow_strip_neg2_ne = (vout_Heckman * nw_minor_axis_outflow_90pc_mass_strip_neg2_ne / (abs(nw_minor_axis_outflow_dy[x] - center[0]) * cmperspaxel)) * (spyear / kgpersolmass) 
    nw_outflow_90_mass_outflow_our_calculation_const_ne += nw_minor_axis_outflow_90pc_mass_outflow_strip_const_ne
    nw_outflow_90_mass_outflow_our_calculation_neg1_ne += nw_minor_axis_outflow_90pc_mass_outflow_strip_neg1_ne
    nw_outflow_90_mass_outflow_our_calculation_neg2_ne += nw_minor_axis_outflow_90pc_mass_outflow_strip_neg2_ne
    nw_outflow_90_mass += nw_minor_axis_outflow_90pc_mass_strip_neg1_ne

for x in range(len(se_minor_axis_outflow_dy)):
    se_minor_axis_outflow_90pc_flux_strip = np.nansum(hbetasingleflux[se_minor_axis_outflow_dy[x], se_minor_axis_outflow_dx - se_minor_axis_outflow_90pc_halfwidth[x]:se_minor_axis_outflow_dx + se_minor_axis_outflow_90pc_halfwidth[x] + 1])
    se_minor_axis_outflow_90pc_lum_strip = se_minor_axis_outflow_90pc_flux_strip * (10 ** (-16)) * 4 * pi * mrk1486distance ** 2 # erg s^-1
    se_minor_axis_outflow_90pc_mass_strip_const_ne = (((1.36 * mh)/(hbetaemissivity * neprofvec_const[se_minor_axis_outflow_dy[x]])) * se_minor_axis_outflow_90pc_lum_strip) # kg
    se_minor_axis_outflow_90pc_mass_strip_neg1_ne = (((1.36 * mh)/(hbetaemissivity * neprofvec_neg1[se_minor_axis_outflow_dy[x]])) * se_minor_axis_outflow_90pc_lum_strip) # kg
    se_minor_axis_outflow_90pc_mass_strip_neg2_ne = (((1.36 * mh)/(hbetaemissivity * neprofvec_neg2[se_minor_axis_outflow_dy[x]])) * se_minor_axis_outflow_90pc_lum_strip) # kg

    se_minor_axis_outflow_90pc_mass_outflow_strip_const_ne = (vout_Heckman * se_minor_axis_outflow_90pc_mass_strip_const_ne / (abs(se_minor_axis_outflow_dy[x] - center[0]) * cmperspaxel)) * (spyear / kgpersolmass) 
    se_minor_axis_outflow_90pc_mass_outflow_strip_neg1_ne = (vout_Heckman * se_minor_axis_outflow_90pc_mass_strip_neg1_ne / (abs(se_minor_axis_outflow_dy[x] - center[0]) * cmperspaxel)) * (spyear / kgpersolmass) 
    se_minor_axis_outflow_90pc_mass_outflow_strip_neg2_ne = (vout_Heckman * se_minor_axis_outflow_90pc_mass_strip_neg2_ne / (abs(se_minor_axis_outflow_dy[x] - center[0]) * cmperspaxel)) * (spyear / kgpersolmass) 
    se_outflow_90_mass_outflow_our_calculation_const_ne += se_minor_axis_outflow_90pc_mass_outflow_strip_const_ne
    se_outflow_90_mass_outflow_our_calculation_neg1_ne += se_minor_axis_outflow_90pc_mass_outflow_strip_neg1_ne
    se_outflow_90_mass_outflow_our_calculation_neg2_ne += se_minor_axis_outflow_90pc_mass_outflow_strip_neg2_ne
    se_outflow_90_mass += se_minor_axis_outflow_90pc_mass_strip_neg1_ne

nw_outflow_50_number_of_spaxels = np.nansum(nw_minor_axis_outflow_50pc_halfwidth) * 2
nw_outflow_50_area_cm_2 = nw_outflow_50_number_of_spaxels * cmperspaxel ** 2 # cm^-2

se_outflow_50_number_of_spaxels = np.nansum(se_minor_axis_outflow_50pc_halfwidth) * 2
se_outflow_50_area_cm_2 = se_outflow_50_number_of_spaxels * cmperspaxel ** 2 # cm^-2

nw_outflow_90_number_of_spaxels = np.nansum(nw_minor_axis_outflow_90pc_halfwidth) * 2
nw_outflow_90_area_cm_2 = nw_outflow_90_number_of_spaxels * cmperspaxel ** 2 # cm^-2

se_outflow_90_number_of_spaxels = np.nansum(se_minor_axis_outflow_90pc_halfwidth) * 2
se_outflow_90_area_cm_2 = se_outflow_90_number_of_spaxels * cmperspaxel ** 2 # cm^-2

nw_outflow_50_mass_density = nw_outflow_50_mass / nw_outflow_50_area_cm_2 # kg cm^-2
nw_outflow_50_mass_outflow_Heckman_calculation = (4 * pi * nw_outflow_50_mass_density * vout_Heckman * rout_Heckman / kgpersolmass) * spyear # Msol yr^-1

se_outflow_50_mass_density = se_outflow_50_mass / se_outflow_50_area_cm_2 # kg cm^-2
se_outflow_50_mass_outflow_Heckman_calculation = (4 * pi * se_outflow_50_mass_density * vout_Heckman * rout_Heckman / kgpersolmass) * spyear # Msol yr^-1

nw_outflow_90_mass_density = nw_outflow_90_mass / nw_outflow_90_area_cm_2 # kg cm^-2
nw_outflow_90_mass_outflow_Heckman_calculation = (4 * pi * nw_outflow_90_mass_density * vout_Heckman * rout_Heckman / kgpersolmass) * spyear # Msol yr^-1

se_outflow_90_mass_density = se_outflow_90_mass / se_outflow_90_area_cm_2 # kg cm^-2
se_outflow_90_mass_outflow_Heckman_calculation = (4 * pi * se_outflow_90_mass_density * vout_Heckman * rout_Heckman / kgpersolmass) * spyear # Msol yr^-1

total_disk_edge_outflow_flux = np.nansum(outflux)
total_disk_outflow_spaxnum = disk_outnumber_spaxels
total_disk_outflow_area = total_disk_outflow_spaxnum * cmperspaxel ** 2 # cm^-2
total_disk_outflow_lum = total_disk_edge_outflow_flux * (10 ** (-16)) * 4 * pi * mrk1486distance ** 2 # erg s^-1
total_disk_outflow_mass = (((1.36 * mh)/(hbetaemissivity * ne)) * total_disk_outflow_lum) # kg
total_disk_outflow_mass_density = total_disk_outflow_mass / total_disk_outflow_area # kg cm^-2
total_disk_outflow_column_density = total_disk_outflow_mass_density / (1.36 * mh) # cm^-2
total_disk_outflow_mass_outflow_rate_Heckman = (4 * pi * total_disk_outflow_mass_density * vout_Heckman * rout_Heckman / kgpersolmass) * spyear # Msol s^-1

outlum = np.full([shape[1], shape[2]], np.nan)
systlum = np.full([shape[1], shape[2]], np.nan)
outmass = np.full([shape[1], shape[2]], np.nan)
outmdot = np.full([shape[1], shape[2]], np.nan)
diskrout = 2

for x in np.arange(galaxyrow[0], galaxyrow[1] + 1, 1):
    for y in np.arange(galaxycol[0], galaxycol[1] + 1, 1):
        outlum[x, y] = outflux[x, y] * (10 ** (-16)) * 4 * pi * mrk1486distance ** 2
        systlum[x, y] = systflux[x, y] * (10 ** (-16)) * 4 * pi * mrk1486distance ** 2
        outmass[x, y] = (((1.36 * mh)/(hbetaemissivity * ne)) * outlum[x, y]) / kgpersolmass
        outmdot[x, y] = ((outvel[x, y] * outmass[x, y]) / (diskrout * 3.086 * 10**16)) * spyear
        if outmdot[x, y] == 0:
            outmdot[x, y] = np.nan
            outvel[x, y] = np.nan
            outflux[x, y] = np.nan
total_disk_outflow_mass_outflow = np.nansum(outmdot)

disk_radius = cmperspaxel * 20
total_disk_face_area = pi * disk_radius ** 2

print(f"Total outflow area (NW 50%): {nw_outflow_50_area_cm_2} cm^-2")
print(f"Total outflow area (SE 50%): {se_outflow_50_area_cm_2} cm^-2")
print(f"Total outflow area (NW 90%): {nw_outflow_90_area_cm_2} cm^-2")
print(f"Total otflow area (SE 90%): {se_outflow_90_area_cm_2} cm^-2")
print(f"Total outflow area (Disk): {total_disk_outflow_area} cm^-2")
print(f"Total disk edge area: {10 * 40 * cmperspaxel ** 2} cm^{-2}")
print(f"Total galaxy disk face area: {total_disk_face_area} cm^-2")

print(f"Total outflow mass (NW 50%): {nw_outflow_50_mass/kgpersolmass} MSol")
print(f"Total outflow mass (SE 50%): {se_outflow_50_mass/kgpersolmass} MSol")
print(f"Total outflow mass (NW 90%): {nw_outflow_90_mass/kgpersolmass} MSol")
print(f"Total outflow mass (SE 90%): {se_outflow_90_mass/kgpersolmass} MSol")
print(f"Total outflow mass (Disk): {total_disk_outflow_mass/kgpersolmass} MSol")

print(f"Total galaxy mass outflow rate (Heckman 2015, NW 50%): {nw_outflow_50_mass_outflow_Heckman_calculation} Msol yr^-1")
print(f"Total galaxy mass outflow rate (Heckman 2015, SE 50%): {se_outflow_50_mass_outflow_Heckman_calculation} Msol yr^-1")
print(f"Total galaxy mass outflow rate (Heckman 2015, NW 90%): {nw_outflow_90_mass_outflow_Heckman_calculation} Msol yr^-1")
print(f"Total galaxy mass outflow rate (Heckman 2015, SE 90%): {se_outflow_90_mass_outflow_Heckman_calculation} Msol yr^-1")
print(f"Total galaxy mass outflow rate (Heckman 2015, Disk): {total_disk_outflow_mass_outflow_rate_Heckman} Msol yr^-1")

print(f"NW 50% mass outflow rate (Our calculation): {nw_outflow_50_mass_outflow_our_calculation} Msol yr^-1")
print(f"SE 50% mass outflow rate (Our calculation): {se_outflow_50_mass_outflow_our_calculation} Msol yr^-1")
print(f"NW 90% mass outflow rate (Our calculation): {nw_outflow_90_mass_outflow_our_calculation_neg1_ne} Msol yr^-1")
print(f"SE 90% mass outflow rate (Our calculation): {se_outflow_90_mass_outflow_our_calculation_neg1_ne} Msol yr^-1")
print(f"Disk mass outfow rate (Our calculation): {total_disk_outflow_mass_outflow} Msol yr^-1")

print("APPENDIX A.1")

print(f"NW 90% mass outflow rate (Our calculation, alpha=0): {nw_outflow_90_mass_outflow_our_calculation_const_ne} Msol yr^-1")
print(f"NW 90% mass outflow rate (Our calculation, alpha=-1): {nw_outflow_90_mass_outflow_our_calculation_neg1_ne} Msol yr^-1")
print(f"NW 90% mass outflow rate (Our calculation, alpha=-2): {nw_outflow_90_mass_outflow_our_calculation_neg2_ne} Msol yr^-1")
print(f"SE 90% mass outflow rate (Our calculation, alpha=0): {se_outflow_90_mass_outflow_our_calculation_const_ne} Msol yr^-1")
print(f"SE 90% mass outflow rate (Our calculation, alpha=-1): {se_outflow_90_mass_outflow_our_calculation_neg1_ne} Msol yr^-1")
print(f"SE 90% mass outflow rate (Our calculation, alpha=-2): {se_outflow_90_mass_outflow_our_calculation_neg2_ne} Msol yr^-1")

print("APPENDIX A.2")

vscaleheights = np.arange(0, 3.5, 0.5)

for vscaleheight in vscaleheights:

    voutprofvec = np.full(shape[1], np.nan)
    for x in xpix:
        rvec[x] = abs(x - center[0]) * kpcperspaxel
        voutprofvec[x] = vprofile(vscaleheight, rvec[x], vout_Heckman)
    
    nw_outflow_90_mass_outflow_our_calculation_vout = 0
    
    for x in range(len(nw_minor_axis_outflow_dy)):
        nw_minor_axis_outflow_90pc_flux_strip_vout = np.nansum(hbetasingleflux[nw_minor_axis_outflow_dy[x], nw_minor_axis_outflow_dx - nw_minor_axis_outflow_90pc_halfwidth[x]:nw_minor_axis_outflow_dx + nw_minor_axis_outflow_90pc_halfwidth[x] + 1])
        nw_minor_axis_outflow_90pc_lum_strip_vout = nw_minor_axis_outflow_90pc_flux_strip_vout * (10 ** (-16)) * 4 * pi * mrk1486distance ** 2 # erg s^-1
        nw_minor_axis_outflow_90pc_mass_strip_vout = (((1.36 * mh)/(hbetaemissivity * neprofvec_neg1[nw_minor_axis_outflow_dy[x]])) * nw_minor_axis_outflow_90pc_lum_strip_vout) # kg
        nw_minor_axis_outflow_90pc_mass_outflow_strip_vout = (voutprofvec[x] * nw_minor_axis_outflow_90pc_mass_strip_vout / (abs(nw_minor_axis_outflow_dy[x] - center[0]) * cmperspaxel)) * (spyear / kgpersolmass) 
        nw_outflow_90_mass_outflow_our_calculation_vout += nw_minor_axis_outflow_90pc_mass_outflow_strip_vout

    print(f"NW 90% mass outflow rate (Our calculation, velocity scale height={vscaleheight}): {nw_outflow_90_mass_outflow_our_calculation_vout} Msol yr^-1")

for vscaleheight in vscaleheights:
    
    voutprofvec = np.full(shape[1], np.nan)
    for x in xpix:
        rvec[x] = abs(x - center[0]) * kpcperspaxel
        voutprofvec[x] = vprofile(vscaleheight, rvec[x], vout_Heckman)

    se_outflow_90_mass_outflow_our_calculation_vout = 0

    for x in range(len(se_minor_axis_outflow_dy)):
        se_minor_axis_outflow_90pc_flux_strip_vout = np.nansum(hbetasingleflux[se_minor_axis_outflow_dy[x], se_minor_axis_outflow_dx - se_minor_axis_outflow_90pc_halfwidth[x]:se_minor_axis_outflow_dx + se_minor_axis_outflow_90pc_halfwidth[x] + 1])
        se_minor_axis_outflow_90pc_lum_strip_vout = se_minor_axis_outflow_90pc_flux_strip_vout * (10 ** (-16)) * 4 * pi * mrk1486distance ** 2 # erg s^-1
        se_minor_axis_outflow_90pc_mass_strip_vout = (((1.36 * mh)/(hbetaemissivity * neprofvec_neg1[se_minor_axis_outflow_dy[x]])) * se_minor_axis_outflow_90pc_lum_strip_vout) # kg
        se_minor_axis_outflow_90pc_mass_outflow_strip_vout = (voutprofvec[x] * se_minor_axis_outflow_90pc_mass_strip_vout / (abs(se_minor_axis_outflow_dy[x] - center[0]) * cmperspaxel)) * (spyear / kgpersolmass) 
        se_outflow_90_mass_outflow_our_calculation_vout += se_minor_axis_outflow_90pc_mass_outflow_strip_vout

    print(f"SE 90% mass outflow rate (Our calculation, velocity scale height={vscaleheight}): {se_outflow_90_mass_outflow_our_calculation_vout} Msol yr^-1")

for x in xpix[2:-2]:
    for y in ypix[2:-2]:
        meansingsig[x, y] = np.nanmean(oiii5007singlesigma[x - 2:x + 2, y - 2:y + 2])
        c23meanfluxrat[x, y] = np.nanmean(c23fluxrat[x - 2:x + 2, y - 2:y + 2])
        meanlineseparation[x, y] = np.nanmean(lineseparation[x - 2:x + 2, y - 2:y + 2])
        meanlineseparationerror[x, y] = np.nanmean(lineseparationerror[x - 2:x + 2, y - 2:y + 2])/16
        oiii5007medianave[x, y] = np.nanmean(oiii5007peak[x - 2:x + 2, y])
        oiii5007singlemeanave[x, y] = np.nanmean(oiii5007singlemean[x - 2:x + 2, y])
        meanoiii5007totalflux[x, y] = np.nanmean(oiii5007totalflux[x - 2:x + 2, y - 2:y + 2])

galaxyrow = (30, 39)
galaxycol = (41, 80)

newcenter = center[1] - 17

newcenter = center[1] - 27
scalednewcenter = int(newcenter * 7.36)
scaledcenter0 = int(center[0] * 7.36)
scaledsparcsec = sparcsec * 7.36

scaling = 7.36

# Fig. 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20))
hstimg = ax1.imshow(img[1198:1713, 1220:1684])
ax1.plot(outflow_prof_dx_spaxels[:29] * scaling, outflow_prof_dy_spaxels_hst[:29] * scaling, color='k', linewidth=5)
ax1.plot(outflow_prof_dx_spaxels[:29] * scaling - outflow_prof_50pc_halfwidth[:29] * scaling, outflow_prof_dy_spaxels_hst[:29] * scaling, color='r', linewidth=5, linestyle='--')
ax1.plot(outflow_prof_dx_spaxels[:29] * scaling + outflow_prof_50pc_halfwidth[:29] * scaling, outflow_prof_dy_spaxels_hst[:29] * scaling, color='r', linewidth=5, linestyle='--')
ax1.plot(outflow_prof_dx_spaxels[:29] * scaling - outflow_prof_90pc_halfwidth[:29] * scaling, outflow_prof_dy_spaxels_hst[:29] * scaling, color='r', linewidth=5)
ax1.plot(outflow_prof_dx_spaxels[:29] * scaling + outflow_prof_90pc_halfwidth[:29] * scaling, outflow_prof_dy_spaxels_hst[:29] * scaling, color='r', linewidth=5)
ax1.plot(outflow_prof_dx_spaxels[29:] * scaling, outflow_prof_dy_spaxels_hst[29:] * scaling, color='k', linewidth=5)
ax1.plot(outflow_prof_dx_spaxels[29:] * scaling - outflow_prof_50pc_halfwidth[29:] * scaling, outflow_prof_dy_spaxels_hst[29:] * scaling, color='r', linewidth=5, linestyle='--')
ax1.plot(outflow_prof_dx_spaxels[29:] * scaling + outflow_prof_50pc_halfwidth[29:] * scaling, outflow_prof_dy_spaxels_hst[29:] * scaling, color='r', linewidth=5, linestyle='--')
ax1.plot(outflow_prof_dx_spaxels[29:] * scaling - outflow_prof_90pc_halfwidth[29:] * scaling, outflow_prof_dy_spaxels_hst[29:] * scaling, color='r', linewidth=5)
ax1.plot(outflow_prof_dx_spaxels[29:] * scaling + outflow_prof_90pc_halfwidth[29:] * scaling, outflow_prof_dy_spaxels_hst[29:] * scaling, color='r', linewidth=5)
# divider = make_axes_locatable(ax1)
# cax1 = divider.append_axes('right', size='5%', pad=0.05)
# cb = fig.colorbar(sigimg, cax=cax1, orientation='vertical')
# cb.set_label(label='log$_{10}$(Flux) (* 10$^{-16}$ erg s$^{-1}$ cm$^{-2}$)', size=30)
# cb.ax.tick_params(labelsize=25)
ax1.set_xlabel('Major axis offset (arcsec)', fontsize=30)
ax1.set_ylabel('Minor axis offset (arcsec)', fontsize=30)
ax1.set_xticks([scalednewcenter - 9*scaledsparcsec, scalednewcenter - 6*scaledsparcsec, scalednewcenter - 3*scaledsparcsec, scalednewcenter, scalednewcenter + 3*scaledsparcsec, scalednewcenter + 6*scaledsparcsec])
ax1.set_xticklabels(['-9', '-6', '-3', '0', '3', '6'], fontsize=25)
ax1.set_yticks([scaledcenter0 - 9*scaledsparcsec, scaledcenter0 - 6*scaledsparcsec, scaledcenter0 - 3*scaledsparcsec, scaledcenter0, scaledcenter0 + 3*scaledsparcsec, scaledcenter0 + 6*scaledsparcsec, scaledcenter0 + 9*scaledsparcsec])
ax1.set_yticklabels(['9', '6', '3', '0', '-3', '-6', '-9'], fontsize=25)
# ax1.invert_yaxis()

fluximg = ax2.imshow(np.log10(meanoiii5007totalflux[:, 27:]), cmap='Blues_r')
ax2.plot(outflow_prof_dx_spaxels[:29], outflow_prof_dy_spaxels[:29], color='k', linewidth=5)
ax2.plot(outflow_prof_dx_spaxels[:29] - outflow_prof_50pc_halfwidth[:29], outflow_prof_dy_spaxels[:29], color='r', linewidth=5, linestyle='--')
ax2.plot(outflow_prof_dx_spaxels[:29] + outflow_prof_50pc_halfwidth[:29], outflow_prof_dy_spaxels[:29], color='r', linewidth=5, linestyle='--')
ax2.plot(outflow_prof_dx_spaxels[:29] - outflow_prof_90pc_halfwidth[:29], outflow_prof_dy_spaxels[:29], color='r', linewidth=5)
ax2.plot(outflow_prof_dx_spaxels[:29] + outflow_prof_90pc_halfwidth[:29], outflow_prof_dy_spaxels[:29], color='r', linewidth=5)
ax2.plot(outflow_prof_dx_spaxels[29:], outflow_prof_dy_spaxels[29:], color='k', linewidth=5)
ax2.plot(outflow_prof_dx_spaxels[29:] - outflow_prof_50pc_halfwidth[29:], outflow_prof_dy_spaxels[29:], color='r', linewidth=5, linestyle='--')
ax2.plot(outflow_prof_dx_spaxels[29:] + outflow_prof_50pc_halfwidth[29:], outflow_prof_dy_spaxels[29:], color='r', linewidth=5, linestyle='--')
ax2.plot(outflow_prof_dx_spaxels[29:] - outflow_prof_90pc_halfwidth[29:], outflow_prof_dy_spaxels[29:], color='r', linewidth=5)
ax2.plot(outflow_prof_dx_spaxels[29:] + outflow_prof_90pc_halfwidth[29:], outflow_prof_dy_spaxels[29:], color='r', linewidth=5)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(fluximg, cax=cax2, orientation='vertical')
cb.set_label(label='log10(Flux) (x 10$^{-16}$ erg s$^{-1}$ cm$^{-2}$)', size=30)
cb.ax.tick_params(labelsize=25)
ax2.set_xlabel('Major axis offset (arcsec)', fontsize=30)
ax2.set_ylabel('Minor axis offset (arcsec)', fontsize=30)
ax2.set_xticks([newcenter - 9*sparcsec, newcenter - 6*sparcsec, newcenter - 3*sparcsec, newcenter, newcenter + 3*sparcsec, newcenter + 6*sparcsec])
ax2.set_xticklabels(['-9', '-6', '-3', '0', '3', '6'], fontsize=25)
ax2.set_yticks([center[0] - 9*sparcsec, center[0] - 6*sparcsec, center[0] - 3*sparcsec, center[0], center[0] + 3*sparcsec, center[0] + 6*sparcsec, center[0] + 9*sparcsec])
ax2.set_yticklabels(['-9', '-6', '-3', '0', '3', '6', '9'], fontsize=25)
ax2.invert_yaxis()

# fig.tight_layout()
# plt.subplots_adjust(wspace=0.5)
plt.savefig(gal + '/' + filename_preamble + '_outflow_region.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# Fig. 3

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))
dispimg = ax1.imshow(meansingsig[:, 27:], cmap='Blues_r', vmax=150)
ax1.plot(outflow_prof_dx_spaxels[:29], outflow_prof_dy_spaxels[:29], color='k', linewidth=5)
ax1.plot(outflow_prof_dx_spaxels[:29] - outflow_prof_50pc_halfwidth[:29], outflow_prof_dy_spaxels[:29], color='r', linewidth=5, linestyle='--')
ax1.plot(outflow_prof_dx_spaxels[:29] + outflow_prof_50pc_halfwidth[:29], outflow_prof_dy_spaxels[:29], color='r', linewidth=5, linestyle='--')
ax1.plot(outflow_prof_dx_spaxels[:29] - outflow_prof_90pc_halfwidth[:29], outflow_prof_dy_spaxels[:29], color='r', linewidth=5)
ax1.plot(outflow_prof_dx_spaxels[:29] + outflow_prof_90pc_halfwidth[:29], outflow_prof_dy_spaxels[:29], color='r', linewidth=5)
ax1.plot(outflow_prof_dx_spaxels[29:], outflow_prof_dy_spaxels[29:], color='k', linewidth=5)
ax1.plot(outflow_prof_dx_spaxels[29:] - outflow_prof_50pc_halfwidth[29:], outflow_prof_dy_spaxels[29:], color='r', linewidth=5, linestyle='--')
ax1.plot(outflow_prof_dx_spaxels[29:] + outflow_prof_50pc_halfwidth[29:], outflow_prof_dy_spaxels[29:], color='r', linewidth=5, linestyle='--')
ax1.plot(outflow_prof_dx_spaxels[29:] - outflow_prof_90pc_halfwidth[29:], outflow_prof_dy_spaxels[29:], color='r', linewidth=5)
ax1.plot(outflow_prof_dx_spaxels[29:] + outflow_prof_90pc_halfwidth[29:], outflow_prof_dy_spaxels[29:], color='r', linewidth=5)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(dispimg, cax=cax1, orientation='vertical')
cb.set_label(label='Single Gaussian $\sigma$ (km s$^{-1}$)', size=30)
cb.ax.tick_params(labelsize=25)
ax1.set_xlabel('Major axis offset (arcsec)', fontsize=30)
ax1.set_ylabel('Minor axis offset (arcsec)', fontsize=30)
ax1.set_xticks([newcenter - 9*sparcsec, newcenter - 6*sparcsec, newcenter - 3*sparcsec, newcenter, newcenter + 3*sparcsec, newcenter + 6*sparcsec])
ax1.set_xticklabels(['-9', '-6', '-3', '0', '3', '6'], fontsize=25)
ax1.set_yticks([center[0] - 9*sparcsec, center[0] - 6*sparcsec, center[0] - 3*sparcsec, center[0], center[0] + 3*sparcsec, center[0] + 6*sparcsec, center[0] + 9*sparcsec])
ax1.set_yticklabels(['-9', '-6', '-3', '0', '3', '6', '9'], fontsize=25)
ax1.invert_yaxis()

sigimg = ax2.imshow(c23meanfluxrat[:, 27:], cmap='Blues_r')
ax2.plot(outflow_prof_dx_spaxels[:29], outflow_prof_dy_spaxels[:29], color='k', linewidth=5)
ax2.plot(outflow_prof_dx_spaxels[:29] - outflow_prof_50pc_halfwidth[:29], outflow_prof_dy_spaxels[:29], color='r', linewidth=5, linestyle='--')
ax2.plot(outflow_prof_dx_spaxels[:29] + outflow_prof_50pc_halfwidth[:29], outflow_prof_dy_spaxels[:29], color='r', linewidth=5, linestyle='--')
ax2.plot(outflow_prof_dx_spaxels[:29] - outflow_prof_90pc_halfwidth[:29], outflow_prof_dy_spaxels[:29], color='r', linewidth=5)
ax2.plot(outflow_prof_dx_spaxels[:29] + outflow_prof_90pc_halfwidth[:29], outflow_prof_dy_spaxels[:29], color='r', linewidth=5)
ax2.plot(outflow_prof_dx_spaxels[29:], outflow_prof_dy_spaxels[29:], color='k', linewidth=5)
ax2.plot(outflow_prof_dx_spaxels[29:] - outflow_prof_50pc_halfwidth[29:], outflow_prof_dy_spaxels[29:], color='r', linewidth=5, linestyle='--')
ax2.plot(outflow_prof_dx_spaxels[29:] + outflow_prof_50pc_halfwidth[29:], outflow_prof_dy_spaxels[29:], color='r', linewidth=5, linestyle='--')
ax2.plot(outflow_prof_dx_spaxels[29:] - outflow_prof_90pc_halfwidth[29:], outflow_prof_dy_spaxels[29:], color='r', linewidth=5)
ax2.plot(outflow_prof_dx_spaxels[29:] + outflow_prof_90pc_halfwidth[29:], outflow_prof_dy_spaxels[29:], color='r', linewidth=5)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(sigimg, cax=cax2, orientation='vertical')
cb.set_label(label='(Flux$_{C2}$ + Flux$_{C3}$) / Total Flux', size=30)
cb.ax.tick_params(labelsize=25)
# contours = ax1.contour(meansingsig[:, 17:], levels=[60, 65, 70, 80, 90, 100], cmap='Reds')
ax2.set_xlabel('Major axis offset (arcsec)', fontsize=30)
ax2.set_ylabel('Minor axis offset (arcsec)', fontsize=30)
ax2.set_xticks([newcenter - 9*sparcsec, newcenter - 6*sparcsec, newcenter - 3*sparcsec, newcenter, newcenter + 3*sparcsec, newcenter + 6*sparcsec])
ax2.set_xticklabels(['-9', '-6', '-3', '0', '3', '6'], fontsize=25)
ax2.set_yticks([center[0] - 9*sparcsec, center[0] - 6*sparcsec, center[0] - 3*sparcsec, center[0], center[0] + 3*sparcsec, center[0] + 6*sparcsec, center[0] + 9*sparcsec])
ax2.set_yticklabels(['-9', '-6', '-3', '0', '3', '6', '9'], fontsize=25)
# ax1.clabel(contours, inline=True, fmt=fmt, fontsize=MEDIUM_SIZE)
ax2.invert_yaxis()
# fig.tight_layout()
# plt.subplots_adjust(wspace=0.5)
plt.savefig(gal + '/' + filename_preamble + '_line_splitting.pdf', bbox_inches='tight')
# plt.show()
plt.close()

# Table 1

# Method (Photometry, Line Splitting) | Shape Assumed (Conical, Frustrum) | Region (North, South) | Opening Angle +/- Error

kpcstep = 5

nw_dy_arcsec = outflow_prof[29:, 1]
nw_halfwidth_50 = outflow_prof[29:, 2]
nw_halfwidth_90 = outflow_prof[29:, 3]

se_dy_arcsec = np.flip(outflow_prof[:29, 1]) * -1.
se_halfwidth_50 = np.flip(outflow_prof[:29, 2])
se_halfwidth_90 = np.flip(outflow_prof[:29, 3])

delx_arcsec = 0.75
dely_arcsec = 0.35

# Photometry

# Conical

# North

nw50_outflow_fit_conical = curve_fit(slope_c0, nw_dy_arcsec, nw_halfwidth_50, [2])[0]
nw50_opening_angle_conical = (math.atan(nw50_outflow_fit_conical[0]) * 180 / math.pi) * 2

nw50_outflow_fit_conical_error = montecarloerror(slope_c0, nw_dy_arcsec, nw50_outflow_fit_conical[0], delx_arcsec, 100)
nw50_opening_angle_conical_error = np.abs(np.sqrt((2 * 180 / pi) * (1 / (1 + nw50_opening_angle_conical ** 2)) * nw50_outflow_fit_conical_error ** 2))

nw90_outflow_fit_conical = curve_fit(slope_c0, nw_dy_arcsec, nw_halfwidth_90, [2])[0]
nw90_opening_angle_conical = (math.atan(nw90_outflow_fit_conical[0]) * 180 / math.pi) * 2

nw90_outflow_fit_conical_error = montecarloerror(slope_c0, nw_dy_arcsec, nw90_outflow_fit_conical[0], delx_arcsec, 100)
nw90_opening_angle_conical_error = np.abs(np.sqrt((2 * 180 / pi) * (1 / (1 + nw90_opening_angle_conical ** 2)) * nw90_outflow_fit_conical_error ** 2))

# nw50_opening_angle_conical_error = 0.

print('Photometry, Conical, North Opening Angle: ' + str(round(nw50_opening_angle_conical, 2)) + ' +/- ' + str(round(nw50_opening_angle_conical_error, 2)))

# South

se50_outflow_fit_conical = curve_fit(slope_c0, se_dy_arcsec, se_halfwidth_50, [2])[0]
se50_opening_angle_conical = (math.atan(se50_outflow_fit_conical[0]) * 180 / math.pi) * 2

se50_outflow_fit_conical_error = montecarloerror(slope_c0, se_dy_arcsec, se50_outflow_fit_conical[0], delx_arcsec, 100)
se50_opening_angle_conical_error = np.abs(np.sqrt((2 * 180 / pi) * (1 / (1 + se50_opening_angle_conical ** 2)) * se50_outflow_fit_conical_error ** 2))

se90_outflow_fit_conical = curve_fit(slope_c0, se_dy_arcsec, se_halfwidth_90, [2])[0]
se90_opening_angle_conical = (math.atan(se90_outflow_fit_conical[0]) * 180 / math.pi) * 2

se90_outflow_fit_conical_error = montecarloerror(slope_c0, se_dy_arcsec, se90_outflow_fit_conical[0], delx_arcsec, 100)
se90_opening_angle_conical_error = np.abs(np.sqrt((2 * 180 / pi) * (1 / (1 + se90_opening_angle_conical ** 2)) * se90_outflow_fit_conical_error ** 2))

print('Photometry, Conical, South Opening Angle: ' + str(round(se50_opening_angle_conical, 2)) + ' +/- ' + str(round(se50_opening_angle_conical_error, 2)))

# Frustrum

# North

nw50_outflow_fit_frustrum = curve_fit(slope, nw_dy_arcsec, nw_halfwidth_50, [2, 1])[0]
nw50_opening_angle_frustrum = (math.atan(nw50_outflow_fit_frustrum[0]) * 180 / math.pi) * 2

nw50_outflow_fit_frustrum_error = montecarloerror(slope_c0, nw_dy_arcsec, nw50_outflow_fit_frustrum[0], delx_arcsec, 100)
nw50_opening_angle_frustrum_error = np.abs(np.sqrt((2 * 180 / pi) * (1 / (1 + nw50_opening_angle_frustrum ** 2)) * nw50_outflow_fit_frustrum_error ** 2))

nw90_outflow_fit = curve_fit(slope, nw_dy_arcsec, nw_halfwidth_90, [2, 1])[0]
nw90_opening_angle_frustrum = (math.atan(nw90_outflow_fit[0]) * 180 / math.pi) * 2

print('Photometry, Frustrum, North Opening Angle: ' + str(round(nw50_opening_angle_frustrum, 2)) + ' +/- ' + str(round(nw50_opening_angle_frustrum_error, 2)))

# South

se50_outflow_fit_frustrum = curve_fit(slope, se_dy_arcsec, se_halfwidth_50, [2, 1])[0]
se50_opening_angle_frustrum = (math.atan(se50_outflow_fit_frustrum[0]) * 180 / math.pi) * 2

se50_outflow_fit_frustrum_error = montecarloerror(slope_c0, se_dy_arcsec, se50_outflow_fit_frustrum[0], delx_arcsec, 100)
se50_opening_angle_frustrum_error = np.abs(np.sqrt((2 * 180 / pi) * (1 / (1 + se50_opening_angle_frustrum ** 2)) * se50_outflow_fit_frustrum_error ** 2))

se90_outflow_fit = curve_fit(slope, se_dy_arcsec, se_halfwidth_90, [2, 1])[0]
se90_opening_angle_frustrum = (math.atan(se90_outflow_fit[0]) * 180 / math.pi) * 2

print('Photometry, Frustrum, South Opening Angle: ' + str(round(se50_opening_angle_frustrum, 2)) + ' +/- ' + str(round(se50_opening_angle_frustrum_error, 2)))


# Line Splitting

# Conical

# North

nwlineseparation = np.nanmean(lineseparation[center[0] + 1 * kpcstep:center[0] + 4 * kpcstep, center[1] - kpcstep * 1:center[1] + kpcstep * 1 + 1])
nwopeninganglelinesplitting = 2 * np.arcsin(nwlineseparation/(2 * 300)) * 180/pi
nwlineseparationshape = np.shape(lineseparation[center[0] + 1 * kpcstep:center[0] + 4 * kpcstep, center[1] - kpcstep * 1:center[1] + kpcstep * 1 + 1])
# print(nwlineseparation)

nwlineseparationerror = np.nanmean(lineseparationerror[center[0] + 1 * kpcstep:center[0] + 4 * kpcstep, center[1] - kpcstep * 1:center[1] + kpcstep * 1 + 1])/(nwlineseparationshape[0] * nwlineseparationshape[1])
nwopeninganglelinesplittingerror = np.abs(np.sqrt(((2 * 180 / pi) ** 2 * (1 / (2 * 300)) ** 2 * nwlineseparationerror) / (1 - (nwlineseparation/(2 * 300)) ** 2)))

print('Line Splitting, Conical, North Opening Angle: ' + str(round(nwopeninganglelinesplitting, 2)) + ' +/- ' + str(round(nwopeninganglelinesplittingerror, 2)))

# South

selineseparation = np.nanmean(lineseparation[center[0] - 4 * kpcstep:center[0] - 1 * kpcstep, center[1] - kpcstep * 1:center[1] + kpcstep * 1 + 1])
seopeninganglelinesplitting = 2 * np.arcsin(selineseparation/(2 * 300)) * 180/pi
selineseparationshape = np.shape(lineseparation[center[0] - 4 * kpcstep:center[0] - 1 * kpcstep, center[1] - kpcstep * 1:center[1] + kpcstep * 1 + 1])
# print(selineseparation)

selineseparationerror = np.nanmean(lineseparation[center[0] - 4 * kpcstep:center[0] - 1 * kpcstep, center[1] - kpcstep * 1:center[1] + kpcstep * 1 + 1])/(selineseparationshape[0] * selineseparationshape[1])
seopeninganglelinesplittingerror = np.abs(np.sqrt(((2 * 180 / pi) ** 2 * (1 / (2 * 300)) ** 2 * selineseparationerror) / (1 - (selineseparation/(2 * 300)) ** 2)))

print('Line Splitting, Conical, South Opening Angle: ' + str(round(seopeninganglelinesplitting, 2)) + ' +/- ' + str(round(seopeninganglelinesplittingerror, 2)))


# Fig. 4

fig = plt.figure(figsize=(20, 12))

gs1 = GridSpec(2, 1)
gs1.update(top=0.98, bottom=0.4, hspace=0.4, wspace=0)
ax1 = plt.subplot(gs1[0, 0])
ax2 = plt.subplot(gs1[1, 0])
#
gs2 = GridSpec(1, 3)
gs2.update(top=0.3, bottom=0.05, hspace=0, wspace=0)
ax3 = plt.subplot(gs2[0, 0])
ax4 = plt.subplot(gs2[0, 1], sharey=ax3)
ax5 = plt.subplot(gs2[0, 2], sharey=ax3)

newcenter = (center[0] - galaxyrow[0], center[1] - galaxycol[0])
sparcsec = 1/0.2915

ccmap = plt.get_cmap('Blues_r')
ccmap.set_bad(color='0.8')
ax1.set_title('Outflow Flux Ratio', fontsize=30)
fratioimg = ax1.imshow(fratio[galaxyrow[0]:galaxyrow[1] + 1, galaxycol[0]:galaxycol[1] + 1], cmap=ccmap, extent=[galaxycol[0], galaxycol[1] + 1, galaxyrow[0], galaxyrow[1] + 1])
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
cb1 = fig.colorbar(fratioimg, cax=cax1, orientation='vertical')
cb1.set_label(label=r'$f_{Broad}/f_{Narrow}$', size=30)
cb1.ax.tick_params(labelsize=25)
ax1.set_xticks([center[1] - 4*sparcsec, center[1] - 2*sparcsec, center[1], center[1] + 2*sparcsec, center[1] + 4*sparcsec])
ax1.set_xticklabels(['-4', '-2', '-0', '2', '4'], fontsize=25)
ax1.set_yticks([center[0] - 1*sparcsec, center[0], center[0] + 1*sparcsec])
ax1.set_yticklabels(['-1', '-0', '1'], fontsize=25)
# ax1.set_xlabel('Major axis offset (arcsec)', fontsize=25)
ax1.set_ylabel('Minor axis offset (arcsec)', fontsize=25)
ax1.yaxis.set_label_coords(-0.05, -0.25)
cax1.invert_yaxis()
gca().invert_yaxis()

ax2.set_title('Outflow Velocity', fontsize=30)
outlumimg = ax2.imshow(outvel[galaxyrow[0]:galaxyrow[1] + 1, galaxycol[0]:galaxycol[1] + 1], cmap=ccmap, extent=[galaxycol[0], galaxycol[1] + 1, galaxyrow[0], galaxyrow[1] + 1], vmax=500)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size='5%', pad=0.05)
cb2 = fig.colorbar(outlumimg, cax=cax2, orientation='vertical')
cb2.set_label(label=r'V$_{Out}$ (km s$^{-1}$)', size=30)
cb2.ax.tick_params(labelsize=25)
ax2.set_xticks([center[1] - 4*sparcsec, center[1] - 2*sparcsec, center[1], center[1] + 2*sparcsec, center[1] + 4*sparcsec])
ax2.set_xticklabels(['-4', '-2', '-0', '2', '4'], fontsize=25)
ax2.set_yticks([center[0] - 1*sparcsec, center[0], center[0] + 1*sparcsec])
ax2.set_yticklabels(['-1', '-0', '1'], fontsize=30)
ax2.set_xlabel('Major axis offset (arcsec)', fontsize=35)
cax2.invert_yaxis()
gca().invert_yaxis()

loc1 = (38, 53)
loc2 = (35, 65)
loc3 = (31, 61)
loc1pos = (round((loc1[0] - center[0]) * 0.2915, 1), round((loc1[1] - center[1]) * 0.2915, 1))
loc2pos = (round((loc2[0] - center[0]) * 0.2915, 1), round((loc2[1] - center[1]) * 0.2915, 1))
loc3pos = (round((loc3[0] - center[0]) * 0.2915, 1), round((loc3[1] - center[1]) * 0.2915, 1))
posletters = ('A', 'B', 'C')
leftpix = 1220
rightpix = 1255
avgreddata = np.full([shape[0], shape[1], shape[2]], np.nan)
for x in range(shape[1]):
    for y in range(shape[2]):
        avgreddata[:, x, y] = np.nanmean(reddata[:, x-1:x+2, y-1:y+2], (1, 2))
for axpos in np.arange(1, 4, 1):
    eval('ax' + str(axpos + 2)).xaxis.set_minor_locator(MultipleLocator(25))
    eval('ax' + str(axpos + 2)).xaxis.set_major_locator(MultipleLocator(200))
    eval('ax' + str(axpos + 2)).tick_params(which='both', top=True, bottom=True, left=True, right=True, direction='in')
    eval('ax' + str(axpos + 2)).yaxis.set_minor_locator(MultipleLocator(0.05))
    eval('ax' + str(axpos + 2)).yaxis.set_major_locator(MultipleLocator(0.1))
    norm1 = np.nanmax(avgreddata[leftpix:rightpix, eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]])
    norm2 = hbetasingleamp[eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]]
    eval('ax' + str(axpos + 2)).scatter(hbetaderedshiftbarywave[leftpix:rightpix], (
    avgreddata[leftpix:rightpix, eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]]) / norm1)
    eval('ax' + str(axpos + 2)).plot(hbetaderedshiftbarywave[leftpix:rightpix], threegauss(hbetaderedshiftbarywave[leftpix:rightpix], hbetalinefits["avg_g1_height"][
        eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2,
                                                                        hbetalinefits["avg_g1_center"][eval('loc' + str(axpos))[0],
                                                                                           eval('loc' + str(axpos))[1]],
                                                                        hbetalinefits["avg_g1_sigma"][eval('loc' + str(axpos))[0],
                                                                                          eval('loc' + str(axpos))[1]],
                                                                        hbetalinefits["avg_g2_height"][eval('loc' + str(axpos))[0],
                                                                                          eval('loc' + str(axpos))[
                                                                                              1]] / norm2,
                                                                        hbetalinefits["avg_g2_center"][eval('loc' + str(axpos))[0],
                                                                                           eval('loc' + str(axpos))[1]],
                                                                        hbetalinefits["avg_g2_sigma"][eval('loc' + str(axpos))[0],
                                                                                          eval('loc' + str(axpos))[1]],
                                                                        hbetalinefits["avg_g3_height"][eval('loc' + str(axpos))[0],
                                                                                          eval('loc' + str(axpos))[
                                                                                              1]] / norm2,
                                                                        hbetalinefits["avg_g3_center"][eval('loc' + str(axpos))[0],
                                                                                           eval('loc' + str(axpos))[1]],
                                                                        hbetalinefits["avg_g2_sigma"][eval('loc' + str(axpos))[0],
                                                                                          eval('loc' + str(axpos))[1]],
                                                                        hbetalinefits["avg_c"][eval('loc' + str(axpos))[0],
                                                                                       eval('loc' + str(axpos))[
                                                                                           1]] / norm2), c='r')
    eval('ax' + str(axpos + 2)).plot(hbetaderedshiftbarywave[leftpix:rightpix],
                                     gauss(hbetaderedshiftbarywave[leftpix:rightpix], hbetalinefits["avg_g1_height"][
                                         eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2,
                                           hbetalinefits["avg_g1_center"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                           hbetalinefits["avg_g1_sigma"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                           hbetalinefits["avg_c"][
                                               eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2))
    if  hbetalinefits["choice"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] > 1:
        eval('ax' + str(axpos + 2)).plot(hbetaderedshiftbarywave[leftpix:rightpix],
                                         gauss(hbetaderedshiftbarywave[leftpix:rightpix], hbetalinefits["avg_g2_height"][
                                             eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2,
                                               hbetalinefits["avg_g2_center"][
                                                   eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                               hbetalinefits["avg_g2_sigma"][
                                                   eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                               hbetalinefits["avg_c"][
                                                   eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2))
        eval('ax' + str(axpos + 2)).plot(hbetaderedshiftbarywave[leftpix:rightpix], 
                                         twogauss(hbetaderedshiftbarywave[leftpix:rightpix],
                                             hbetalinefits["avg_g1_height"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2,
                                             hbetalinefits["avg_g1_center"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                             hbetalinefits["avg_g1_sigma"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                             hbetalinefits["avg_g2_height"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2,
                                             hbetalinefits["avg_g2_center"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                             hbetalinefits["avg_g2_sigma"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                             hbetalinefits["avg_c"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2))
    if hbetalinefits["choice"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] > 2:
        eval('ax' + str(axpos + 2)).plot(hbetaderedshiftbarywave[leftpix:rightpix],
                                         gauss(hbetaderedshiftbarywave[leftpix:rightpix], hbetalinefits["avg_g3_height"][
                                             eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2,
                                               hbetalinefits["avg_g3_center"][
                                                   eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                               hbetalinefits["avg_g3_sigma"][
                                                   eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                               hbetalinefits["avg_c"][
                                                   eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2))
        eval('ax' + str(axpos + 2)).plot(hbetaderedshiftbarywave[leftpix:rightpix], 
                                         threegauss(hbetaderedshiftbarywave[leftpix:rightpix],
                                             hbetalinefits["avg_g1_height"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2,
                                             hbetalinefits["avg_g1_center"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                             hbetalinefits["avg_g1_sigma"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                             hbetalinefits["avg_g2_height"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2,
                                             hbetalinefits["avg_g2_center"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                             hbetalinefits["avg_g2_sigma"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                             hbetalinefits["avg_g3_height"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2,
                                             hbetalinefits["avg_g3_center"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                             hbetalinefits["avg_g3_sigma"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]],
                                             hbetalinefits["avg_c"][eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[1]] / norm2))
    textstr = 'Position ' + posletters[axpos - 1] + ': ' + '\n' \
              + str(round(eval('loc' + str(axpos) + 'pos')[1], 1)) \
              + ', ' + str(round(eval('loc' + str(axpos) + 'pos')[0], 1)) + ' arcsec' + '\n' \
              + '$\sigma_{1G}:$' + str(int(hbetasinglesigma[eval('loc' + str(axpos))[0], eval('loc' + str(axpos))[
        1]])) + ' km s$^{-1}$'
    eval('ax' + str(axpos + 2)).text(0.02, 0.95, textstr, transform=eval('ax' + str(axpos + 2)).transAxes, fontsize=20,
                                     verticalalignment='top')
    eval('ax' + str(axpos + 2)).set_ylim([-0.15, 1.2])
    ax1.text(eval('loc' + str(axpos))[1], eval('loc' + str(axpos))[0], posletters[axpos - 1],
             fontsize=30, color='r', weight='bold')
    ax2.text(eval('loc' + str(axpos))[1], eval('loc' + str(axpos))[0], posletters[axpos - 1],
             fontsize=30, color='r', weight='bold')
ax3.tick_params(which='major', labeltop=False, labelbottom=True, labelleft=True, labelright=False, labelsize=20)
ax4.tick_params(which='major', labeltop=False, labelbottom=True, labelleft=False, labelright=False, labelsize=20)
ax5.tick_params(which='major', labeltop=False, labelbottom=True, labelleft=False, labelright=True, labelsize=20)
ax3.set_ylim(top=0.25, bottom=-0.05)
ax4.set_ylim(top=0.25, bottom=-0.05)
ax5.set_ylim(top=0.25, bottom=-0.05)
ax3.set_ylabel('Flux (Norm.)', fontsize = 25)
ax4.set_xlabel('Velocity (km s$^{-1}$)', fontsize = 25)
plt.savefig(gal + '/' + filename_preamble + '_Disk Outflow Map.pdf', bbox_inches='tight')
# plt.show()
plt.close()
