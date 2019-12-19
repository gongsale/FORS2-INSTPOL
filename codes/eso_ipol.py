## THIS FILE CONTAINS MULTIPLE ROUTINES USED FOR REDUCTION AND ANALYSIS OF ESO/FORS2/IPOL DATA
## Following is a list of the routines, a more general description can be found at the beginning
##  of each routine

## --------------------------------------------------------------------
## --- GENERAL ROUTINES (FOR BOTH, POINT SOURCE AND EXTENDED POLARIMETRY)
## --------------------------------------------------------------------

## COMBINE_IMAGES: Combine several images into one
## PREPARE_FOLDER: Routine that organizes raw data (science,bias,flat) into folders and
##                  creates map files
## GET_OFFSETS: If there are multiple offsets or iterations, return offset info based
## READ_FITS: Read fits file
## MASTER_BIAS: Function to stack biases and obtain a master bias
## COSMIC_RAYS: Correct image for cosmic rays
## DATA_FLAT: Function to obtain flat based on data by summing all angles
## NOISE: Function to get a noise map from a given signal map
## SEPARATE_BEAMS: separate ordinary and extraordinary beams from image
## STICK_CHIPS: Combine chip1/2 images into single
## CHROMATISM_CORRECTION: Correct angle of the half wave plates according to filter
## POLARIZATION: Function to get normalized flux differences, Q, U and polarization/angle
## QUPOLARIZATION: Calculate polarization/angle if Q and U known

## --------------------------------------------------------------------
## --- OTHER SECONDARY ROUTINES --------------------------------------
## --------------------------------------------------------------------
## INICENTER: Find estimate of center interactively
## CENTROID: Find a centroid of image
## SHIFTIM: Shift image by scalar offset
## ALIGN: Align two images interactively
## FIND_GAIN: find gain of an image (NOT WORKING)
## ASTROMETRY: Find astrometry of an image
## PLOT_FLAT: Plot data flats and stats

## --------------------------------------------------------------------
## ---- ROUTINES FOR POINT SOURCE POLARIMETRY ------------------------
## --------------------------------------------------------------------
## PSF_PHOT: Get PSF photometry 
## APERTURE_PHOT: Get aperture photometry
## WRITE_PHOTFILE: Function to write in dat files
##                  the final point source polarization variables
## AVERAGE_ITERATIONS: Average photometry and pol,angle of a given source
##                     that had multiple images (same filter,HWP-angle)
## ANALYSE_FITLPHPOL: Analyse polarization/angle vs filter (and fit Serkowski)

## --------------------------------------------------------------------
## ---- ROUTINES FOR EXTENDED POLARIMETRY ----------------------------
## --------------------------------------------------------------------
## GALISOPHOT: Function to do ellitpical isophot fits to galaxy image and obtain a mask to fit background
## GET_STRIPS: Function to obtain optimal strip offsets that divide ordinary/extraordin
## FIND_SHIFT: Function to obtain offsets between ordinary/extraordinary images
## FIND_STARS: Function to find stars in an image (DAOFIND)
## EBEAM_SHIFT: Shift extraordinary beam according to offsets from 'find_stars'
## BIN_IMAGE: Bins the image into smaller boxes
## PLOTPOL: Perform final polarization map plots
## RADIUS_DEPENDENCE: Plot pol/angle vs pixel radius
## ANALYSE_ANGSTRIPS: Analyse the strip position vs angle
## ANALYSE_FILTSTRIPS: Analyse strip positions vs filter
## ANALYSE_ANGQUADPARS: Analyse quadratic parameters of find_stars vs angle
## ANALYSE_FILTQUADPARS: Analyse quadratic parameters of find_stars vs filter


################################################################################
################################################################################
################################################################################

## --------------------------------------------------------------------
## ------------------------ GENERAL ---------------------------------
## --------------------------------------------------------------------
import os,sys
import string
import numpy as np
from astropy.io import fits,ascii
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from skimage.morphology import disk,square
from skimage.filters import rank
import pdb ## STOP; pdb.set_trace()
import pickle

home=os.path.expanduser('~')

## --------------------------------------------------------------------
## ------------- FUNCTION TO WRITE PHOT PHOL ------------------------
## --------------------------------------------------------------------
## PURPOSE: Write in ascii format the final point source polarization
##          obtained from routine 'polarization'
## INPUT:   1. savefile: path+basic_file name where to output quantities
##          2. fluxes of ordinary beam (list in different angles)
##          3. flux errors of ordinary beam (list in different angles)
##          4. fluxes of extraordinary beam (list in different angles)
##          5. flux errors of extraordinary beam (list in different angles)
##          6. polarization output tuple of 'polarization' routine
##          7. error in polarization output tuple of 'polarization' routine
## OUTPUT: This function does not return anything but writes output into a file
##         with savefile name ('...phot.dat') and creates a plot ('..phot.png')

def write_photfile(savefile,ophot,ephot,erophot,erephot,phinfo):

    pol,angle,Q,U,F = phinfo['pol'],phinfo['angle'],phinfo['Q'],phinfo['U'],phinfo['fdiff']
    erpol,erangle,erQ,erU,erF = phinfo['erpol'],phinfo['erangle'],phinfo['erQ'],phinfo['erU'],phinfo['erfdiff']
    erpol,erangle,erQ,erU,erF = erphpol
    photfile = open(savefile+'phot.dat','w')
    photfile.write("APERTURE PHOTOMETRY SUMMARY\n")
    photfile.write("HWP-angle  ORD-phot  ORD-errphot  EXT-phot  EXT-errphot  Fourier  Fourier-err: \n")
    for index in range(len(F)):
        photfile.write("%i        %f %f %f %f %f %f\n" %(index*22.5,ophot[index],erophot[index],
                                                         ephot[index],erephot[index],
                                                         F[index],erF[index]))
    photfile.write("U-Stokes: %f \n" %U)
    photfile.write("U-Stokes-error: %f \n" %erU)
    photfile.write("Q-Stokes: %f \n" %Q)
    photfile.write("Q-Stokes-error: %f \n" %erQ)
    photfile.write("Angle: %f \n" %angle)
    photfile.write("Angle error: %f \n" %erangle)
    photfile.write("Polarization: %f \n" %pol)
    photfile.write("Polarization error: %f \n" %erpol)
    photfile.close()

    ##save also python file
    np.save(savefile+'phot.npy',(pol,angle,Q,U,F,erpol,erangle,erQ,erU,erF))
    
    ##plot
    nang = len(F)
    ang = 22.5
    kangle = np.arange(nang)*ang
    fig,ax = plt.subplots(3,sharex=True)
    fig.subplots_adjust(hspace=0)
    ax[0].errorbar(kangle,ophot.reshape(-1),yerr=erophot.reshape(-1),fmt='o')
    ax[0].set_ylabel('Ordinary Beam Flux')
    ax[0].ticklabel_format(stype='sci',axis='y',scilimits=(0,0))
    ax[1].errorbar(kangle,ephot.reshape(-1),yerr=erephot.reshape(-1),fmt='o')
    ax[1].set_ylabel('Extraordinary Beam Flux')
    ax[1].ticklabel_format(stype='sci',scilimits=(0,0))
    ax[2].errorbar(kangle,F.reshape(-1),yerr=erF.reshape(-1),fmt='o')
    pltangle = np.arange(nang*20)*ang/20
    ax[2].plot(pltangle,pol[0]*np.cos(4*pltangle/180*np.pi-2*angle[0]/180*np.pi),'-')
    ax[2].set_ylabel('Fourier coefficient')
    ax[2].set_xlabel('Angle HWP')
    ax[2].set_xticks(kangle)
    plt.savefig(savefile+'phot.png')
    plt.close("all")
    
## --------------------------------------------------------------------
## ------------- FUNCTION TO CORRECT Q,U FOR CHROMATISM -------------
## PURPOSE: This does the correction of angle of the half wave plate
##          (see ESO manual section 4.6.2)
## INPUT: 1. angle (scalar) of the HWP
##        2. filter at which to do correction (any of 'U','b_HIGH','v_HIGH','R_SPECIAL','I_BESS')
## OUTPUT: newangle corrected

def chromatism_correction(angle,filt):#(q,u,filt):

    # -- from ESO user manual:
    filters = np.array(['u_HIGH','b_HIGH','v_HIGH','R_SPECIAL','I_BESS','H_Alpha','OII_8000'])
    waves = np.array([361,440,557,655,768,656.3,381.4])#nm
    chrom = np.array([-2.07,1.54,1.80,-1.19,-2.89,-1.21,-1.14])#/180.0*np.pi#deg to rad, Halpha,OIII:linear interp
    
    # -- this filt
    chcorr = chrom[np.asarray(filters == filt)][0]
    
    # -- equations (according to Cikota17):
    #qcorr = q*np.cos(2*chcorr)-u*np.sin(2*chcorr)
    #ucorr = q*np.sin(2*chcorr)+u*np.cos(2*chcorr)

    # -- equation (according to ESO manual):
    newangle = angle-chcorr
    
    return newangle#qcorr,ucorr

## ---------------------------------------------------------------------
## ---------- FUNCTION TO GET S/N -------------- ----------------------
## ---------------------------------------------------------------------
## PURPOSE: This function gets a noise map using photutils (sigma-clipping in box)
## INPUT: 1. Image to get noise from
##        2. Mask map (with good pts to use)
##        3. Box size (in pixels) where to do sigma-clipping
## OPTIONAL INPUT:
##        sigmaclip    sigma cut to do clipping when calculating background (def:2.0)
## OUTPUT: noise image
## DEPENDENCIES: photutils (python)

def noise(image,mask,radpix,sigmaclip=2.0,savefile=None):

    if (savefile is not None):
        if (os.path.isfile(savefile+'-noise.fits')):
            print("   Found existing noise file: %s" %(savefile+'-noise.fits'))
            err = fits.open(savefile+'-noise.fits')
            return err[0].data
    
    #from astropy.stats import SigmaClip
    from photutils.background import MMMBackground
    from photutils import SigmaClip,Background2D
    print("   Finding noise map")

    ## Background noise [http://photutils.readthedocs.io/en/stable/background.html]
    sigma_clip = SigmaClip(sigma=sigmaclip, iters=10) 
    bkg_estimator = MMMBackground(sigma_clip=sigma_clip)
    bkg = Background2D(image,(radpix,radpix),mask=~mask,filter_size=(3,3),
                       bkg_estimator=bkg_estimator,exclude_mesh_percentile=60)#sigma_clip=sigma_clip, 
    print("      Median background found: %f, median bkg noise: %f"
          %(bkg.background_median,bkg.background_rms_median)  )

    ##Include Poisson noise from sources [http://photutils.readthedocs.io/en/stable/aperture.html]
    # from photutils.utils import calc_total_error
    # effective_gain = 500   # seconds (can be scalar or 2d arr)
    # error = calc_total_error(data, bkg_error, effective_gain)     

    if savefile is not None:
        fits.writeto(savefile+'-noise.fits',bkg.background_rms,clobber=True)
    
    #return bkg.background,bkg.background_rms# * mask
    return bkg.background_rms# * mask
    
## ---------------------------------------------------------------------
## ---------- FUNCTION TO CALCULATE ERROR IN POLARIZATION --------------
## ---------------------------------------------------------------------
## PURPOSE: This function gets error of polarization based on S/N (eq. 10,11 Patat06)
## INPUT: 1. Ordinary beam (can be a matrix)
##        2. Extraordinary beam (same size as ordinary beam)
##        3. Polarization map from polarization fct
##        4. Angle map from polarization fct
## OPTIONAL INPUT:
##        - mask/emask:  Masks of beam/ebeam
##        - errbeam/errebeam: Errors in Ordinary/Extraordinary beams
##        - savefile: where to save/load results
##        - radpix:  When calculating noise (def: 40)
##        - bias: Boolean to correct for polarization bias
##        - method: Method to correct for polarization bias (def: 'WK74')
## OUTPUT: erpol,erangle

def erpolarization(beam,ebeam,poldeg,angle,radpix=40,savefile=None,method='WK74',
                   mask=None,emask=None,bias=True,errbeam=None,errebeam=None):

    ##-- load if file exists
    if savefile is not None:
        if (os.path.isfile(savefile+'-erpol.fits')):
            erpol = fits.open(savefile+'-erpol.fits')
            erangle = fits.open(savefile+'-erangle.fits')
            if bias: poldeg = polbias(poldeg,erpol[0].data,method=method)
            return poldeg,erpol[0].data,erangle[0].data
    
    ##-- nr of angles
    N = len(beam)

    ##--Mask
    if mask is None:
        tot,etot = np.sum(beam,0),np.sum(ebeam,0)
        tmask = ((tot > 0) & (etot > 0))
    else:
        tmask = (mask & emask)
        if np.shape(tmask) == np.shape(beam):
            tmask = (np.sum(tmask,axis=0) == N)
        #tmask = smask[0]
        #for i in np.arange(1,N): tmask*=smask[i]
    
    ##-- Error in pol,angle: use eq.10,11 of Pata06 with sum of beam+ebeam for noise
    #- sum (beam+ebeam) and get median of all N (could also be the sum)
    intensity = np.median(beam+ebeam,axis=0) #np.sum(beam+ebeam,axis=0)
    if errbeam is None:
        ## Background and noise
        bkg_rms = noise(intensity,tmask,radpix)
    else:
        ## Sum noises
        err = np.sqrt(errbeam**2+errebeam**2)
        bkg_rms = np.median(err,axis=0)
    
    #- S/N
    snr = np.zeros(np.shape(intensity),dtype=float)
    snr[tmask] = intensity[tmask]/bkg_rms[tmask]
    medsnr = np.nanmedian(snr[tmask])
    stdsnr = np.nanmedian(np.abs(snr[tmask]-medsnr))
    print("      Median/MAD SNR: %.4e,%.4e" %(medsnr,stdsnr))
    erpoldeg = np.zeros(np.shape(intensity),dtype=float)
    erangle = np.zeros(np.shape(intensity),dtype=float)
    erpoldeg[tmask] = 1.0/(np.sqrt(N/2)*snr[tmask])
    erangle[tmask] = erpoldeg[tmask]/(2*poldeg[tmask])/np.pi*180.0

    if savefile is not None:
        fits.writeto(savefile+'-erpol.fits',erpoldeg,clobber=True)
        fits.writeto(savefile+'-erangle.fits',erangle,clobber=True)

    if bias: poldeg = polbias(poldeg,erpoldeg,mask=tmask,method=method)
    return poldeg,erpoldeg,erangle

## ---------------------------------------------------------------------
## ---------- FUNCTION TO CORRECT POLARIZATION BIAS --------------
## ---------------------------------------------------------------------
## PURPOSE: This function correct the polarization bias
## INPUT: 1. Pol degree
##        2. Error on pol degree
## OPTIONAL INPUT:
##        - mask 
##        - method: 'WR74' = Wardle & Kronberg 1974
##                  'P14'  = Plaszczynski et al. 2014
##
## OUTPUT: erpol,erangle

def polbias(poldeg,polerr,method='WK74',mask=None):

    if method == '': method = 'WK74'
    newpol = np.zeros(poldeg.shape,dtype=float)
    newpol.fill(np.nan)
    if mask is None: mask = (poldeg > 0)
    
    if method == 'WK74':
        newpol[mask] = np.sqrt(poldeg[mask]**2.0 - polerr[mask]**2.0)
    elif method == 'P14':
        newpol[mask] = poldeg[mask] - polerr[mask]**2.0 *\
                       (1.0-np.exp(-poldeg[mask]**2.0/polerr[mask]**2.0))/(2*poldeg[mask])
    else:
        print("   ERROR in polbias: unknown method: "+method)

    newpol[((mask) & ~np.isfinite(newpol))] = 0.0
        
    return newpol

## ---------------------------------------------------------------------
## ---------- FUNCTION TO CALCULATE STOKES PARAMETERS ----------------------
## ---------------------------------------------------------------------
## PURPOSE: Calculates normalized flux differences, Q and U Stokes parameters
##          (see e.g. eq. 4.4-4-6 of ESO manual)
## INPUT:
##        1. Ordinary beam (can be a matrix)
##        2. Extraordinary beam (same size as ordinary beam)
## OPTIONAL INPUT:
##        - mask/emask: Mask map (values that are ok) of ordinary/extraordinary beam
##        - errbeam/errebeam: Error map of ordinary/extraordinary beam
##        - savefile: name of path+basic_file name where to save results
## OUTPUT: Returns (q,u),(er_q,er_u)

def stokes(beam,ebeam,savefile=None,mask=None,emask=None,errbeam=None,errebeam=None):
    
    ##Check if already done
    if (savefile is not None):
        if (os.path.isfile(savefile+'-QStokes.fits')):
            print("   Found existing QU files: %s" %(savefile+'-QStokes.fits'))
            qarr,uarr = fits.open(savefile+'-QStokes.fits'),fits.open(savefile+'-UStokes.fits')
            intarr = fits.open(savefile+'-intensity.fits')
            fd,erfd = fits.open(savefile+'-Fdiff.fits'),fits.open(savefile+'-erFdiff.fits')         
            af,eraf = fits.open(savefile+'-aFourier.fits'),fits.open(savefile+'-eraFourier.fits')
            bf,erbf = fits.open(savefile+'-bFourier.fits'),fits.open(savefile+'-erbFourier.fits')
           
            #dictionary
            stokes = {}
            stokes['Q'],stokes['erQ'] = qarr[0].data
            stokes['U'],stokes['erU'] = uarr[0].data
            stokes['intensity'] = intarr[0].data 
            stokes['fdiff'],stokes['erfdiff'] = fd[0].data,erfd[0].data
            stokes['afourier'],stokes['erafourier'] = af[0].data,eraf[0].data
            stokes['bfourier'],stokes['erbfourier'] = bf[0].data,erbf[0].data
            return stokes

    ndim = np.ndim(beam[0])
    if ndim > 1: (ypix,xpix) = np.shape(beam[0])
    else: ypix,xpix = np.size(beam[0]),1
    
    ##number of angles
    N = len(beam)

    #masks
    if mask is None:
        tot,etot = np.sum(beam,0),np.sum(ebeam,0)
        tmask = ((tot > 0) & (etot > 0))
    else:
        tmask = (mask & emask)
        if np.shape(tmask) == np.shape(beam):
            tmask = (np.sum(tmask,axis=0) == N)

    if ndim == 1:
        tmask = tmask.reshape(np.size(beam[0]),1)
        
    ##-- get flux-diff & stokes parameters (Patat eqs 5,7,8)
    intensity = np.median(beam+ebeam,axis=0) #np.sum(beam+ebeam,axis=0)
    fdiff,er_fdiff = np.zeros((N,ypix,xpix),dtype=float),np.zeros((N,ypix,xpix),dtype=float)
    q_stokes,er_q_stokes = np.zeros((ypix,xpix),dtype=float),np.zeros((ypix,xpix),dtype=float)
    u_stokes,er_u_stokes = np.zeros((ypix,xpix),dtype=float),np.zeros((ypix,xpix),dtype=float)    
    for a in range(0,N):
        fdiff[a,tmask] = (beam[a][tmask]-ebeam[a][tmask])/(beam[a][tmask]+ebeam[a][tmask])
        q_stokes[tmask] += 2.0/N*np.cos(np.pi/2.0*a)*fdiff[a,tmask]
        u_stokes[tmask] += 2.0/N*np.sin(np.pi/2.0*a)*fdiff[a,tmask]
          
        ## --error propagation
        if errbeam is not None:
            er_fdiff[a,tmask] = np.sqrt((2*ebeam[a][tmask]*errbeam[a][tmask])**2+
                                        (2*beam[a][tmask]*errebeam[a][tmask])**2)/((beam[a][tmask]+ebeam[a][tmask])**2)
            er_q_stokes[tmask] += (2.0/N*np.cos(np.pi/2.0*a)*er_fdiff[a,tmask])**2
            er_u_stokes[tmask] += (2.0/N*np.sin(np.pi/2.0*a)*er_fdiff[a,tmask])**2
            
    ##--normalize to intensity -> already done in previous equations!!
    #q_stokes[tmask] = q_stokes[tmask]/intensity[tmask]
    #u_stokes[tmask] = u_stokes[tmask]/intensity[tmask]
    #if errbeam is not None:
    #    er_q_stokes[tmask] = er_u_stokes[tmask]/intensity[tmask]
    #    er_u_stokes = er_q_stokes[tmask]/intensity[tmask]

    ##-- get Fourier coeffs - eq 9 for nk = N/2 (why not?)
    ##   errors from erFdiff - but see Patat06 for S/N way
    Nf = np.float(N)
    afourier,bfourier = np.zeros((N/2,ypix,xpix),dtype=float),np.zeros((N/2,ypix,xpix),dtype=float)
    erafourier,erbfourier = np.zeros((N/2,ypix,xpix),dtype=float),np.zeros((N/2,ypix,xpix),dtype=float)
    afourier[0,:,:] = 1.0/Nf*np.sum(fdiff,axis=0)
    for k in range(1,N/2):
        Fcosmult = [fdiff[i,tmask]*np.cos(k*2.0*np.pi*i/Nf) for i in np.arange(0,N,1)]
        afourier[k,tmask] = 2.0/Nf*np.sum(Fcosmult,axis=0)
        Fsinmult = [fdiff[i,tmask]*np.sin(k*2.0*np.pi*i/Nf) for i in np.arange(0,N,1)]
        bfourier[k,tmask] = 2.0/Nf*np.sum(Fsinmult,axis=0)
        Fcos2mult = [(er_fdiff[i,tmask]*np.cos(k*2.0*np.pi*i/Nf))**2.0 for i in np.arange(0,N,1)]
        erafourier[k,tmask] = 2.0/Nf*np.sum(Fcos2mult,axis=0)
        Fsin2mult = [(er_fdiff[i,tmask]*np.sin(k*2.0*np.pi*i/Nf))**2.0 for i in np.arange(0,N,1)]
        erbfourier[k,tmask] = 2.0/Nf*np.sum(Fsin2mult,axis=0)

    ##-- save results
    if savefile is not None:
        fits.writeto(savefile+'-QStokes.fits',np.asarray((q_stokes,er_q_stokes),dtype=float),clobber=True)
        fits.writeto(savefile+'-UStokes.fits',np.asarray((u_stokes,er_u_stokes),dtype=float),clobber=True)
       
        fits.writeto(savefile+'-intensity.fits',intensity,clobber=True)
        fits.writeto(savefile+'-aFourier.fits',afourier,clobber=True)
        fits.writeto(savefile+'-eraFourier.fits',erafourier,clobber=True)
        fits.writeto(savefile+'-bFourier.fits',bfourier,clobber=True)
        fits.writeto(savefile+'-erbFourier.fits',erbfourier,clobber=True)
        fits.writeto(savefile+'-Fdiff.fits',fdiff,clobber=True)
        fits.writeto(savefile+'-erFdiff.fits',er_fdiff,clobber=True)

    #dictionary
    stokes = {}
    stokes['Q'],stokes['U'] = q_stokes,u_stokes
    stokes['erQ'],stokes['erU'] = er_q_stokes,er_u_stokes
    stokes['intensity'] = intensity
    stokes['fdiff'],stokes['erfdiff'] = fdiff,er_fdiff
    stokes['afourier'],stokes['erafourier'] = afourier,erafourier
    stokes['bfourier'],stokes['erbfourier'] = afourier,erbfourier      
    return stokes


## -------------------------------------------------------------------------------
## -------- QUcorrect: FUNCTION TO CORRECT Q/U PARAMETERS FOR Q0/U0 AND PLOT -----
## ------------------------------------------------------------------------------
## PURPOSE: Correct Q and U values for some Q0/U0 value either provided by user
##          and/or calculated at center or binned center position
## INPUT: 1. Q Stokes image
##        2. U Stokes image
##        3. savefile to plot QU plane
## OPTIONAL INPUT:
##        - mask of Q/U
##        - errQ/errU: error in Q/U useful for errors in center values
##        - Q0/U0 values at which to correct all Q/U if corr ='given'
##            (otherwise calculated accroding to corr)
##        - corr: String to know which center Q0/U0 correction to make (def: 'None')
##           e.g. when doing moon instrumental pol 
##              'None': no correction (default)
##              'cen' : correct for value of center pixel value of 'center' position 
##              'bincen': correct for value of center bin (15pix radius) around 'center' position
##              'peakhist': correct for value of peak in 2d distribution
##              'med': median value of all Q/U
##              'given': provided in Q0/U0
##        - fcorr: correct for field polarization
##              'None': no correction (default)
##              'hyppar': correct for rotated hyperboic paraboloid fit
##                             (loaded from Information/hypparab_instQ/U.dat)
##              'loadmap': loads map from Information/FILTER_instQ/U.fits
##              'inla': loads INLA map from Information/FILTER_inlaQ/U.fits
##        - scatter: When plotting do scatter plot or 2d hist (def: False)
##        - parfit: see plotstokes
##        - fitmask: mask to fit fct (see plotstokes)
##        - inla: see plotstokes
## OUTPUT:
##       Corrected Q/U
## SIDE EFFECTS: calls plotstokes to do QU plane plot
## COMMENT: include error propagation from Q0/U0????

def polfct(Q,U):
    pol = np.sqrt(U**2+Q**2)
    ang = 0.5*np.sign(U)*np.arccos(Q/pol)/np.pi*180
    if np.size(Q) == 1:
        if ang < 0: ang += 180
    else:
        ang[np.isfinite(ang) & (ang < 0)] += 180
    return pol,ang
#angle = 0.5*np.arctan2(u_stokes,q_stokes)/np.pi*180#
def erpolfct(Q,U,erQ,erU):
    erpol = np.sqrt(((Q*erQ)**2+(U*erU)**2)/(Q**2+U**2))
    erang = 0.5*np.sqrt((Q*erU)**2+(U*erQ)**2)/((1+(U/Q)**2)*Q**2)
    return erpol,erang
        
def QUcorrect(q_stokes,u_stokes,savefile=None,Q0=None,U0=None,filt=None,
              x=None,y=None,parfit=False,inla=False,nx=2049,ny=2064,fitmask=None,
              scatter=False,center=None,corr='None',fcorr='None',mask=None,errQ=None,errU=None):

    (ypix,xpix) = np.shape(q_stokes)
    if x is None: nx,ny = xpix,ypix
    Qcorr,Ucorr = 0,0
    er_q_stokes,er_u_stokes = errQ,errU
    qucorr = {}
    qucorr['Qraw'],qucorr['Uraw'] = q_stokes.copy(),u_stokes.copy()
    
    ##-- mask
    if mask is None:
        mask = np.isfinite(q_stokes) & np.isfinite(u_stokes) & (q_stokes != 0) & (u_stokes != 0)
        
    ##-- 1. Calculate center Q/U values 
    q0_stokes,u0_stokes = None,None
    er_q0_stokes,er_u0_stokes = None,None

    if (xpix > 1) & (ypix > 1): # only if full matrix
    
        if center is None:
            tcenter = [ypix/2,xpix/2]
        else: tcenter = np.asarray(center,dtype=int) 

        ##--Center
        if scatter:
            u0_stokes = u_stokes[tcenter[1],tcenter[0]]
            q0_stokes = q_stokes[tcenter[1],tcenter[0]]
            if errQ is not None:
                er_q0_stokes = errQ[tcenter[1],tcenter[0]]
                er_u0_stokes = errU[tcenter[1],tcenter[0]]
        ##--Bincenter
        else:
            from astropy.stats import sigma_clipped_stats
            sigmaclip,radpix = 2.0,15
            mask0 = mask[tcenter[1]-radpix:tcenter[1]+radpix,tcenter[0]-radpix:tcenter[0]+radpix]
            u0bin = u_stokes[tcenter[1]-radpix:tcenter[1]+radpix,tcenter[0]-radpix:tcenter[0]+radpix]
            q0bin = q_stokes[tcenter[1]-radpix:tcenter[1]+radpix,tcenter[0]-radpix:tcenter[0]+radpix]
            mean,q0_stokes,er_q0_stokes = sigma_clipped_stats(q0bin,mask=~mask0,sigma=sigmaclip,iters=10)
            mean,u0_stokes,er_u0_stokes = sigma_clipped_stats(u0bin,mask=~mask0,sigma=sigmaclip,iters=10)          
        #if 'cen' in corr: 
        #    Qcorr,Ucorr = q0_stokes,u0_stokes
                
        ##-- Calculate center pol/angle for info
        p0,ang0 = polfct(q0_stokes,u0_stokes)

        print("     Center Q and U: %f %f" %(q0_stokes,u0_stokes))
        print("     Center Pol and ang: %f %f" %(p0,ang0))
    
        if errQ is not None:
            erp0,erang0 = erpolfct(q0_stokes,u0_stokes,er_q0_stokes,er_u0_stokes)
            print("     Center erQ and erU: %f %f" %(er_q0_stokes,er_u0_stokes))
            print("     Center erPol and erang: %f %f" %(erp0,erang0))

    elif 'cen' in corr.lower():
        print("ERROR IN QUcorrect: Need matrix of Q/U to correct by center")
        sys.exit(-1)
        

    #intensity0 = intensity[tcenter[1],tcenter[0]]
    #angle = 0.5*np.arctan2(u_stokes,q_stokes)/np.pi*180
    #poldeg = np.sqrt(q_stokes**2+u_stokes**2)
    #poldeg1 = q_stokes/(intensity0*np.cos(2*angle/180*np.pi))
    #poldeg2 = u_stokes/(intensity0*np.sin(2*angle/180*np.pi))
    #poldeg = 0.5*(poldeg1+poldeg2)    

    ##-- Plot
    print("   Plotting QU ")
    Qpeak,Upeak,Qmed,Umed = plotstokes(q_stokes,u_stokes,savefile=savefile,mask=mask,
                                       scatter=scatter,Q0=q0_stokes,U0=u0_stokes,x=x,y=y,
                                       center=center,erQ0=er_q0_stokes,erU0=er_u0_stokes)

    ##--  pol and ang
    ppeak,angpeak = polfct(Qpeak,Upeak)
    print("     Most frequent Q and U: %f %f" %(Qpeak,Upeak))
    print("     Most frequent Pol and ang: %f %f" %(ppeak,angpeak))

    pmed,angmed = polfct(Qmed,Umed)
    print("     Median Q and U: %f %f" %(Qmed,Umed))
    print("     Median Pol and ang: %f %f" %(pmed,angmed))

    if corr.lower() == 'given':
        if Q0 is None:
            print("Error in QUcorrect: if you input Q0/U0 then corr should be 'given'")
            sys.exit(-1)
        #if len(Q0) == 1:
        Qcorr,Ucorr = Q0,U0
        pcorr,angcorr = polfct(Qcorr,Ucorr)
        print("     Given Q and U: %f %f" %(Qcorr,Ucorr))
        print("     Given Pol and ang: %f %f" %(pcorr,angcorr))
        #elif Q0.shape != q_stokes.shape:
        #    print("Error in QUcorrect: Q0/U0 not right shape!")
        #    sys.exit(-1)
        #else:
        #    Qcorr,Ucorr = Q0[mask],U0[mask]
            
    elif corr.lower() == 'peakhist':
        Qcorr,Ucorr = Qpeak,Upeak
    elif corr.lower() == 'med':
        Qcorr,Ucorr = Qmed,Umed
    elif 'cen' in corr.lower():
        Qcorr,Ucorr = q0_stokes,u0_stokes

    ##-- Correct central instrumental polarization (Patat06 eq 18, where Qcen=p*I_0*cos(2*phi))
    q_stokes[mask] = q_stokes[mask]-Qcorr
    u_stokes[mask] = u_stokes[mask]-Ucorr
    
    ##-- Plot after central correction
    cencorr=''
    if (corr.lower() != 'none' or parfit) and savefile is not None:
        print("   Plotting QU center-corrected ")
        if corr.lower() != 'none': cencorr="-corr"+corr
        Qpeak,Upeak,Qmed,Umed = plotstokes(q_stokes,u_stokes,savefile=savefile+cencorr,x=x,y=y,filt=filt,
                                           mask=mask,scatter=scatter,Q0=q0_stokes,U0=u0_stokes,
                                           fitmask=fitmask,parfit=parfit,
                                           center=center,erQ0=er_q0_stokes,erU0=er_u0_stokes,inla=inla)
      
        fits.writeto(savefile+cencorr+'-QStokes.fits',
                     np.asarray((q_stokes,er_q_stokes),dtype=float),clobber=True)
        fits.writeto(savefile+cencorr+'-UStokes.fits',
                     np.asarray((u_stokes,er_u_stokes),dtype=float),clobber=True)

    ##-- Write corr
    qucorr['Q0'],qucorr['U0'] = Qcorr,Ucorr
    qucorr['Qcencorr'],qucorr['Ucencorr'] = q_stokes.copy(),u_stokes.copy()
    qucorr['corrname'] = cencorr
        
    ## 2. Field instrumental correction    
    fieldcorr=''
    qucorr['Qinstpol'],qucorr['Uinstpol'] = None,None
    qucorr['Qfieldcorr'],qucorr['Ufieldcorr'] = None,None
    if fcorr.lower() != 'none':
     
        if filt is None:
             print("Error in QUcorrect: if you input fcorr, you need a filter")
             sys.exit(-1)

        if fcorr.lower() == 'loadmap':
            HQc,Qarr = read_fits(home+"/crisp/FORS2-POL/Information/"+filt+"_instQ.fits")
            HUc,Uarr = read_fits(home+"/crisp/FORS2-POL/Information/"+filt+"_instU.fits") 
            Qfcorr,erQfcorr = Qarr; Ufcorr,erUfcorr = Uarr

        elif fcorr.lower() == 'inla':
            HQc,Qfcorr = read_fits(home+"/crisp/FORS2-POL/Information/"+filt+"_inlaQ.fits")
            HUc,Ufcorr = read_fits(home+"/crisp/FORS2-POL/Information/"+filt+"_inlaU.fits")

                       
        elif 'hyppar' in fcorr.lower():
            ccenter=[ny/2,nx/2] if center is None else center
            xm,ym = np.linspace(0,nx-1,nx), np.linspace(0,ny-1,ny)
            xx,yy = np.meshgrid(xm-ccenter[1],ym-ccenter[0])
            ax,ay = xx.reshape(-1),yy.reshape(-1)
            rothypparaboloid = lambda xy,a,b,theta,x0,y0:\
                               ((xy[0]-x0)*np.sin(theta)+(xy[1]-y0)*np.cos(theta))**2.0/b**2.0 - \
                               ((xy[0]-x0)*np.cos(theta)-(xy[1]-y0)*np.sin(theta))**2.0/a**2.0
            crothypparaboloid = lambda xy,a,b,theta,x0,y0,cst:\
                               ((xy[0]-x0)*np.sin(theta)+(xy[1]-y0)*np.cos(theta))**2.0/b**2.0 - \
                               ((xy[0]-x0)*np.cos(theta)-(xy[1]-y0)*np.sin(theta))**2.0/a**2.0 + cst
            if fcorr.lower() == 'hyppar':
                tfile = home+"/crisp/FORS2-POL/Information/hypparab_instQU.dat"
                usefilts = np.loadtxt(tfile,usecols=0,dtype=object)
                allpars = np.loadtxt(tfile,usecols=(1,2,3,4,5,6,7,8,9,10))
                i = np.argwhere(np.array(usefilts) == filt).reshape(-1)[0]
                Qpars,Upars = allpars[i,0:5],allpars[i,5:10]
            else:
                Qpars,erQpars = np.loadtxt(savefile+cencorr+'-Qmodel.dat')
                Upars,erUpars = np.loadtxt(savefile+cencorr+'-Umodel.dat')
                    
            Qfcorr = crothypparaboloid((ax,ay),*Qpars)
            Ufcorr = crothypparaboloid((ax,ay),*Upars)
            Qfcorr,Ufcorr = Qfcorr.reshape((ny,nx)),Ufcorr.reshape((ny,nx)) 

        
        ##-- Correct instrumental spatial polarization (Patat06 eq 18, where Qcen=p*I_0*cos(2*phi))
        if x is None:
            q_stokes[mask] = q_stokes[mask]-Qfcorr[mask]
            u_stokes[mask] = u_stokes[mask]-Ufcorr[mask]
        else:
            yi,xi = np.asarray(np.floor(y.reshape(-1)),dtype=int),np.asarray(np.floor(x.reshape(-1)),dtype=int)
            q_stokes[:,0] = q_stokes[:,0]-Qfcorr[yi,xi]
            u_stokes[:,0] = u_stokes[:,0]-Ufcorr[yi,xi]

        ##-- Residual median/MAD of corrected Q/U
        medQres = np.median(q_stokes[mask])
        stdQres = np.median(np.abs(q_stokes[mask]-medQres))
        print("      Residual Qcorr median/MAD: %.4e,%.4e" %(medQres,stdQres)) 
        medUres = np.median(u_stokes[mask])
        stdUres = np.median(np.abs(u_stokes[mask]-medUres))
        print("      Residual Ucorr median/MAD: %.4e,%.4e" %(medUres,stdUres))     
            
        ##-- Write
        fieldcorr='-fcorr'+fcorr
        qucorr['Qinstpol'],qucorr['Uinstpol'] = Qfcorr,Ufcorr
        qucorr['Qfieldcorr'],qucorr['Ufieldcorr'] = q_stokes.copy(),u_stokes.copy()
        qucorr['corrname'] = cencorr+'-fcorr'+fcorr
            
        ##-- Plot after field correction
        if savefile is not None:
            print("   Plotting QU field-corrected")
            Qpeak,Upeak,Qmed,Umed = plotstokes(q_stokes,u_stokes,savefile=savefile+cencorr+fieldcorr,center=center,
                                               erQ=er_q_stokes,erU=er_u_stokes,scatter=scatter,x=x,y=y)#mask=mask,

            fits.writeto(savefile+cencorr+fieldcorr+'-QStokes.fits',
                         np.asarray((q_stokes,er_q_stokes),dtype=float),clobber=True)
            fits.writeto(savefile+cencorr+fieldcorr+'-UStokes.fits',
                         np.asarray((u_stokes,er_u_stokes),dtype=float),clobber=True)
       
        
    return q_stokes,u_stokes,qucorr

## ---------------------------------------------------------------------
## ---------- FUNCTION TO CALCULATE POLARIZATION ----------------------
## ---------------------------------------------------------------------
## PURPOSE: Calculates linear polarization and angle from Q/U Stokes calculated with 'stokes' fct
##           (see e.g. eq. 4.4-4-6 of ESO manual)
## INPUT:
##        1. Q Stokes image
##        2. U Stokes image
##        3. Filter at which to do this (any of 'U','b_HIGH','v_HIGH','R_SPECIAL','I_BESS')
## OPTIONAL INPUT:
##        - mask/emask: Mask map (values that are ok) of ordinary/extraordinary beam
##        - errQ/errU: Error map of Q/U
##        - savefile: name of path+basic_file name where to save results
##        - chrom: Boolean to do chromatic correction (default: True).
##                 See routine 'chromatism_correction'
## OUTPUT: Returns poldeg,angle

def QUpolarization(q_stokes,u_stokes,filt,savefile=None,chrom=True,mask=None,emask=None,
                   errQ=None,errU=None):

    ##Check if already done
    if (savefile is not None):
        if (os.path.isfile(savefile+'-pol.fits')):
            print("   Found existing pol file: %s" %(savefile+'-pol.fits'))
            pol = fits.open(savefile+'-pol.fits')
            angle = fits.open(savefile+'-angle.fits')
            return pol[0].data,angle[0].data
        #if (os.path.isfile(savefile+'-pol.pkl')):
        #    print("   Found existing pol file: %s" %(savefile+'-pol.pkl'))
        #    with open(savefile+'-pol.pkl') as f:  # Python 3: open(..., 'rb')
        #        pol = pickle.load(f)
        #    return pol

    print("   Calculating Q/U polarization and angle")
    (ypix,xpix) = np.shape(q_stokes)
    
    #masks
    if mask is None:
        mask = (q_stokes != 0) & (u_stokes != 0)
      
    ##-- Pol and Angle (from Landi Degl'Innocenti07)|
    poldeg,angle = np.zeros(np.shape(q_stokes),dtype=float),np.zeros(np.shape(q_stokes),dtype=float)
    poldeg[mask] = np.sqrt(q_stokes[mask]**2+u_stokes[mask]**2)
    angle[mask] = 0.5*np.sign(u_stokes[mask])*np.arccos(q_stokes[mask]/poldeg[mask])/np.pi*180
    ##before:
    #angle = 0.5*np.arctan2(u_stokes,q_stokes)/np.pi*180#arctan2
    
    ##-- angle fix
    angle[np.isfinite(angle) & (angle < 0)] += 180
    
    ##-- chromatism correction!
    if chrom:
        angle = chromatism_correction(angle,filt)
        
    ##-- Error in pol, angle: propagation: USING NOW SNR IN ERPOLARIZATION 
    if errQ is not None:
        errQ,errU = np.sqrt(errQ),np.sqrt(errU)
        erpoldeg = np.sqrt((q_stokes**2*errQ**2+u_stokes**2*errU**2)/
                           (q_stokes**2+u_stokes**2))
        erangle = 0.5*np.sqrt((q_stokes*errU)**2+(u_stokes*errQ)**2)/((1+(u_stokes/q_stokes)**2)*q_stokes**2)


    ##-- Before pol dictionary
    #pol = {}
    #pol['pol'],pol['erpol'] = poldeg,erpoldeg
    #pol['angle'],pol['erangle'] = poldeg,erpoldeg
    #pol['Q'],pol['erQ'] = q_stokes,er_q_stokes
    #pol['U'],pol['erU'] = u_stokes,er_u_stokes
    #pol['fdiff'],pol['erfdiff'] = fdiff,er_fdiff
    #if corrcen != 'None':
    #    pol['Q0'],pol['erQ0'] = q0_stokes,er_q0_stokes
    #    pol['U0'],pol['erU0'] = u0_stokes,er_u0_stokes
    #    pol['pol0'],pol['erpol0'] = p0,erp0
    #    pol['angle0'],pol['erangle0'] = ang0,erang0    

    if savefile is not None:
        fits.writeto(savefile+'-pol.fits',poldeg,clobber=True)
        fits.writeto(savefile+'-angle.fits',angle,clobber=True)
        if errQ is not None:
            fits.writeto(savefile+'-erpol0.fits',erpoldeg,clobber=True)
            fits.writeto(savefile+'-erangle0.fits',erangle,clobber=True)
        #with open(savefile+'-pol.pkl', 'w') as f: #python3: 'wb'
        #    pickle.dump(pol, f)
        
    return poldeg,angle

## ---------------------------------------------------------------------
## ---------- FUNCTION TO CALCULATE POLARIZATION2 ----------------------
## ---------------------------------------------------------------------
## PURPOSE: Calculates linear polarization and angle from fit to Fdiff values
##           
## INPUT:
##        1. savefile: where to read fdiff coefficients from files
##        2. Filter at which to do this (any of 'U','b_HIGH','v_HIGH','R_SPECIAL','I_BESS') -- chrom correction
## OPTIONAL INPUT:
##        - chrom: Boolean to do chromatic correction (default: True).
##                 See routine 'chromatism_correction'
## OUTPUT: Returns poldeg,angle

def Fpolarization(savefile,filt,chrom=True):

    ##See if file exists
    if os.path.isfile(savefile+'-Fpol.fits'):
        print("   Found existing Fpol file: %s" %(savefile+'-Fpol.fits'))
        pol = fits.open(savefile+'-Fpol.fits')
        ang = fits.open(savefile+'-Fangle.fits')
        return pol[0].data,ang[0].data

    print("   Calculating F polarization and angle")
    
    ##load Fdiff
    if not os.path.isfile(savefile+'-Fdiff.fits'):
        print("   ERROR in Fpolarization: file %s not found" %(savefile+'-Fdiff.fits'))
    four = fits.open(savefile+'-Fdiff.fits')
    erfour = fits.open(savefile+'-erFdiff.fits')
    F,erF = four[0].data,erfour[0].data

    #fit vars
    from scipy import optimize
    cosfct = lambda x,a,c,d: a*np.cos(4*x/180.0*np.pi+c/180.0*np.pi)+d #b=4
    p0 = [0.1,-2.0*45,0.0] # initial guess
    
    (ypix,xpix) = np.shape(F[0])
    nang = len(F)
    ang = 22.5
    kangle = np.arange(nang)*ang
    F,erF = np.asarray(F),np.asarray(erF)
    pol,ang = np.zeros((ypix,xpix),dtype=float),np.zeros((ypix,xpix),dtype=float),
        
    ##loop pixels for fit
    for j in range(0,ypix):
        for i in range(0,xpix):
            fdiff = F[:,j,i]
            erfdiff = erF[:,j,i]
            sumfdiff = np.sum(fdiff)
            
            if (np.isfinite(sumfdiff)) & (sumfdiff != 0):
                pars,cov = optimize.curve_fit(cosfct,kangle,fdiff,sigma=erfdiff,p0=p0)
                pol[j,i] = pars[0]
                ang[j,i] = pars[1]/(-2.0)
                #tsin = cosfct(pltangle,*pars)

    ##-- chromatism correction!
    if chrom:
        ang = chromatism_correction(ang,filt)
                
    fits.writeto(savefile+'-Fpol.fits',pol,clobber=True)        
    fits.writeto(savefile+'-Fangle.fits',ang,clobber=True)        
    return pol,ang        

# ------------------------------------------------------------------
# ----------- COMPARE OFLUX/EFLUX ----------------------------------
#----------------------------------------------------------------
# PURPOSE: Compare differently measured ord/ext fluxes (AP vs PF)
# INPUT: oflux1,eflux<1,oflux2,eflux2
#        savefile: where to save plot
# EFFECTS: generates plots of comparison

def compare_flux(flux1,eflux1,flux2,eflux2,savefile,maxdiff=None):

    nangles = len(flux1)
    nstars = len(flux1[0])
    angles = map(str,np.arange(0,nangles*22.5,22.5))
    #allind = np.zeros(
    
    fig,ax = plt.subplots(nangles,figsize=(10,12))
    fig.subplots_adjust(wspace=0,hspace=0.3)
    indfin = np.zeros(np.shape(flux1),dtype=bool)
    for a in range(0,nangles):
        
        ax[a].plot([0, 1], [0, 1], transform=ax[a].transAxes)
        ax[a].scatter(flux1[a],flux2[a],label='all')
        diff = np.abs(flux1[a]-flux2[a])/flux1[a]
        #print(np.median(diff))
        if maxdiff is not None:
            ind = (diff < maxdiff)
            ax[a].scatter(flux1[a,ind],flux2[a,ind],c='orange',
                          label='diff < '+str(maxdiff))
            indfin[a] = ind
            if a==0: ax[a].legend()

        ax[a].set_yscale('log')
        ax[a].set_xscale('log')
        ax[a].set_xlim([1e3,1e6])
        ax[a].set_ylim([1e3,1e6])
        ax[a].set_xlabel('Obeam PSF '+angles[a])
        ax[a].set_ylabel('Obeam AP '+angles[a])
    plt.savefig(savefile+'-oflux_comp.png')
    fig,ax = plt.subplots(nangles,figsize=(10,12))
    fig.subplots_adjust(wspace=0,hspace=0.3)
    eindfin = np.zeros(np.shape(eflux1),dtype=bool)
    for a in range(0,nangles):
        ax[a].plot([0, 1], [0, 1], transform=ax[a].transAxes)
        ax[a].scatter(eflux1[a],eflux2[a],label='all')
        diff = np.abs(eflux1[a]-eflux2[a])/eflux1[a]
        #print(np.median(diff))
        if maxdiff is not None:
            ind = (diff < maxdiff)
            ax[a].scatter(eflux1[a,ind],eflux2[a,ind],c='orange',
                          label='diff < '+str(maxdiff))
            eindfin[a] = ind
            if a==0: ax[a].legend()
        ax[a].set_yscale('log')
        ax[a].set_xscale('log')
        ax[a].set_xlim([1e3,1e6])
        ax[a].set_ylim([1e3,1e6])
        ax[a].set_xlabel('Ebeam PSF '+angles[a])
        ax[a].set_ylabel('Ebeam AP '+angles[a])
    plt.savefig(savefile+'-eflux_comp.png')
    plt.close('all')

    tmask = np.ones(np.shape(flux1[0]),dtype=bool)
    if maxdiff is not None:
        smask = (indfin & eindfin)
        tmask = (np.sum(smask,0) == nangles)

    return tmask
        
# ------------------------------------------------------------------
# ----------- COMPARE POL/ANGLE ----------------------------------
#----------------------------------------------------------------
# PURPOSE: Compare differently measured polarization/angles
# INPUT: pol1,angle1,pol2,angle2
#        savefile: where to save plot
# EFFECTS: generates plots of comparison

def compare_polangle(pol1,ang1,pol2,ang2,savefile,xtit='QU',ytit='F',mask=None):

    tpol1,tpol2 = pol1.reshape(-1),pol2.reshape(-1)### minus!!! OJO!
    tang1,tang2 = ang1.reshape(-1),ang2.reshape(-1)
    if mask is None:
        mask = (tpol1 != 0) & (tpol2 != 0)
    loP1,upP1 = np.percentile(tpol1[mask],5),np.percentile(tpol1[mask],95)
    loP2,upP2 = np.percentile(tpol2[mask],5),np.percentile(tpol2[mask],95)
    loA1,upA1 = np.percentile(tang1[mask],5),np.percentile(tang1[mask],95)
    loA2,upA2 = np.percentile(tang2[mask],5),np.percentile(tang2[mask],95)
    
    fig,ax = plt.subplots(2,figsize=(7,7))
    ax[0].scatter(tpol1[mask],tpol2[mask])
    ax[0].plot([0,1],[0,1],'m',transform = ax[0].transAxes)
    ax[0].set_xlabel(xtit+' POL')
    ax[0].set_ylabel(ytit+' POL')
    ax[0].set_xlim([loP1,upP1]); ax[0].set_ylim([loP2,upP2])
    ax[1].scatter(tang1[mask],tang2[mask])
    ax[1].plot([0,1],[0,1],'m',transform = ax[1].transAxes)
    ax[1].set_xlabel(xtit+' ANGLE')
    ax[1].set_ylabel(ytit+' ANGLE')
    ax[1].set_xlim([loA1,upA1]); ax[1].set_ylim([loA2,upA2])
    plt.savefig(savefile+'-comppolangle.png')
    plt.close('all')
    
## -------------------------------------------------------------------
## --------- FUNCTION TO PLOT STOKES --------------------------
## -------------------------------------------------------------------
## PURPOSE: Plot stokes maps (Q, U and QUplane)
## INPUT: 1. Q map
##        2. U map
##        3. savefile where to save fig
## OPTIONAL INPUT:
##        -savefile: path+basic-file name where to plot
##        -scatter: instead of 2d histogram do scatter plot (for binned values) def:false
##        -x,y: instead of maps, indivdiual values at those positions
##        -Q0,U0: overplot those values
##        -center: if passed (and scatter True), then plots y position as colormap
##        -parfit: Perfom hyperbolic paraboloid fit to Q and U maps
##        -inla: Plot inla (read from Information folder) and residuals (need filter)
##        -fitmask: Only fit data outside this mask - def: None (all)
## OUTPUT: Nothing is returned but plots are created with savefile name and
##          extension: '..-QUplane.png','-Q.png',-U.png'

def plotstokes(Q,U,savefile=None,scatter=False,center=None,x=None,y=None,
               Q0=None,U0=None,mask=None,erQ=None,erU=None,erQ0=None,erU0=None,
               parfit=False,fitmask=None,inla=False,filt=None):
    
          
    ##basic (for fit mostly)
    ny,nx = np.shape(Q)
    ccenter=[ny/2,nx/2] if center is None else center
    xm,ym = np.linspace(0,nx-1,nx), np.linspace(0,ny-1,ny)
    xx,yy = np.meshgrid(xm-ccenter[1],ym-ccenter[0])
    rr = np.sqrt(xx**2.0+yy**2.0)    
    axx,ayy,allQ,allU = xx.reshape(-1),yy.reshape(-1),Q.reshape(-1),U.reshape(-1)
    if erQ is not None: allerQ,allerU = erQ.reshape(-1),erU.reshape(-1)
    else: allerQ,allerU = np.full(np.shape(allQ),0.0001),np.full(np.shape(allU),0.0001)
    
    ##Mask
    if mask is None:
        mask =  np.isfinite(Q) & np.isfinite(U) & (Q != 0) & (U != 0)
    if fitmask is not None:
        fitmask = (mask) & (~fitmask)
        dofitmask = True
    else:
        fitmask = mask.copy()
        dofitmask = False
    ind = fitmask.reshape(-1)
    
    ##QU plane
    fs = 18
    fig,ax = plt.subplots(1,figsize=(12,9))
    if scatter:
        #uniq = (np.unique(Q,return_index=True))[1] #doest not matter?
        loU,upU = np.percentile(U[mask],1),np.percentile(U[mask],99)
        loQ,upQ = np.percentile(Q[mask],1),np.percentile(Q[mask],99)
        h2d = np.histogram2d(Q[mask],U[mask],bins=30,range=[[loQ,upQ],[loU,upU]])
        if center is not None:
            c = ax.scatter(Q[mask],U[mask],s=6,c=np.abs(yy[mask]),cmap='Blues')
            cb = fig.colorbar(c,ax=ax)
            cb.set_label('|$y-y_0$| [pix]',rotation=270,labelpad=20,fontsize=fs)
            cb.ax.tick_params(labelsize=fs-4) 
        else:
            ax.scatter(Q[mask],U[mask],s=6)
    else:
        loU,upU = np.percentile(U[mask],5),np.percentile(U[mask],95)
        loQ,upQ = np.percentile(Q[mask],5),np.percentile(Q[mask],95)
        ax.hist2d(Q[mask],U[mask],cmap='rainbow',bins=100,range=[[loQ,upQ],[loU,upU]])
        h2d = np.histogram2d(Q[mask],U[mask],bins=30,range=[[loQ,upQ],[loU,upU]])
        
    h2dy,h2dx = np.unravel_index(np.argmax(h2d[0]),np.shape(h2d[0]))
    dQ,dU = 0.5*(h2d[1][1]-h2d[1][0]),0.5*(h2d[2][1]-h2d[2][0])
    Qmax,Umax = h2d[1][h2dx]+dQ,h2d[2][h2dy]+dU
    Qmed,Umed = np.median(Q[mask]),np.median(U[mask])
    ax.axvline(0,c='gray',linestyle='--')
    ax.axhline(0,c='gray',linestyle='--')
    ax.scatter(Qmed,Umed,s=50,marker='x',c='orange',label='Median')
    #ax.scatter(Qmax,Umax,s=50,marker='+',c='purple')
    #ax.errorbar(Qmax,Umax,xerr=dQ,yerr=dU,markersize=5,marker='o',color='purple',label='Peak')
    if (Q0 is not None) & (U0 is not None):
        if erQ0 is not None:
            ax.errorbar(Q0,U0,xerr=erQ0,yerr=erU0,markersize=5,marker='o',color='cyan',label='Center')
        else:
            ax.scatter(Q0,U0,s=50,marker='x',c='cyan',label='Center')
    ax.set_ylabel("U",fontsize=fs)
    ax.set_xlabel("Q",fontsize=fs)
    ax.tick_params(labelsize=fs-4)
    ax.set_xlim([loQ,upQ])
    ax.set_ylim([loU,upU])
    ax.legend(fontsize=fs-2,loc='lower right')

    ##Typical error: too small!
    if erQ is not None:
        print("      Median Q/U error: %.4e,%.4e" %(np.median(erQ[mask]),np.median(erU[mask])))
        #ax.errorbar(upQ,upU,xerr=np.median(erQ[mask]),yerr=np.median(erQ[mask]),marker="")
    
    if savefile is not None:
        plt.savefig(savefile+'-QUplane.png')

    ## plot Q,U    
    from matplotlib import cm
    fig,ax = plt.subplots(1)
    Q[~mask],U[~mask] = np.nan,np.nan
    loQ = np.percentile(Q[mask].flatten(), 5) 
    upQ = np.percentile(Q[mask].flatten(), 95)
    if x is None:
        c = ax.imshow(Q,clim=(loQ,upQ),cmap='rainbow',interpolation='nearest')
    else:
        c = ax.scatter(x[mask],y[mask],c=Q[mask],
                       norm=cm.colors.Normalize(vmax=upQ,vmin=loQ),s=3)
    ax.set_xlabel('x [pix]',fontsize=fs)
    ax.set_ylabel('y [pix]',fontsize=fs)
    ax.tick_params(labelsize=fs-4) 
    cb = fig.colorbar(c,ax=ax)
    cb.ax.tick_params(labelsize=fs-4) 
    cb.set_label('Q', rotation=270, labelpad=15,fontsize=fs)
    ax.set_ylim(ax.get_ylim()[::-1])
    if savefile is not None:
        fig.savefig(savefile+'-Q.png')

    fig,ax = plt.subplots(1)
    loU = np.percentile(U[mask].flatten(), 5)
    upU = np.percentile(U[mask].flatten(), 95)
    if x is None:
        c = ax.imshow(U,clim=(loU,upU),cmap='rainbow',interpolation='nearest')
    else:
        c = ax.scatter(x[mask],y[mask],c=U[mask],
                       norm=cm.colors.Normalize(vmax=upU,vmin=loU),s=5)
    ax.set_xlabel('x [pix]',fontsize=fs)
    ax.set_ylabel('y [pix]',fontsize=fs)
    ax.tick_params(labelsize=fs-4) 
    cb = fig.colorbar(c,ax=ax)
    cb.ax.tick_params(labelsize=fs-4) 
    cb.set_label('U', rotation=270, labelpad=15,fontsize=fs)
    ax.set_ylim(ax.get_ylim()[::-1])
    if savefile is not None:
        fig.savefig(savefile+'-U.png')

    ## HYPERBOL PARABOLOID FIT
    if parfit:
        fx,fy,fQ,fU,ferQ,ferU = axx[ind],ayy[ind],allQ[ind],allU[ind],allerQ[ind],allerU[ind]
        if scatter:
            uniq = (np.unique(fQ,return_index=True))[1]
            fx,fy,fQ,fU,ferQ,ferU = fx[uniq],fy[uniq],fQ[uniq],fU[uniq],ferQ[uniq],ferU[uniq]

        from scipy import optimize
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        #- rotated hyperbolic paraboloid
        rothypparaboloid = lambda xy,a,b,theta,x0,y0:\
                           ((xy[0]-x0)*np.sin(theta)+(xy[1]-y0)*np.cos(theta))**2.0/b**2.0 - \
                           ((xy[0]-x0)*np.cos(theta)-(xy[1]-y0)*np.sin(theta))**2.0/a**2.0  
        p0 = [1.0,1.0,0.0,0.0,0.0]

        #- rotated hyperbolic paraboloid with center value
        crothypparaboloid = lambda xy,a,b,theta,x0,y0,cst:\
                              ((xy[0]-x0)*np.sin(theta)+(xy[1]-y0)*np.cos(theta))**2.0/b**2.0 - \
                              ((xy[0]-x0)*np.cos(theta)-(xy[1]-y0)*np.sin(theta))**2.0/a**2.0 + cst 
        cp0 = [1.0,1.0,0.0,0.0,0.0,0.0]
        
        ## --------- Q
        try: 
            fitQhpars,fitQhpcov = optimize.curve_fit(rothypparaboloid,(fx,fy),fQ,p0=p0,sigma=ferQ)
            fitQerhpars = np.sqrt(np.diag(fitQhpcov))
        except:
            fitQhpars,fitQerhpars = p0,np.zeros(len(p0))*np.nan
        Qhparr = rothypparaboloid((axx,ayy),*fitQhpars)
        Qhp = Qhparr.reshape((ny,nx))
        resQhp = Q - Qhp
        print("      RotHyperParaboloid fit parameters Q: %.4e,%.4e,%.4e,%.4e,%.4e"
              %(fitQhpars[0],fitQhpars[1],fitQhpars[2],fitQhpars[3],fitQhpars[4]))
        print("      RotHyperParaboloid fit error parameters Q: %.4e,%.4e,%.4e,%.4e,%.4e"
              %(fitQerhpars[0],fitQerhpars[1],fitQerhpars[2],fitQerhpars[3],fitQerhpars[4]))
        medQhpres = np.median(resQhp[mask])
        stdQhpres = np.median(np.abs(resQhp[mask]-medQhpres))
        print("      Residual median/MAD Q: %.4e,%.4e" %(medQhpres,stdQhpres))

   
        try: 
            fitcQhpars,fitcQhpcov = optimize.curve_fit(crothypparaboloid,(fx,fy),fQ,
                                                       p0=cp0,sigma=ferQ)
            fitcQerhpars = np.sqrt(np.diag(fitcQhpcov))
        except:
            fitcQhpars,fitcQerhpars = cp0,np.zeros(len(cp0))*np.nan
        cQhparr = crothypparaboloid((axx,ayy),*fitcQhpars)
        cQhp = cQhparr.reshape((ny,nx))
        rescQhp = Q - cQhp
        print("      CstRotHyperParaboloid fit parameters Q: %.4e,%.4e,%.4e,%.4e,%.4e,%.4e"
              %(fitcQhpars[0],fitcQhpars[1],fitcQhpars[2],fitcQhpars[3],fitcQhpars[4],fitcQhpars[5]))
        print("      CstRotHyperParaboloid fit error parameters Q: %.4e,%.4e,%.4e,%.4e,%.4e,%.4e"
              %(fitcQerhpars[0],fitcQerhpars[1],fitcQerhpars[2],fitcQerhpars[3],fitcQerhpars[4],fitcQerhpars[5]))
        medcQhpres = np.median(rescQhp[mask])
        stdcQhpres = np.median(np.abs(rescQhp[mask]-medcQhpres))
        print("      Residual median/MAD Q: %.4e,%.4e" %(medcQhpres,stdcQhpres))

        fits.writeto(savefile+'-Qmodel.fits',cQhp,clobber=True) 
        np.savetxt(savefile+'-Qmodel.dat',(fitcQhpars,fitcQerhpars),fmt='%8e',
                   header='CstRotHyperParaboloid fit Q parameters\nResidual median/MAD Q: %.4e %.4e'
                   %(medcQhpres,stdcQhpres))
        
        
        ## --------- U
        try:
            fitUhpars,fitUhpcov = optimize.curve_fit(rothypparaboloid,(fx,fy),fU,p0=p0,sigma=ferU)
            fitUerhpars = np.sqrt(np.diag(fitUhpcov))
        except:
            fitUhpars,fitUerhpars = p0,np.zeros(len(p0))*np.nan
        Uhparr = rothypparaboloid((axx,ayy),*fitUhpars)
        Uhp = Uhparr.reshape((ny,nx))
        resUhp = U - Uhp
        print("      RotHyperParaboloid fit parameters U: %.4e,%.4e,%.4e,%.4e,%.4e"
              %(fitUhpars[0],fitUhpars[1],fitUhpars[2],fitUhpars[3],fitUhpars[4]))
        print("      RotHyperParaboloid fit error parameters U: %.4e,%.4e,%.4e,%.4e,%.4e"
              %(fitUerhpars[0],fitUerhpars[1],fitUerhpars[2],fitUerhpars[3],fitUerhpars[4]))
        medUhpres = np.median(resUhp[mask])
        stdUhpres = np.median(np.abs(resUhp[mask]-medUhpres))
        print("      Residual median/MAD U: %.4e,%.4e" %(medUhpres,stdUhpres)) 
        
        try: 
            fitcUhpars,fitcUhpcov = optimize.curve_fit(crothypparaboloid,(fx,fy),fU,p0=cp0,sigma=ferU)
            fitcUerhpars = np.sqrt(np.diag(fitcUhpcov))
        except:
            fitcUhpars,fitcUerhpars = cp0,np.zeros(len(cp0))*np.nan
        cUhparr = crothypparaboloid((axx,ayy),*fitcUhpars)
        cUhp = cUhparr.reshape((ny,nx))
        rescUhp = U - cUhp
        print("      CstRotHyperParaboloid fit parameters U: %.4e,%.4e,%.4e,%.4e,%.4e,%.4e"
              %(fitcUhpars[0],fitcUhpars[1],fitcUhpars[2],fitcUhpars[3],fitcUhpars[4],fitcUhpars[5]))
        print("      CstRotHyperParaboloid fit error parameters U: %.4e,%.4e,%.4e,%.4e,%.4e,%.4e"
              %(fitcUerhpars[0],fitcUerhpars[1],fitcUerhpars[2],fitcUerhpars[3],fitcUerhpars[4],fitcUerhpars[5]))
        medcUhpres = np.median(rescUhp[mask])
        stdcUhpres = np.median(np.abs(rescUhp[mask]-medcUhpres))
        print("      Residual median/MAD U: %.4e,%.4e" %(medcUhpres,stdcUhpres))

        fits.writeto(savefile+'-Umodel.fits',cUhp,clobber=True) 
        np.savetxt(savefile+'-Umodel.dat',(fitcUhpars,fitcUerhpars),fmt='%8e',
                   header='CstRotHyperParaboloid fit U parameters\nResidual median/MAD U: %.4e %.4e'
                   %(medcUhpres,stdcUhpres))

        
        ## PLOT model and residual
        fs = 8
        
        ## Q plot
        fig,ax = plt.subplots(3,1,figsize=(4,7),sharex=True,sharey=True)
        fig.subplots_adjust(hspace=0,wspace=0.1)
        if x is None:
            im1 = ax[0].imshow(Q,clim=(loQ,upQ),cmap='rainbow',interpolation='nearest')
        else:
            im1 = ax[0].scatter(x[mask],y[mask],c=Q[mask],
                           norm=cm.colors.Normalize(vmax=upQ,vmin=loQ),s=3)
        if dofitmask: ax[0].imshow(~fitmask, cmap='binary', alpha=0.4)
        ax[0].set_ylabel('y [pix]',fontsize=fs)
        ax[0].invert_yaxis()        
        ax[0].text(0.70, 0.1, 'Q map', horizontalalignment='center',fontsize=fs,
                   verticalalignment='center', transform=ax[0].transAxes)
        ax[0].tick_params(labelsize=fs-2)
        im2 = ax[1].imshow(cQhp,clim=(loQ,upQ),cmap='rainbow')
        ax[1].set_ylabel('y [pix]',fontsize=fs)
        ax[1].invert_yaxis()
        ax[1].text(0.60, 0.1, 'Rotated HyperParaboloid', horizontalalignment='center',fontsize=fs,
                     verticalalignment='center', transform=ax[1].transAxes)
        ax[1].tick_params(labelsize=fs-2)
        lorQ,uprQ = np.percentile(rescQhp[mask],5),np.percentile(rescQhp[mask],95)
        imres = ax[2].imshow(rescQhp,clim=(lorQ,uprQ),cmap='binary')
        ax[2].set_ylabel('y [pix]',fontsize=fs)
        ax[2].set_xlabel('x [pix]',fontsize=fs)
        ax[2].invert_yaxis()
        ax[2].text(0.70, 0.1, 'Residual', horizontalalignment='center',fontsize=fs,
                   verticalalignment='center', transform=ax[2].transAxes)
        ax[2].tick_params(labelsize=fs-2)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im1, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im2, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(imres, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2)
        plt.savefig(savefile+'-Qmodel.png')

        ## U plot
        fig,ax = plt.subplots(3,1,figsize=(4,7),sharex=True,sharey=True)
        fig.subplots_adjust(hspace=0,wspace=0.1)
        if x is None:
            im1 = ax[0].imshow(U,clim=(loU,upU),cmap='rainbow',interpolation='nearest')
        else:
            im1 = ax[0].scatter(x[mask],y[mask],c=U[mask],
                           norm=cm.colors.Normalize(vmax=upU,vmin=loU),s=3)
        if dofitmask: ax[0].imshow(~fitmask, cmap='binary', alpha=0.4)
        ax[0].set_ylabel('y [pix]',fontsize=fs)
        ax[0].invert_yaxis()        
        ax[0].text(0.70, 0.1, 'U map', horizontalalignment='center',fontsize=fs,
                   verticalalignment='center', transform=ax[0].transAxes)
        ax[0].tick_params(labelsize=fs-2)
        im2 = ax[1].imshow(cUhp,clim=(loU,upU),cmap='rainbow')
        ax[1].set_ylabel('y [pix]',fontsize=fs)
        ax[1].invert_yaxis()
        ax[1].text(0.60, 0.1, 'Rotated HyperParaboloid', horizontalalignment='center',fontsize=fs,
                     verticalalignment='center', transform=ax[1].transAxes)
        lorU,uprU = np.percentile(rescUhp[mask],5),np.percentile(rescUhp[mask],95)
        ax[1].tick_params(labelsize=fs-2)
        imres = ax[2].imshow(rescUhp,clim=(lorU,uprU),cmap='binary')
        ax[2].set_ylabel('y [pix]',fontsize=fs)
        ax[2].set_xlabel('x [pix]',fontsize=fs)
        ax[2].invert_yaxis()
        ax[2].text(0.70, 0.1, 'Residual', horizontalalignment='center',fontsize=fs,
                   verticalalignment='center', transform=ax[2].transAxes)
        ax[2].tick_params(labelsize=fs-2)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im1, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im2, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(imres, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2)
        plt.savefig(savefile+'-Umodel.png')

    ## INLA 
    if inla:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        if filt is None:
            print("   ERROR IN PLOTSTOKES: need to pass filter in INLA mode")
            sys.exit('stop')
        HQi,Qi = read_fits(home+"/crisp/FORS2-POL/Information/"+filt+"_inlaQ.fits")
        HUi,Ui = read_fits(home+"/crisp/FORS2-POL/Information/"+filt+"_inlaU.fits") 
            
        fx,fy,fQ,fU,ferQ,ferU = axx[ind],ayy[ind],allQ[ind],allU[ind],allerQ[ind],allerU[ind]
        if scatter:
            uniq = (np.unique(fQ,return_index=True))[1]
            fx,fy,fQ,fU,ferQ,ferU = fx[uniq],fy[uniq],fQ[uniq],fU[uniq],ferQ[uniq],ferU[uniq]

        resQi = Q - Qi
        medQires = np.median(resQi[mask])
        stdQires = np.median(np.abs(resQi[mask]-medQires))
        print("      Residual INLA Q median/MAD: %.4e,%.4e" %(medQires,stdQires))
            
        resUi = U - Ui
        medUires = np.median(resUi[mask])
        stdUires = np.median(np.abs(resUi[mask]-medUires))
        print("      Residual INLA U median/MAD: %.4e,%.4e" %(medUires,stdUires)) 
        
        ## PLOT model and residual
        fs = 8
        ## Q plot
        fig,ax = plt.subplots(3,1,figsize=(4,7),sharex=True,sharey=True)
        fig.subplots_adjust(hspace=0,wspace=0.1)
        if x is None:
            im1 = ax[0].imshow(Q,clim=(loQ,upQ),cmap='rainbow',interpolation='nearest')
        else:
            im1 = ax[0].scatter(x[mask],y[mask],c=Q[mask],
                           norm=cm.colors.Normalize(vmax=upQ,vmin=loQ),s=3)
        ax[0].set_ylabel('y [pix]',fontsize=fs)
        ax[0].invert_yaxis()        
        ax[0].text(0.70, 0.1, 'Q map', horizontalalignment='center',fontsize=fs,
                   verticalalignment='center', transform=ax[0].transAxes)
        im2 = ax[1].imshow(Qi,clim=(loQ,upQ),cmap='rainbow')
        ax[0].tick_params(labelsize=fs-2)
        ax[1].set_ylabel('y [pix]',fontsize=fs)
        ax[1].invert_yaxis()
        ax[1].text(0.70, 0.1, 'INLA', horizontalalignment='center',fontsize=fs,
                     verticalalignment='center', transform=ax[1].transAxes)
        ax[1].tick_params(labelsize=fs-2)
        lorQ,uprQ = np.percentile(resQi[mask],5),np.percentile(resQi[mask],95)
        imres = ax[2].imshow(resQi,clim=(lorQ,uprQ),cmap='binary')
        ax[2].set_ylabel('y [pix]',fontsize=fs)
        ax[2].set_xlabel('x [pix]',fontsize=fs)
        ax[2].invert_yaxis()
        ax[2].text(0.70, 0.1, 'Residual', horizontalalignment='center',fontsize=fs,
                   verticalalignment='center', transform=ax[2].transAxes)
        ax[2].tick_params(labelsize=fs-2)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im1, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im2, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(imres, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2)
        plt.savefig(savefile+'-Qinla.png')

        ## U plot
        fig,ax = plt.subplots(3,1,figsize=(4,7),sharex=True,sharey=True)
        fig.subplots_adjust(hspace=0,wspace=0.1)
        if x is None:
            im1 = ax[0].imshow(U,clim=(loU,upU),cmap='rainbow',interpolation='nearest')
        else:
            im1 = ax[0].scatter(x[mask],y[mask],c=U[mask],
                           norm=cm.colors.Normalize(vmax=upU,vmin=loU),s=3)
        ax[0].set_ylabel('y [pix]',fontsize=fs)
        ax[0].invert_yaxis()        
        ax[0].text(0.70, 0.1, 'U map', horizontalalignment='center',fontsize=fs,
                   verticalalignment='center', transform=ax[0].transAxes)
        ax[0].tick_params(labelsize=fs-2)
        im2 = ax[1].imshow(Ui,clim=(loQ,upQ),cmap='rainbow')
        ax[1].set_ylabel('y [pix]',fontsize=fs)
        ax[1].invert_yaxis()
        ax[1].text(0.70, 0.1, 'INLA', horizontalalignment='center',fontsize=fs,
                     verticalalignment='center', transform=ax[1].transAxes)
        ax[1].tick_params(labelsize=fs-2)
        lorU,uprU = np.percentile(resUi[mask],5),np.percentile(resUi[mask],95)
        imres = ax[2].imshow(resUi,clim=(lorU,uprU),cmap='binary')
        ax[2].set_ylabel('y [pix]',fontsize=fs)
        ax[2].set_xlabel('x [pix]',fontsize=fs)
        ax[2].invert_yaxis()
        ax[2].text(0.70, 0.1, 'Residual', horizontalalignment='center',fontsize=fs,
                   verticalalignment='center', transform=ax[2].transAxes)
        ax[2].tick_params(labelsize=fs-2)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im1, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im2, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(imres, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2)
        plt.savefig(savefile+'-Uinla.png')
        
        
    plt.close('all')
    return Qmax,Umax,Qmed,Umed

## -------------------------------------------------------------------
## --------- FUNCTION TO PLOT POLARIZATION FOR FIELD STARS-------------
## -------------------------------------------------------------------
## PURPOSE: Plot polarization/angle maps
## INPUT: 1. x-coordinates of stars
##        2. y-coordinates of stars
##        3. Polarization of stars
##        4. Angle of stars
## OPTIONAL INPUT:
##        -center: center of map where to start if doing binning
##        -savefile: path+basic-file name where to save results
## OUTPUT: Nothing is returned but plots are created with savefile name and
##          extension ('..-pol.png', '..-angle.png', '..pol-angle.png')
def xyplotpol(x,y,pol,angle,center=None,savefile=None,mask=None):

    ##OJO: MISSING POL-ANGLE????
    
    print("   Plotting pol/angle of stars ")
    
    from matplotlib import cm

    if center is None:
        center = [1025,1226]

    rad = np.sqrt((x-center[1])**2.0+(y-center[0])**2.0)
    singpol,singangle = pol.reshape(-1),angle.reshape(-1)#arcsinh(pol)
    if mask is None: mask = (singpol > 0) #& (singpol < 0.05)
    lo = np.percentile(singpol[mask], 5)
    up = np.percentile(singpol[mask], 95)
    alo = np.percentile(singangle[mask], 5)
    aup = np.percentile(singangle[mask], 95)

    fs = 16
    f, ax = plt.subplots(1)#,1,sharex=True, sharey=True)
    #ax[0].tripcolor(x,y,singpol)
    c=ax.scatter(x[mask],y[mask],c=singpol[mask],#cmap=cm.PRGn,
                 norm=cm.colors.Normalize(vmax=up, vmin=lo),s=3)
    cb = f.colorbar(c,ax=ax)
    cb.set_label('Polarization',rotation=270,labelpad=15,fontsize=fs)
    ax.set_ylabel('y (pix)',fontsize=fs)
    ax.set_xlabel('x (pix)',fontsize=fs)
    ax.tick_params(labelsize=fs-4)
    ax.set_ylim([0,2049])
    plt.savefig(savefile+'-pol.png')

    f, ax = plt.subplots(1)#,1, sharex=True, sharey=True)#figsize=(10,10),
    c = ax.scatter(x[mask],y[mask],c=singangle[mask],#cmap=cm.PRGn,
                   norm=cm.colors.Normalize(vmax=aup, vmin=alo),s=3)
    cb = f.colorbar(c,ax=ax)
    cb.set_label('Angle',rotation=270,labelpad=15,fontsize=fs)
    ax.set_ylabel('y (pix)',fontsize=fs)
    ax.set_xlabel('x (pix)',fontsize=fs)
    ax.tick_params(labelsize=fs-4)
    plt.savefig(savefile+'-angle.png')
    plt.close(f)
    
    #ax[1].tricontourf(x[m],y[m],singpol[m])#,10)#levels=[0,0.05,0.1,0.15,0.2,0.4,1]) #,20) choose 20 contour levels, just to show how good its interpolation is
    #ax[1].plot(x[m],y[m], 'ko ')
    #ax[0].plot(x,y, 'ko ',symsize=0.5)
   
    
    f,ax = plt.subplots(4,1,figsize=(6,9))
    ax[0].hist(singpol[mask],bins=np.arange(lo,up,(up-lo)/10.0))#,log=True)
    ax[0].set_xlim([lo,up])
    ax[0].set_ylabel('Nstars')
    ax[0].set_xlabel('Pol')
    ax[1].plot(rad[mask],singpol[mask],'o',markersize=5)
    ax[1].set_ylim([lo,up])
    ax[1].set_ylabel('Pol')
    ax[1].set_xlabel('radius(pix)')
    ax[2].plot(x[mask],singpol[mask],'o',markersize=5)
    ax[2].set_ylim([lo,up])
    ax[2].set_ylabel('Pol')
    ax[2].set_xlabel('x(pix)')
    ax[3].plot(y[mask],singpol[mask],'o',markersize=5)
    ax[3].set_ylim([lo,up])
    ax[3].set_ylabel('Pol')
    ax[3].set_xlabel('y(pix)')
    plt.savefig(savefile+'-pol-radius.png')
    plt.close(f)
    
    #from scipy.interpolate import interp2d
    #fpol = interp2d(x[m],y[m],singpol[m],kind="linear")
    #fang = interp2d(x[m],y[m],singangle[m],kind="linear")
    #fimg = interp2d(x[m],y[m],singimage[m],kind="linear")
    #x_coords = np.arange(min(x),max(x)+1)
    #y_coords = np.arange(min(y),max(y)+1)
    #pol = fpol(x_coords,y_coords)
    #angle = fang(x_coords,y_coords)
    #image = fimg(x_coords,y_coords)
    #if Q is not None:
    #    singQ,singU = Q.reshape(-1),U.reshape(-1)
    #    fQ = interp2d(x[m],y[m],singQ[m],kind='linear')
    #    fU = interp2d(x[m],y[m],singU[m],kind='linear')
    #    Q = fQ(x_coords,y_coords)
    #    U = fU(x_coords,y_coords)
## -------------------------------------------------------------------
## --------- FUNCTION TO PLOT POLARIZATION --------------------------
## -------------------------------------------------------------------
## PURPOSE: Plot polarization/angle maps
## INPUT: 
##        1. Polarization map
##        2. Angle map
## OPTIONAL INPUT:
##        -img: image map
##        -step: pixel size of window in case of doing binning
##        -center: center of map where to start if doing binning
##        -savefile: path+basic-file name where to save results
##        -fitradius: to plot zoom-in of region inside this radius
## OUTPUT: Nothing is returned but plots are created with savefile name and
##          extension ('..-pol.png', '..-angle.png', '..pol-angle.png')

def plotpol(pol,angle,erpol=None,erangle=None,image=None,polrange=None,
            step=7,center=None,savefile=None,x=None,y=None,fitradius=None):

    print("   Plotting pol/angle maps ")
    #CONTOURS!
            
    xi,xf,yi,yf = 188,1862,434,1965
    xini,yini = 0,0
    if center is not None:
        xstep,ystep = np.int(center[0])/step,np.int(center[1])/step
        xini,yini = np.int(center[0])-xstep*step,np.int(center[1])-ystep*step

    mask = np.isfinite(pol) & (pol > 0)#np.ma.masked_where(pol > 0,pol)    

    #plots: pol and angle indvidually
    fs = 16
    fig,ax = plt.subplots(1)
    img = pol#np.arcsinh(pol)
    timg = np.copy(img[:,xi:xf])
    timg[timg == 0] = np.nan
    if polrange is None:
        lo = np.percentile(timg[np.isfinite(timg)].flatten(), 5)
        up = np.percentile(timg[np.isfinite(timg)].flatten(), 95)
    else:
        lo,up = polrange[0],polrange[1]
        
    c = ax.imshow(img,clim=(lo,up),cmap='rainbow')
    ax.set_xlabel('x [pix]',fontsize=fs)
    ax.set_ylabel('y [pix]',fontsize=fs)
    ax.tick_params(labelsize=fs-4)
    cb = fig.colorbar(c,ax=ax)
    cb.ax.tick_params(labelsize=fs-4) 
    cb.set_label('Polarization', rotation=270, labelpad=15,fontsize=fs)
    if x is not None:
        ax.scatter(x,y,200,facecolors='None')
    ax.set_ylim(ax.get_ylim()[::-1])
    fig.savefig(savefile+'-pol.png')

    if erpol is not None:
        fig,ax = plt.subplots(1)
        img = erpol#np.arcsinh(pol)
        timg = np.copy(img[:,xi:xf])
        timg[timg == 0] = np.nan
        lo = np.percentile(timg[np.isfinite(timg)].flatten(), 5)
        up = np.percentile(timg[np.isfinite(timg)].flatten(), 95)
        c = ax.imshow(img,clim=(lo,up),cmap='binary')
        ax.tick_params(labelsize=fs-4)
        ax.set_xlabel('x [pix]',fontsize=fs)
        ax.set_ylabel('y [pix]',fontsize=fs)
        cb = fig.colorbar(c,ax=ax)
        cb.ax.tick_params(labelsize=fs-4) 
        cb.set_label('Polarization Error', rotation=270, labelpad=15,fontsize=fs)
        if x is not None:
            ax.scatter(x,y,200,facecolors='None')
        ax.set_ylim(ax.get_ylim()[::-1])
        fig.savefig(savefile+'-erpol.png')
    
    fig,ax = plt.subplots(1)
    img = angle#np.arcsinh(angle)
    timg = np.copy(img[:,xi:xf])
    timg[timg == 0] = np.nan
    lo = np.percentile(timg[np.isfinite(timg)].flatten(), 5)
    up = np.percentile(timg[np.isfinite(timg)].flatten(), 95)
    c = ax.imshow(img,clim=(lo,up),cmap='rainbow')
    ax.set_xlabel('x [pix]',fontsize=fs)
    ax.set_ylabel('y [pix]',fontsize=fs)
    ax.tick_params(labelsize=fs-4)
    cb = fig.colorbar(c,ax=ax)
    cb.ax.tick_params(labelsize=fs-4) 
    cb.set_label('Angle', rotation=270,labelpad=15,fontsize=fs)
    if x is not None:
        ax.scatter(x,y,200,facecolors='None')
    ax.set_ylim(ax.get_ylim()[::-1])
    fig.savefig(savefile+'-angle.png')

    if erangle is not None:
        fig,ax = plt.subplots(1)
        img = erangle#np.arcsinh(pol)
        timg = np.copy(img[:,xi:xf])
        timg[timg == 0] = np.nan
        lo = np.percentile(timg[np.isfinite(timg)].flatten(), 5)
        up = np.percentile(timg[np.isfinite(timg)].flatten(), 95)
        c = ax.imshow(img,clim=(lo,up),cmap='binary')
        ax.set_xlabel('x [pix]',fontsize=fs)
        ax.set_ylabel('y [pix]',fontsize=fs)
        ax.tick_params(labelsize=fs-4)
        cb = fig.colorbar(c,ax=ax)
        cb.ax.tick_params(labelsize=fs-4) 
        cb.set_label('Angle Error', rotation=270, labelpad=15,fontsize=fs)
        if x is not None:
            ax.scatter(x,y,200,facecolors='None')
        ax.set_ylim(ax.get_ylim()[::-1])
        fig.savefig(savefile+'-erangle.png')
    
    vals = sigma_clip(pol[np.isfinite(pol) & (pol >0)],sigma=4.0)
    scale = step/vals.max()*5
    
    (ny,nx)=pol.shape
    x,y = np.meshgrid(np.arange(xini,nx,step),np.arange(yini,ny,step))
    u = pol[yini::step,xini::step]*np.cos(angle[yini::step,xini::step]/180*np.pi)*scale
    v = pol[yini::step,xini::step]*np.sin(angle[yini::step,xini::step]/180*np.pi)*scale
    xymask = ((x >= xi) & (x <=xf) & (y>= yi) & (y <=yf))
 
    fig,ax = plt.subplots(1,figsize=(9,9))
    if image is not None:
        img = image
        lo = np.percentile(img[np.isfinite(img)].flatten(), 5)
        up = np.percentile(img[np.isfinite(img)].flatten(), 99.5)
        ax.imshow(img,clim=(lo,up))
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    ax.quiver(x[xymask],y[xymask],u[xymask],v[xymask],color='black',
              headlength=0, pivot='middle', scale=3, linewidth=3.5, units='xy', angles='uv',
              width=3.0, alpha= 1.0, headwidth=1,headaxislength=0)
    ax.set_ylim(ax.get_ylim()[::-1])#ax.invert_yaxis()#
    fig.savefig(savefile+'-pol-angle.png')
    
    ##Within fitradius
    if fitradius is not None:
        fr = int(fitradius)
        subpol = pol[center[1]-fr:center[1]+fr,center[0]-fr:center[0]+fr]
        subvals = sigma_clip(subpol[np.isfinite(subpol) & (subpol >0)],sigma=4.0)
        sscale = step/subvals.max()*2
        subu,subv = u/scale*sscale,v/scale*sscale
        
        fig,ax = plt.subplots(1,figsize=(9,9))
        if image is not None:
            subimg = image[center[1]-fr:center[1]+fr,center[0]-fr:center[0]+fr]
            slo = np.percentile(subimg[np.isfinite(subimg)].flatten(), 5)
            sup = np.percentile(subimg[np.isfinite(subimg)].flatten(), 99.5)
            ax.imshow(img,clim=(slo,sup))
        ax.set_xlabel('x [pix]')
        ax.set_ylabel('y [pix]')
        ax.set_xlim([center[0]-fr,center[0]+fr])
        ax.set_ylim([center[1]-fr,center[1]+fr])
        ax.quiver(x[xymask],y[xymask],subu[xymask],subv[xymask],color='black',
                  headlength=0, pivot='middle', scale=3, linewidth=3.5, units='xy', angles='uv',
                  width=1.5, alpha= 1.0, headwidth=1,headaxislength=0)
        ax.set_ylim(ax.get_ylim()[::-1])
        fig.savefig(savefile+'-sub-pol-angle.png')

    
    fig,ax = plt.subplots(1,figsize=(9,9))
    if image is not None:
        img = image
        lo = np.percentile(img[np.isfinite(img)].flatten(), 5)
        up = np.percentile(img[np.isfinite(img)].flatten(), 99.5)
        ax.imshow(img,clim=(lo,up))
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    ax.quiver(x[xymask],y[xymask],u[xymask],v[xymask],color='black',
              headlength=0, pivot='middle', scale=3, linewidth=3.5, units='xy', angles='uv',
              width=3.0, alpha= 1.0, headwidth=1,headaxislength=0)
    ax.set_ylim(ax.get_ylim()[::-1])
    fig.savefig(savefile+'-pol-angle.png')

    
    plt.close('all')
            
    ##-- generate length of lines based on maximum
    ## def: maxlength=7
    #mask = (pol > 0)
    #polma = np.ma.array(pol,mask=~mask)#excludes True mask values of computation
    #vals = sigma_clip(pol[mask],sigma=4.0)
    #maxpol = vals.max()#np.nanmax(pol)
    #length = pol/maxpol*maxlength
    #(ypix,xpix) = np.shape(pol)
    #y,x = np.mgrid[0:ypix,0:xpix]
    #endx = x + length * np.sin(angle)
    #endy = y + length * np.cos(angle)
    
    #fig = plt.figure()
    #ax = plt.subplot(111)
    #ax.set_ylim([0, ypix])
    #ax.set_xlim([0, xpix])
    #for i in range(0,xpix):
    #    for j in range(0,ypix):
    #        if pol[j,i] > 0:
    #            ax.plot([x[j,i],endx[j,i]],[y[j,i],endy[j,i]],color='black',linewidth=0.3)
    ##--contour plot
    ##ax.contour(x[mask],y[mask],pol[mask],5)
    #ax.contour(x,y,polma)#,5)
                
    #if savefile is not None:            
    #    plt.savefig(savefile+'.png')     

## ----------------------------------------------------------------------
## --------------- COMBINE IMAGES ------------------------------------------
## -----------------------------------------------------------------------
## PURPOSE: simple median of different images
## INPUT: list of file names of images
##        savefile:    output combined file
##        method  :    'median' (def),
##                     'absmedian' (divide each img by its median)
##                     'mean'
##                     'weighted' (need errors) 
## OPTIONAL INPUT:
##        erfiles :    list of errfiles images
##        erstd   :    include std into error
##        align   :    Align images using center ra,dec
##        RA,DEC  :    Center Ra and dec to align (def: use from header)
##        

def combine_images(files,savefile,method='median',erfiles=None,erstd=False,
                   ra=None,dec=None,align=False,pixscale=0.126,binning=2):

    ## see if file exists
    if os.path.isfile(savefile):
        print("   Found existing combined file %s" %savefile)
        head,data = read_fits(savefile)
        return data
    
    ## load images
    allmed = np.zeros(len(files))
    for f in range(0,len(files)):
        head,data = read_fits(files[f])
        erdata = None
        
        ## is error included?
        if np.size(np.shape(data)) == 3:
            erdata = data[1,:,:]
            data = data[0,:,:]
        
        if f == 0:
            alldata = np.zeros((len(files),data.shape[0],data.shape[1]),dtype=float)
            allerror = np.zeros((len(files),data.shape[0],data.shape[1]),dtype=float)
            allra,alldec = np.zeros(len(files),dtype=float),np.zeros(len(files),dtype=float)
            firsthead = head
        allmed[f] = np.nanmedian(data) 
        if method == 'absmedian': data /= allmed[f]
        alldata[f,:,:] = data
        
                
        if erfiles is not None: erhead,erdata = read_fits(erfiles[f])
        if erdata is not None: allerror[f,:,:] = erdata
        
        if ra is None:
            if 'RA' in head:
                allra[f],alldec[f] = np.float(head['RA']),np.float(head['DEC'])
        else:
            allra[f],alldec[f] = ra[f],dec[f]

    totalmed = np.nanmedian(allmed)
    if method == 'absmedian': alldata*=totalmed
    
    ## align
    if align:
        from astropy.coordinates import SkyCoord
        positions = [SkyCoord(ira,idec,frame='fk5',unit='deg') for ira,idec in zip(allra,alldec)]
        position1 = positions[0]

        #minpixra,maxpixra = np.min(pixsepra),np.max(pixsepra)
        #minpixdec,maxpixdec = np.min(pixsepdec),np.max(pixsepdec)
  
        for f in range(1,len(files)):
            sepra,sepdec = position1.spherical_offsets_to(positions[f])
            pixsepra = np.int(sepra.value/pixscale*3600/binning)
            pixsepdec = np.int(sepdec.value/pixscale*3600/binning)
            alldata[f,:,:] = shiftim(alldata[f,:,:],pixsepra,pixsepdec)
            if erdata is not None:
                allerror[f,:,:] = shiftim(allerror[f,:,:],pixsepra,pixsepdec)

    if 'median' in method:            
        comb = np.nanmedian(alldata,axis=0)
        ercomb = np.zeros(comb.shape)
        if erstd: ercomb = np.nanmedian(np.abs(comb-alldata),axis=0) #MAD
        if erdata is not None: ercomb = np.sqrt(ercomb**2. + np.nanmedian(allerror,axis=0)**2.0)
    elif method =='mean':
        comb = np.nanmean(alldata,axis=0)
        ercomb = np.zeros(comb.shape)
        if erstd: ercomb = np.nanstd(alldata,axis=0) 
        if erdata is not None: ercomb = np.sqrt(ercomb**2.0 + np.nanmean(allerror,axis=0)**2.0)
    elif method == 'weighted':
        if erdata is None: print("ERROR in combine_images: No erfiles passed!")
        allerror[allerror == 0] = np.nan
        comb = np.nansum(alldata/allerror**2.0,axis=0)/np.nansum(1.0/allerror**2.0,axis=0)
        ercomb = np.nansum(1.0/allerror,axis=0)/np.nansum(1.0/allerror**2.0,axis=0)
        if erstd: ercomb = np.sqrt(np.nanstd(alldata,axis=0)**2.0 + ercomb**2.0)
        
    if (erdata is not None):
        fits.writeto(savefile,(comb,ercomb),header=firsthead,clobber=True)
        return comb,ercomb
    else:
        fits.writeto(savefile,comb,header=firsthead,clobber=True)
        return comb
    
        
## ----------------------------------------------------------------------
## --------------- MASTER BIAS ------------------------------------------
## -----------------------------------------------------------------------
## PURPOSE: calculate master_bias (median) from a given date.
##          Note: For this to work, bias files need to be organized per folder,
##                each corresponding to a night=date.
##          See routine 'prepare_folder'
## INPUT: MJD date at which master_bias is desired, e.g: '57811'
## OPTIONAL INPUT:
##          biasdir: root folder where bias files are found,
##                   def: home/crisp/FORS2-POL/bias
## OUTPUT:
##        mbias1,mbias2: master bias for chip1/chip2

def master_bias(date,biasdir=home+'/crisp/FORS2-POL/bias/'):

    if not os.path.exists(biasdir+date+'/'):
        print("   ERROR IN MASTER_BIAS: directory %s does not exist" %(biasdir+date))
        sys.exit('stop')
        
    if os.path.isfile(biasdir+date+"/master_bias1.fits"):
        mb1 = fits.open(biasdir+date+"/master_bias1.fits")
        mb2 = fits.open(biasdir+date+"/master_bias2.fits")
        return mb1[0].data,mb2[0].data

    #read biasmap
    biasmap = np.loadtxt(biasdir+date+'/biasmap.dat',
                         dtype={'names':('file','type','target','filter',
                                         'angle','mjd','chip','moon'),
                                'formats':('O','O','O','O','f','f','O','O')})
    biasfiles1 = [bmap['file'] for b,bmap in enumerate(biasmap) if (bmap['chip'] == 'CHIP1')]
    biasfiles2 = [bmap['file'] for b,bmap in enumerate(biasmap) if (bmap['chip'] == 'CHIP2')]
    if len(biasfiles1) != len(biasfiles2):
        print("   ERROR IN MASTER_BIAS: wrong files in" %(biasdir+date))
        sys.exit('stop')
    biases1 = np.array([fits.getdata(biasdir+date+'/'+bfile) for bfile in biasfiles1])
    biases2 = np.array([fits.getdata(biasdir+date+'/'+bfile) for bfile in biasfiles2])

    #median
    master_bias1 = np.median(biases1,axis=0)
    master_bias2 = np.median(biases2,axis=0)

    #save
    fits.writeto(biasdir+date+"/master_bias1.fits",master_bias1,clobber=True)
    fits.writeto(biasdir+date+"/master_bias2.fits",master_bias2,clobber=True)
    return master_bias1,master_bias2

## ----------------------------------------------------------------------
## --------------- DATA FLAT ------------------------------------------
## -----------------------------------------------------------------------
## PURPOSE: Calculate data_flat from data itself summing all angles of obeam and ebeam,
##          then divide obeam/ebeam at correct position of each pixel 
## INPUT:
##       - File names of all angles of given filter for CHIP1
##       - File names of all angles of given filter for CHIP2   
## OPTIONAL INPUT:
##       - savefile: base name to save fits and plots
##       - doplot:   True/False(def): Make plots of flat and stats.
##       - dobias:   True/False(def). Do bias correction first. If True pass bias1/bias2
##       - bias1/bias2: master bias for CHIP1/CHIP2
##       - docosmic: True/False(def): Do cosmic ray correction first.
##       - dobin:    True/False(def): Do also binning. This is used as a test for now... When
##                   actual binning is done in main code, it corrects first with normal flat.
##       - center:   Where to start binning (see "bin_image").
##       - binsise:  Size of binning box  (see "bin_image").
##       - binsigma: Sigma for outliers of binning  (see "bin_image").
## OUTPUT:
##        Dictionary with 'flat' map to multiply extraordinary beam
##                   and  'binflat' map (if dobin set)

## possibly: INLA on flat!!

def data_flat(files1,files2,savefile=None,doplot=False,\
	      dobias=False,bias1=None,bias2=None,docosmic=False,dobin=False,
	      center=None,ecenter=None,binsize=None,binsigma=None):	

    ret = {}
    
    ##-- see if file exist
    if ((savefile is not None) & (os.path.isfile(savefile+'-flat.fits')) &
        (not dobin or os.path.isfile(savefile+'-binflat.fits'))):
                                 
        print("   Found existing separated file: %s" %savefile+'-flat.fits')
        flat = fits.open(savefile+'-flat.fits')
        ret['flat'] = flat[0].data
        if doplot: plot_flat(ret['flat'],savefile)
        
        if (dobin) & (savefile is not None) & (os.path.isfile(savefile+'-binflat.fits')):
            binflat = fits.open(savefile+'-binflat.fits')
            ret['binflat'] = binflat[0].data
            if doplot: plot_flat(ret['binflat'],savefile,dobin=True)
        return ret

    print(" Calculating flat from sum of all angles")
    
    #checks
    if dobias and (bias1 is None or bias2 is None):
        print(" Error bias input into data_flat")

    #let's assume pointing is great and no matching between files is needed
	
    #load data
    data1,data2 =[],[]
    for f in range(0,len(files1)):
        h1,d1 = read_fits(files1[f])
        h2,d2 = read_fits(files2[f])
        if dobias: 
            d1,d2 = d1-bias1,d2-bias2
        if docosmic: 
            d1 = cosmic_rays(d1,h1,outfile=savefile+'-chip1')
            d2 = cosmic_rays(d2,h2,outfile=savefile+'-chip2')

        data1.append(d1);data2.append(d2)

    #simple sum
    sum1,sum2 = np.sum(data1,0),np.sum(data2,0)

    #separate beams
    osum1,esum1,osum2,esum2 = separate_beams(sum1,sum2,default=True,
		                             savefile1=savefile+'-sumang',
                                             savefile2=savefile+'-sumang')

    #stick chips
    osum = stick_chips(osum1,osum2,h1,h2,savefile=savefile+'-sumang-obeam')
    esum = stick_chips(esum1,esum2,h1,h2,savefile=savefile+'-sumang-ebeam')

    ##-- Masks 
    omask, emask = (osum > 0), (esum > 0)

    ##-- Correct ebeam/obeam shift
    ydiff = find_shift(osum,esum,savefile=savefile+'-chip12-sumang',default=True)
    esum,emask = ebeam_shift(esum,ydiff,mask=emask,savefile=savefile+'-sumang-ebeam-merged')

    ##-- Divide one by the other
    flat = np.zeros(np.shape(osum),dtype=float)
    flat[omask] = osum[omask]/esum[omask]
    ret['flat'] = flat
            
    ##-- save
    if (savefile is not None):
        fits.writeto(savefile+'-flat.fits',flat,clobber=True)

    ##-- plot statistics
    if doplot:
        plot_flat(flat,savefile)
    
    ##--Binning
    if dobin:
        bin_osum,errbin_osum = bin_image(osum,omask,radpix=binsize,sigmaclip=binsigma,
			                 fullbin=True,center=center,
                                         savefile=savefile+'-sumang-obeam-merged')
        bin_esum,errbin_esum = bin_image(esum,emask,radpix=binsize,sigmaclip=binsigma,
			                 fullbin=True,center=ecenter,
                                         savefile=savefile+'-sumang-ebeam-merged-shifted')
		
        binmask,binemask = (bin_osum > 0),(bin_esum > 0)
        binflat = np.zeros(np.shape(osum),dtype=float)
        binflat[binmask] = bin_osum[binmask]/bin_esum[binmask]
        ret['binflat'] = binflat
        if (savefile is not None):
            fits.writeto(savefile+'-binflat.fits',binflat,clobber=True)
        if doplot:
            plot_flat(binflat,savefile,dobin=True)

    return ret
      ## BEFORE: traditional flat
	#get masks (from separate_beams)
	#mask1,mask2 = np.zeros(np.shape(d1),dtype=bool),np.zeros(np.shape(d2),dtype=bool)
	#xmin1,xmax1,xmin2,xmax2 = 187,1861,188,1864 
	#strips1,strips2 = get_strips(plotfile=savefile)
	#med1,omed1 = np.zeros(len(strips1)/2,dtype=float),np.zeros(len(strips1)/2,dtype=float) 
	#med2,omed2 = np.zeros(len(strips2)/2,dtype=float),np.zeros(len(strips2)/2,dtype=float) 
#
	#for s in range(0,len(strips1)-1):
		#if s%2 == 0:
			#mask1[strips1[s,1]:strips1[s+1,0],xmin1:xmax1+1] = True
			#med1[s/2] = np.median(sum1[strips1[s,1]:strips1[s+1,0],xmin1:xmax1+1])
		#else:
			#mask1[strips1[s,1]:strips1[s+1,0],xmin1:xmax1+1] = True
			#omed1[s/2] = np.median(sum1[strips1[s,1]:strips1[s+1,0],xmin1:xmax1+1])
	#for s in range(0,len(strips2)-1):
		#if s%2 != 0:
			#mask2[strips2[s,1]:strips2[s+1,0],xmin2:xmax2+1] = True
			#med2[s/2] = np.median(sum2[strips2[s,1]:strips2[s+1,0],xmin2:xmax2+1])
		#else:
			#mask2[strips2[s,1]:strips2[s+1,0],xmin2:xmax2+1] = True
			#omed2[s/2] = np.median(sum2[strips2[s,1]:strips2[s+1,0],xmin2:xmax2+1])
#	
	##flat
	#flat1,flat2 = np.zeros(np.shape(d1),dtype=float),np.zeros(np.shape(d2),dtype=float)
#
	##common median
	#summed1,summed2 = np.median(sum1[mask1]),np.median(sum2[mask2])
	#flat1[mask1],flat2[mask2] = summed1/sum1[mask1],summed2/sum2[mask2]

#	#save
#	if (savefile is not None):
#		fits.writeto(savefile+'-chip1-flat.fits',flat1,clobber=True)
#		fits.writeto(savefile+'-chip2-flat.fits',flat2,clobber=True)
#	return flat1,flat2


## ----------------------------------------------------------------------
## --------------- PLOT_FLAT ------------------------------------------
## -----------------------------------------------------------------------
## PURPOSE: Plot flat and flat statistics from "data_flat"          
## INPUT:
##       - flat image
##       - savefile: base name to save plots
## OPTIONAL INPUT:
##       - dobin:    True/False(def): Takes into account that it's a binned image.
## OUTPUT:
##        Plots '...(bin)flat.png' and '..(bin)flatstats.png'

def plot_flat(flat,savefile,dobin=False):

    mask = (np.isfinite(flat) & (flat > 0))
    if dobin: bf='bin'
    else: bf=''
    
    ##-- Image plot
    fs = 16
    fig,ax = plt.subplots(1,figsize=(12,7))
    l1 = np.percentile(flat[mask].flatten(), 5)
    l2 = np.percentile(flat[mask].flatten(), 95)
    c = ax.imshow(flat,clim=(l1,l2))
    ax.set_xlabel("x [pix]",fontsize=fs)
    ax.set_ylabel("y [pix]",fontsize=fs)
    ax.tick_params(labelsize=fs-4)
    cb = fig.colorbar(c,ax=ax)
    cb.set_label('Flat (Obeam/Ebeam)',rotation=270,labelpad=15,fontsize=fs)
    cb.ax.tick_params(labelsize=fs-4) 
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.savefig(savefile+'-'+bf+'flat.png')
    plt.close(fig)
    
    ##-- Plots: flat vs ypix,xpix, hist flat
    ny,nx = np.shape(flat)
    x,y = np.linspace(0,nx-1,nx), np.linspace(0,ny-1,ny)
    xx,yy = np.meshgrid(x,y)
        
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.33
    bottom, height = 0.1, 0.85
    left_plot = [left, bottom, width, height]
    right_plot = [left+width, bottom, width, height]
    #left_histx = [left, bottom+height+0.02, width, 0.2]
    #right_histx = [left+width,bottom+height+0.02,width,0.2]
    histy = [left+2*width+0.02, bottom, 0.2, height]
    
    fig = plt.figure(1, figsize=(16,8))
    axXpixFlat = plt.axes(left_plot)
    axYpixFlat = plt.axes(right_plot)
    #axHistXpix = plt.axes(left_histx)
    #axHistYpix = plt.axes(right_histx)
    axHistFlat = plt.axes(histy)

    # no labels
    #axHistXpix.xaxis.set_major_formatter(nullfmt)
    #axHistYpix.xaxis.set_major_formatter(nullfmt)
    axYpixFlat.yaxis.set_major_formatter(nullfmt)
    axHistFlat.yaxis.set_major_formatter(nullfmt)
    axHistFlat.set_xlabel("N")
    axXpixFlat.set_xlabel("x (pixel)")
    axXpixFlat.set_ylabel("Flat (Obeam/Ebeam)")
    axYpixFlat.set_xlabel("y (pixel)")
    # bins
    xbins,ybins = np.arange(0,nx,30),np.arange(0,ny,30)
    lo = np.percentile(flat[mask].reshape(-1),5)
    up = np.percentile(flat[mask].reshape(-1),95)
    fbins = np.arange(lo,up+0.001,0.001)
    # the plots:
    if dobin:
        x,y,f = xx[mask].reshape(-1),yy[mask].reshape(-1),flat[mask].reshape(-1)
        uniq = (np.unique(f,return_index=True))[1]
        axXpixFlat.plot(x[uniq],f[uniq],'o',alpha=0.5,markersize=4.0)
        axYpixFlat.plot(y[uniq],f[uniq],'o',alpha=0.2,markersize=4.0)
        axHistFlat.hist(f[uniq],orientation='horizontal',bins=fbins,log=True)
    else:
        axXpixFlat.hist2d(xx[mask],flat[mask],bins=[xbins,fbins],cmap='rainbow',\
                          range=[[0,nx-1],[lo,up]])
        axYpixFlat.hist2d(yy[mask],flat[mask],bins=[ybins,fbins],cmap='rainbow',\
                          range=[[0,ny-1],[lo,up]])        
        axHistFlat.hist(flat[mask],orientation='horizontal',bins=fbins,log=True)
    #axXpixFlat.scatter(xx[mask].reshape(-1),flat[mask].reshape(-1))
    #axYpixFlat.scatter(yy[mask].reshape(-1),flat[mask].reshape(-1))
    #axHistXpix.hist(xx[mask].reshape(-1),bins=xbins,log=True)
    #axHistYpix.hist(yy[mask].reshape(-1),bins=ybins,log=True)
        
    axXpixFlat.set_ylim((lo,up))
    axYpixFlat.set_ylim((lo,up))
    axHistFlat.set_ylim((lo,up))
    
    fig.savefig(savefile+'-'+bf+'flatstats.png')
    plt.close(fig)

## ---------------------------------------------------------------
## --------- FIND GAIN ------------------------------------
## ---------------------------------------------------------------
## This function is not finished: under construction

def find_gain(date,mbias1,mbias2,flatdir=home+'/crisp/FORS2-POL/flat/'):

    if not os.path.exists(flatdir+date+'/'):
        print("   ERROR IN FIND_GAIN: directory %s does not exist" %(flatdir+date))
        sys.exit('stop')

    #read flatmap
    flatmap = np.loadtxt(flatdir+date+'/flatmap.dat',
                         dtype={'names':('file','galaxy','target','filter',
                                         'angle','mjd','chip','moon'),
                                'formats':('O','O','O','O','f','f','O','O')})
    filters = np.unique(flatmap['filter'])

    #loop filters
    for fi in range(0,len(filters)):
    
        flatfiles = flatmap['file'][flatmap['filter'] == filters[fi]]# os.listdir(flatdir+date+'/')
        flatfiles1 = [ffile for f,ffile in enumerate(flatfiles) if (f%2 == 0)]
        flatfiles2 = [ffile for f,ffile in enumerate(flatfiles) if (f%2 != 0)]
        if len(flatfiles1) != len(flatfiles2):
            print("   ERROR IN FIND_GAIN: wrong files in" %(flatdir+date))
            sys.exit('stop')
        flats1 = np.array([fits.getdata(flatdir+date+'/'+ffile) for ffile in flatfiles1])
        flats2 = np.array([fits.getdata(flatdir+date+'/'+ffile) for ffile in flatfiles2])
        nflats = len(flatfiles1)
    
        #calculate val
        meanval1 = np.mean(flats1-mbias1,axis=0)
        meanval2 = np.mean(flats2-mbias2,axis=0)
        variance1 = np.var(flats1[1:nflats,:,:]-flats1[0:nflats-1,:,:],axis=0)/2.0
        variance2 = np.var(flats2[1:nflats,:,:]-flats2[0:nflats-1,:,:],axis=0)/2.0

        #sigma clip
        #sig=3.0
        #m1,v1 = sigma_clip(meanval1,sigma=sig),sigma_clip(variance1,sigma=sig)
        #ind1 = ((meanval1 > m1.mean()-sig*m1.std()) & (meanval1 < m1.mean()+sig*m1.std()) &
        #        (variance1 > v1.mean()-sig*v1.std()) & (variance1 < v1.mean()+sig*v1.std()))
        #m2,v2 = sigma_clip(meanval2,sigma=sig),sigma_clip(variance2,sigma=sig)
        #ind2 = ((meanval2 > m2.mean()-sig*m2.std()) & (meanval2 < m2.mean()+sig*m2.std()) &
        #        (variance2 > v2.mean()-sig*v2.std()) & (variance2 < v2.mean()+sig*v2.std()))

        #mask
        xi2,xf2,yi2,yf2  = 186,1861,318,1029
        xi1,xf1,yi1,yf1  = 186,1861,6,963
        
        #plot 
        fig,ax = plt.subplots(2)
        #ax[0].plot(meanval1[ind1].reshape(-1),variance1[ind1].reshape(-1),'.')
        ax[0].plot(meanval1[yi1:yf1,xi1:xf1].reshape(-1),variance1[yi1:yf1,xi1:xf1].reshape(-1),'.')
        ax[0].set_title('CHIP 1')
        ax[0].set_ylabel('Variance Counts')
        ax[0].set_xlabel('Pixel Count')
        ax[1].plot(meanval2[yi2:yf2,xi2:xf2].reshape(-1),variance2[yi2:yf2,xi2:xf2].reshape(-1),'.')
        ax[1].set_title('CHIP 2')
        ax[1].set_ylabel('Variance Counts')
        ax[1].set_xlabel('Pixel Count')  
        plt.savefig(flatdir+date+'/'+filters[fi]+'-gain.png')
    plt.close('all')
        
## -----------------------------------------------------------------------------------
## --------- INVESTIGATE BEAM SEPARATION ---------------------------------------------
## ------------------------------------------------------------------------------------
## PURPOSE: Function to obtain the optimal strip separation in a given image
##          This is done by finding abrupt changes in the derivative; it does not work
##          perfectly, so there is an interactive component to this.
## OPTIONAL INPUT:
##                image: Image to get strips from. If this is not set,
##                       strip offsets returned are default saved ones
##                plotfile: Path+basic_file name where to plot result
## OUTPUT: Returns the offsets measured.
##         If image is not provided then the strip offsets returned are default saved ones.
##         Format:
##         It also creates plots: '..strips.png' and writes found ones in '..strips.dat'


## --- These are auxiliary functions
def firstit(vec):
    for i in range(0,np.size(vec)):
        if (i == 0):
            uniqvec = np.asarray(vec[i])
            continue
        if any((uniqvec > vec[i]-5) & (uniqvec < vec[i]+5)):
            continue
        else:
            uniqvec = np.append(uniqvec,vec[i]) 
    return uniqvec

def lastit(vec):
    for i in range(np.size(vec)-1,-1,-1):
        if (i == np.size(vec)-1):
            uniqvec = np.asarray(vec[i])
            continue
        if any((uniqvec > vec[i]-5) & (uniqvec < vec[i]+5)):
            continue
        else:
            uniqvec = np.append(uniqvec,vec[i]) 
    return uniqvec[::-1]


def get_strips(image=None,plotfile=None,dycut=1000.0):

    #old:
    #ostrips1 = [(6,29),(124,209),(305,391),(488,573),(669,754),(851,938)]
    #estrips1 = [(31,115),(211,296),(393,477),(574,657),(755,839)]
    #ostrips2 = [(438,526),(621,706),(802,890),(983,1028)]
    #estrips2 = [(346,433),(528,613),(710,797),(891,976)]

    #DEFAULT
    if image is None:
        usefilts = ['b_HIGH','v_HIGH','R_SPECIAL','I_BESS','H_Alpha','OII_8000']
        n1,n2 = 12,9
        stripfile = home+"/crisp/FORS2-POL/Information/default_strips.dat"
        allstrips = np.loadtxt(stripfile)
        tfilt = plotfile.split('/')[-1].split('-')[0]
        i = np.argwhere(np.array(usefilts) == tfilt).reshape(-1)[0]
        strips1 = np.zeros((n1,2),dtype=int)
        strips2 = np.zeros((n2,2),dtype=int)
        strips1[:,0] = allstrips[0:n1,i*2+0]
        strips1[:,1] = allstrips[0:n1,i*2+1]
        strips2[:,0] = allstrips[n1:n1+n2,i*2+0]
        strips2[:,1] = allstrips[n1:n1+n2,i*2+1]
        #strips1[:,0] = allstrips[0:n1,0]-0.5*allstrips[0:n1,i+1]
        #strips1[:,1] = allstrips[0:n1,0]+0.5*allstrips[0:n1,i+1]
        #strips2[:,0] = allstrips[n1:n1+n2,0]-0.5*allstrips[n1:n1+n2,i+1]
        #strips2[:,1] = allstrips[n1:n1+n2,0]+0.5*allstrips[n1:n1+n2,i+1]
        return strips1,strips2
    
    ny,nx = np.shape(image)
    ymed = np.median(image,axis=1)
    dy = ymed[1:ny]-ymed[0:ny-1]
    ini = np.argwhere(dy < -dycut)
    fin = np.argwhere(dy > dycut)
    if (len(ini) < 3) and (len(fin) < 3):
        ini= np.argwhere(dy<=-dycut)[0:2]
        fin= np.argwhere(dy>=dycut)[0:2]

    #plot
    fig,ax = plt.subplots(1,figsize=(10,10))
    ax.plot(np.arange(0,ny-1),dy)
    ax.set_xlabel('y [pix]')
    ax.set_ylabel('dcounts/dy')
    fig.show()
    if plotfile is not None:
        plt.savefig(plotfile+'-dystrips.png')
    
    #uniq-firt ini,fin
    uniqini = firstit(ini)
    uniqini = uniqini[(uniqini > 0) & (uniqini < ny-1)] 
    uniqfin = lastit(fin)
    uniqfin = uniqfin[(uniqfin > 0) & (uniqfin < ny-1)] 
    #iterate offsets
    if np.size(uniqini) > np.size(uniqfin):
        itones = uniqini
        otones = uniqfin
    else:
        itones = uniqfin
        otones = uniqini

    offsets = np.zeros((np.size(itones),2),dtype=int)
    for i,it in enumerate(itones):
        ins = np.argwhere((otones > it-16) & (otones < it+16))
        if (sum(ins) >= 1) and (np.size(ins) == 1):
            offsets[i,:] = np.sort([it,otones[ins]])
        elif (sum(ins) >= 1) and (np.size(ins) > 1):
            minot = np.argmin(np.abs(it-otones[ins]))
            offsets[i,:] = np.sort([it,otones[ins[minot]]])
        else:
            offsets[i,:] = [it,it]
    for o,ot in enumerate(otones):
        if ot not in offsets:
            offsets = np.insert(offsets,np.size(offsets[:,0]),[ot,ot],axis=0)
    offsets = offsets[np.argsort(offsets[:,0]),:]


    #divide in e/o beam
    #ostrips = np.asarray([off  for i,off in enumerate(offsets) if (i%2 == 0)])
    #estrips = np.asarray([off  for i,off in enumerate(offsets) if (i%2 != 0)])

    ##plot
    img = image
    #img = np.arcsinh(img) # if not flat
    #offsets = offsets[1:len(offsets)] ##machete
    fig,ax = plt.subplots(1,figsize=(10,10))
    lo = np.percentile(img[np.isfinite(img)].flatten(), 5)
    up = np.percentile(img[np.isfinite(img)].flatten(), 99.5)
    ax.imshow(img,clim=(lo,up))
    for i,off in enumerate(offsets):
        ax.plot([0,nx],[off[0],off[0]],'b-')
        ax.plot([0,nx],[off[1],off[1]],'g-')
        ax.text(12,off[1]+5,str(i),color='green')
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    ax.invert_yaxis()
    fig.show()
    #user line input
    print("      Number of lines found: %i" %len(offsets))
    elim = raw_input("      Which lines to eliminate separated by comma ('n' for none)? ")
    if elim.lower() != 'n':
        nelim = np.asarray(elim.split(','),dtype=int)
        if len(nelim) == len(offsets[:,0]): del(offsets)
        else: offsets = np.vstack(off for i,off in enumerate(offsets) if i not in nelim)
    add = raw_input("      Which lines (y-pix) approx. to add separated by comma ('n' for none)? ")
    if add.lower() != 'n':
        nadd = np.asarray(add.split(','),dtype=int)
        mid = np.argwhere((dy == 1) | (dy == -1))
        for a in nadd:
            midmin = np.argmin(np.abs(a - mid))
            try: offsets = np.insert(offsets,np.size(offsets[:,0]),[mid[midmin]],axis=0)
            except: offsets = np.reshape([mid[midmin],mid[midmin]],(1,2))
        offsets = offsets[np.argsort(offsets[:,0]),:]
    #replot strips in yellow
    if (elim.lower() != 'n') | (add.lower() != 'n'):
        for i,off in enumerate(offsets):
            ax.plot([0,nx],[off[0],off[0]],'y-')
            ax.plot([0,nx],[off[1],off[1]],'y-')
        fig.canvas.draw()
        fig.canvas.flush_events()
        
    #save
    if plotfile is not None:
        plt.savefig(plotfile+'-strips.png')
        np.savetxt(plotfile+'-strips.dat',(offsets),fmt='%20s')
        #ascii.write(offsets,plotfile+'-strips.dat',overwrite=True)

    plt.close('all')
    return offsets#ostrips,estrips




## -------------------------------------------------------------------------
## ---------- FUNCTION TO MATCH A LIST OF COORDINATES IN DIFFERENT ANGLES---
## -------------------------------------------------------------------------
## PURPOSE: Match star coordinates in ordinary and extraordinary photometry of several HWP angles
##          It uses the results of iterative PSF photometry from photutils.
## INPUT:
##       1. List of table of ordinary x/y-coordinates and phot (with all angles)
##       2. List of table of extraordinary x/y-coordinates and phot (with all angles)
## OPTIONAL INPUT:
##       - dpix: maximum distance (in pix) to consider match (def:1.0) 
##       - posfree: (def: True) If false, no matching is done
##       - savefile: for plotting
## OUTPUT: final x coordinates
##         final y coordinates
##         final ordinary photometry
##         final extraordinary photometry
##         final error on ord/ext phot
## EFFECTS:   It creates several plots: 'psfflux_compang' and 'psfflux_compbeam'
## DEPENDENCIES: ind_close fct

def ind_close(xall,x,yall,y,dpix):

    #distance
    rall = np.sqrt(xall**2+yall**2)
    r = np.sqrt(x**2+y**2)

    #only nearby ones
    ind = (np.argwhere((np.abs(rall-r) < dpix) & \
                       (np.abs(xall-x) < dpix) & (np.abs(yall-y) < dpix)))[:,0]

    if len(ind) <= 0:
        return -1
    
    #sort
    sort = np.argsort(np.abs(rall[ind]-r))
    return ind[sort[0]]
    

def match_sources(oresarr,eresarr,savefile=None,posfree=False,dpix=1.0,signois=-1,aper=False):

    nangles = len(oresarr)  
    angles = map(str,np.arange(0,nangles*22.5,22.5))

    fl='ap' if aper else 'psf'

    if not posfree:
        pos = ''
        nstars = len(oresarr[0])
        flux,errflux = np.zeros((nangles,nstars,1),dtype=float),np.zeros((nangles,nstars,1),dtype=float)
        eflux,erreflux = np.zeros((nangles,nstars,1),dtype=float),np.zeros((nangles,nstars,1),dtype=float)
        x,y =  oresarr[0]['xfit'].reshape(nstars,1),oresarr[0]['yfit'].reshape(nstars,1)
        for a in range(0,nangles):
            flux[a,:,0],errflux[a,:,0] = oresarr[a][fl+'flux'],oresarr[a][fl+'error']
            eflux[a,:,0],erreflux[a,:,0] = eresarr[a][fl+'flux'],eresarr[a][fl+'error']
        ascii.write([x.reshape(-1),y.reshape(-1)],savefile+pos+'-'+'flux.dat',
                    names=['xfit','yfit'],#,fl+'flux',fl+'error','e'+fl+'flux','e'+fl+'error'],
                    overwrite=True)
        return x,y,flux,eflux,errflux,erreflux

    
    from astropy.table import Table
    pos='-posfree'

    if (savefile is not None) & (os.path.isfile(savefile+pos+'-'+'flux.dat')):
        print("   Found matched "+fl.upper()+" sources of all angles and obeam/ebeam")
        ores1 = ascii.read(savefile+pos+'-'+'flux.dat')
        indfin = np.arange(0,len(ores1),1,dtype=int)

    else:    
        print("   Matching "+fl.upper()+" sources of all angles and obeam/ebeam")

        ## add keys to first res
        ores1,eres1 = Table(oresarr[0]),Table(eresarr[0])    
        for a in range(0,nangles):
            ores1[fl+'flux-'+angles[a]] = np.nan
            ores1[fl+'error-'+angles[a]] = np.nan
            ores1['e'+fl+'flux-'+angles[a]] = np.nan
            ores1['e'+fl+'error-'+angles[a]] = np.nan
            
        ##  match all o/e and all angles
        indfin = []
        for i in range(0,len(ores1)):

            ores1[i][fl+'flux-'+angles[0]] = ores1[i][fl+'flux']
            ores1[i][fl+'error-'+angles[0]] = ores1[i][fl+'error']
    
            ## with ebeam
            ieind = ind_close(eres1['xfit'],ores1[i]['xfit'],
                              eres1['yfit'],ores1[i]['yfit'],dpix)
            if ieind == -1:
                continue
    
            ores1[i]['e'+fl+'flux-'+angles[0]] = eres1[ieind][fl+'flux']
            ores1[i]['e'+fl+'error-'+angles[0]] = eres1[ieind][fl+'error']
    
            ## with other angles
            indarr = np.zeros(nangles,dtype=bool)
            eindarr = np.zeros(nangles,dtype=bool)
            for a in range(1,nangles):
                #obeam
                ores = oresarr[a]
                ind = ind_close(ores['xfit'],ores1[i]['xfit'],ores['yfit'],ores1[i]['yfit'],dpix)
                if ind != -1:
                    ores1[i][fl+'flux-'+angles[a]] = ores[ind][fl+'flux']
                    ores1[i][fl+'error-'+angles[a]] = ores[ind][fl+'error']
                    indarr[a] = True
                #ebeam
                eres = eresarr[a]
                eind = ind_close(eres['xfit'],ores1[i]['xfit'],eres['yfit'],ores1[i]['yfit'],dpix)
                if eind != -1:
                    ores1[i]['e'+fl+'flux-'+angles[a]] = eres[eind][fl+'flux']
                    ores1[i]['e'+fl+'error-'+angles[a]] = eres[eind][fl+'error']
                    eindarr[a] = True
                #print(np.sum(ieind),np.sum(ind),np.sum(eind))
            
            if (np.sum(indarr) == nangles-1) & (np.sum(eindarr) == nangles-1) & (ieind != 1):
                indfin.append(i)

        ascii.write(ores1[indfin],savefile+pos+'-'+'flux.dat',overwrite=True)

    ## signal-to-noise: only for starfinder IDL calclulation
    if signois > 0:
        indsn = np.argwhere(ores1[indfin][fl+'flux-0.0']/ores1[indfin][fl+'error-0.0'] >= signois).reshape(-1)
        
    ## plot o/e comparison
    if savefile is not None:
        fig,ax = plt.subplots(nangles,figsize=(10,12))
        for a in range(0,nangles):
            ax[a].plot([0, 1], [0, 1], transform=ax[a].transAxes)
            ax[a].scatter(ores1[indfin][fl+'flux-'+angles[a]],ores1[indfin]['e'+fl+'flux-'+angles[a]])
            if signois > 0:
                ax[a].scatter(ores1[indfin[indsn]][fl+'flux-'+angles[a]],
                              ores1[indfin[indsn]]['e'+fl+'flux-'+angles[a]],c='orange',
                              label='S/N > '+str(signois))
                if a ==0: ax[a].legend()
            ax[a].set_yscale('log')
            ax[a].set_xscale('log')
            ax[a].set_xlim([1e3,1e6])
            ax[a].set_ylim([1e3,1e6])
            ax[a].set_xlabel('Obeam '+angles[a])
            ax[a].set_ylabel('Ebeam '+angles[a])
        plt.savefig(savefile+pos+'-flux_compbeam.png')

        #diff = (ores1[indfin]['psfflux-'+angles[a]]-ores1[indfin]['epsfflux-'+angles[a]])/
        

        ## plot angle comparison
        import itertools as it
        combs = list(it.combinations(np.arange(0,nangles),2))
        ncombs = len(combs)
        fig,ax = plt.subplots(2,ncombs,figsize=(15,10))  
        for c in range(0,ncombs):
            comb = combs[c]
            ax[0,c].scatter(ores1[indfin][fl+'flux-'+angles[comb[0]]],
                            ores1[indfin][fl+'flux-'+angles[comb[1]]])
            if signois > 0:
                ax[0,c].scatter(ores1[indfin[indsn]][fl+'flux-'+angles[comb[0]]],
                              ores1[indfin[indsn]][fl+'flux-'+angles[comb[1]]],c='orange',
                              label='S/N > '+str(signois))
                if c ==0: ax[0,c].legend()
            ax[0,c].set_yscale('log')
            ax[0,c].set_xscale('log')
            ax[0,c].set_xlim([1e3,1e6])
            ax[0,c].set_ylim([1e3,1e6])
            ax[0,c].plot([0, 1], [0, 1], transform=ax[0,c].transAxes)
            ax[0,c].set_xlabel('Obeam '+angles[comb[0]])
            ax[0,c].set_ylabel('Obeam '+angles[comb[1]])

            ax[1,c].scatter(ores1[indfin]['e'+fl+'flux-'+angles[comb[0]]],
                    ores1[indfin]['e'+fl+'flux-'+angles[comb[1]]])
            if signois > 0:
                ax[1,c].scatter(ores1[indfin[indsn]]['e'+fl+'flux-'+angles[comb[0]]],
                                ores1[indfin[indsn]]['e'+fl+'flux-'+angles[comb[1]]],c='orange',
                                label='S/N > '+str(signois))
            ax[1,c].set_yscale('log')
            ax[1,c].set_xscale('log')
            ax[1,c].set_xlim([1e3,1e6])
            ax[1,c].set_ylim([1e3,1e6])
            ax[1,c].plot([0, 1], [0, 1], transform=ax[1,c].transAxes)
            ax[1,c].set_xlabel('Ebeam '+angles[comb[0]])
            ax[1,c].set_ylabel('Ebeam '+angles[comb[1]])
        plt.savefig(savefile+pos+'-flux_compang.png')
        plt.close('all')

    ##pick only brightest??       
    if signois > 0:
        indfin = indfin[indsn]
    
    #final
    nstars = len(indfin)
    print("      Found %i matching  sources" %nstars)
    flux,errflux = np.zeros((nangles,nstars,1),dtype=float),np.zeros((nangles,nstars,1),dtype=float)
    eflux,erreflux = np.zeros((nangles,nstars,1),dtype=float),np.zeros((nangles,nstars,1),dtype=float)
    for a in range(0,nangles):
        flux[a,:,0] = ores1[indfin][fl+'flux-'+angles[a]]
        errflux[a,:,0] = ores1[indfin][fl+'error-'+angles[a]]
        eflux[a,:,0] = ores1[indfin]['e'+fl+'flux-'+angles[a]]
        erreflux[a,:,0] = ores1[indfin]['e'+fl+'error-'+angles[a]]
    x = np.asarray(ores1[indfin]['xfit']).reshape(nstars,1)
    y = np.asarray(ores1[indfin]['yfit']).reshape(nstars,1) 

    return x,y,flux,eflux,errflux,erreflux


## -------------------------------------------------------------------------
## ---------- FUNCTION TO FIND SHIFT OF AN ORDINARY/EXTRAORD IMAGES--------
## -------------------------------------------------------------------------
## PURPOSE: Find stars in ordinary and extraordinary images and then match them to obtain
##          the offsets between the two. It uses DAOPHOT from photutils (see find_stars).
##          The optimum offset seems to be a quadratic relation, so such a fit is performed.
## INPUT:
##       1. Ordinary beam image
##       2. Extraordinary beam image
## OPTIONAL INPUT:
##       - fwhm: Full width half max to search for stars (def: 5)
##       - threshold: The absolute image value above sky to which select sources (def: 5)
##       - savefile: Path+basic_file name to save results
##       - default: Boolean to get default offsets between ord/extraord instead of calculating
##                  def: True
## OUTPUT: y-difference (as a function of y pix) between ordinary/extraordinary images
##         It creates several plots: '..xydiff.png','..xyhist.png','fluxdiff.png','.fluxcomp.png'
##         It also makes quadratic fit and saves as '..quadpars.dat'
## DEPENDENCIES: find_stars function (photutils),astropy,scipy (python)

def find_shift(image1,image2,fwhm=5.0,threshold=5.0,savefile=None,default=True):


    #quadfunc = lambda p,x: p[1]+p[2]*(x-p[0])+p[3]*(x-p[0])**2.0#p[1]+0*(x-450)+p[3]*(x-450)**2.0
    quadfunc = lambda p,x: p[1]+0*(x-450)+p[3]*(x-450)**2.0
    cubefunc = lambda p,x: p[1]+p[2]*(x-p[0])+p[3]*(x-p[0])**2.0+p[4]*(x-p[0])**3.0
    errfunc = lambda p,x,y: quadfunc(p,x)-y

    ##load default
    if default:
        tfile = home+"/crisp/FORS2-POL/Information/default_quadpars.dat"
        allpars = np.loadtxt(tfile,usecols=(1,2,3,4))  
        usefilts = np.loadtxt(tfile,usecols=0,dtype=object)#['b_HIGH','v_HIGH','R_SPECIAL','I_BESS','H_Alpha'] 
        tfilt = savefile.split('/')[-1].split('-')[0]
        i = np.argwhere(np.array(usefilts) == tfilt).reshape(-1)[0]
        ny,nx = np.shape(image2)
        y = np.linspace(0,ny-1,ny)
        ydiff = np.round(quadfunc(allpars[i,:],y),1)
        return ydiff
    
    ##load if it exists
    if (savefile is not None):
        tfile = savefile+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-quadpars.dat'
        if os.path.isfile(tfile):
            parinfo  = ascii.read(tfile)
            pars = np.asarray([p for p in parinfo[0]])
            ny,nx = np.shape(image2)
            y = np.linspace(0,ny-1,ny)
            ydiff = np.round(quadfunc(pars,y),1)
            return ydiff


    from scipy import optimize
    from astropy.stats import sigma_clipped_stats
    
    print("   Matching stars")
    ny1,nx1 = np.shape(image1)
    sources1 = find_stars(image1,fwhm=fwhm,threshold=threshold)
    sources2 = find_stars(image2,fwhm=fwhm,threshold=threshold)

    ##match x coordinates
    match2 = [np.argmin(np.abs(x-sources2["xcentroid"])) for x in sources1["xcentroid"]]
    diffx = sources1["xcentroid"] - sources2["xcentroid"][match2]
    diffy = sources1["ycentroid"] - sources2["ycentroid"][match2]
    diffflux = (sources1["flux"] - sources2["flux"][match2])/sources1["flux"]
    diffmag = (sources1["mag"] - sources2["mag"][match2])/sources1["mag"]


    ## Error on positions from errimg
    #errsources1 = find_stars(img+errimg,fwhm=fwhm,threshold=threshold)
    #errsources2 = find_stars(img-errimg,fwhm=fwhm,threshold=threshold)
    #match1 = [np.argmin(np.abs(x-errsources1["xcentroid"])) for x in sources["xcentroid"]]
    #match2 = [np.argmin(np.abs(x-errsources2["xcentroid"])) for x in sources["xcentroid"]]
    #errx1 = sources["xcentroid"] - errsources1["xcentroid"][match1]
    #errx2 = sources["xcentroid"] - errsources2["xcentroid"][match2]
    #erry1 = sources["ycentroid"] - errsources1["ycentroid"][match1]
    #erry2 = sources["ycentroid"] - errsources2["ycentroid"][match2]
    #sources['errx'] = 0.5*(errx1+errx2)
    #sources['erry'] = 0.5*(erry1+erry2)
    #pdb.set_trace()
         
    ##median
    win = np.argwhere((sources1["xcentroid"] > 10) & (sources1["xcentroid"] < nx1-10) &
                      (np.abs(diffx) < 10) & (np.abs(diffy) < 150)).reshape(-1)
    xmean,xmedian,xstd = sigma_clipped_stats(diffx[win],sigma=3.0,iters=5)
    ymean,ymedian,ystd = sigma_clipped_stats(diffy[win],sigma=3.0,iters=5)
    gmatch = np.argwhere((np.abs(diffx[win]-xmedian) < 3.0*xstd) &
                         (np.abs(diffy[win]-ymedian) < 3.0*ystd)).reshape(-1)
    gmatch = win[gmatch]
    meandiffx = np.median(diffx[gmatch])
    meandiffy = np.median(diffy[gmatch])
    meandiffflux = np.median(diffflux[gmatch])
    meandiffmag = np.median(diffmag[gmatch])
    print("     -> Median x-difference: %f" %meandiffx)
    print("     -> Median y-difference: %f" %meandiffy) 

    ##fits: diffy vs y for now
    p0 = [450,88,0.0,1.0]#,1.0]
    fity = np.array(sources1["ycentroid"][gmatch],dtype='float').flatten(-1)
    fitdiffy = np.array(diffy[gmatch],dtype='float').flatten(-1)
    pars,succ = optimize.leastsq(errfunc,p0[:],args=(fity,fitdiffy))
    #pars,parscov = optimize.curve_fit(quadfunc,fity,fitdiffy,p0=p0)
    #parserr = np.sqrt(np.diag(parscov))
    yarr = np.linspace(np.min(fity),np.max(fity),np.max(fity)-np.min(fity),dtype='int')
    dyarr = quadfunc(pars,yarr)
    print('     -> Quadpars: %f %f %f %f' %(pars[0],pars[1],pars[2],pars[3]))
    
    ##plot histos
    fig,ax = plt.subplots(2,figsize=(8,6))
    ax[0].hist(diffx[gmatch])
    ax[0].set_xlabel("x-difference (pix)")
    ax[0].set_xlim([np.min(diffx[gmatch]),np.max(diffx[gmatch])])
    ax[0].axvline(meandiffx,color='k',linestyle='dashed')
    ax[0].text(0.9,0.8,'N='+str(np.size(gmatch)),transform=ax[0].transAxes,
               horizontalalignment='center',verticalalignment='center')
    ax[1].hist(diffy[gmatch])
    ax[1].set_xlabel("y-difference (pix)")
    ax[1].set_xlim([np.min(diffy[gmatch]),np.max(diffy[gmatch])])
    ax[1].axvline(meandiffy,color='k',linestyle='dashed')
    #fig.show()
    if savefile is not None:            
        plt.savefig(savefile+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-xyhist.png')     

    ##plot graphs
    fig,ax = plt.subplots(2,2,figsize=(9,6),sharex='col',sharey='row')
    fig.subplots_adjust(wspace=0,hspace=0)
    ax[0,0].plot(sources1["xcentroid"][gmatch],diffx[gmatch],'o')
    ax[0,0].set_ylabel("x-diff (pix)")
    ax[0,0].text(0.1,0.8,'N='+str(np.size(gmatch)),transform=ax[0,0].transAxes,
               horizontalalignment='center',verticalalignment='center')
    ax[1,0].plot(sources1["xcentroid"][gmatch],diffy[gmatch],'o')
    ax[1,0].set_xlabel("x-obeam (pix)")
    ax[1,0].set_ylabel("y-diff (pix)")
    ax[0,1].plot(sources1["ycentroid"][gmatch],diffx[gmatch],'o')
    ax[1,1].plot(sources1["ycentroid"][gmatch],diffy[gmatch],'o')
    ax[1,1].set_xlabel("y-obeam (pix)")
    ax[1,1].grid(linestyle='--')
    ax[1,1].plot(yarr,dyarr,'--')
    #fig.show()
    if savefile is not None:            
        plt.savefig(savefile+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-xydiff.png')

    fig,ax = plt.subplots(1,2,figsize=(9,6),sharex='col',sharey='row')
    fig.subplots_adjust(wspace=0,hspace=0)
    ax[0,0].plot(sources1["xcentroid"][gmatch],diffx[gmatch],'o')
    ax[0,0].set_ylabel("x-diff (pix)")
    ax[0,0].text(0.1,0.8,'N='+str(np.size(gmatch)),transform=ax[0,0].transAxes,
               horizontalalignment='center',verticalalignment='center')
    ax[1,0].plot(sources1["xcentroid"][gmatch],diffy[gmatch],'o')
    ax[1,0].set_xlabel("x-obeam (pix)")
    ax[1,0].set_ylabel("y-diff (pix)")
    ax[0,1].plot(sources1["ycentroid"][gmatch],diffx[gmatch],'o')
    ax[1,1].plot(sources1["ycentroid"][gmatch],diffy[gmatch],'o')
    ax[1,1].set_xlabel("y-obeam (pix)")
    ax[1,1].grid(linestyle='--')
    ax[1,1].plot(yarr,dyarr,'--')
    #fig.show()
    if savefile is not None:            
        plt.savefig(savefile+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-xydiff2.png')     

    fig,ax = plt.subplots(2,2,figsize=(9,6),sharey='row',sharex='col')
    fig.subplots_adjust(wspace=0,hspace=0)
    ax[0,0].plot(sources1["xcentroid"][gmatch],diffflux[gmatch],'o')
    ax[0,0].axhline(meandiffflux,color='b',linestyle='dashed')
    ax[0,0].set_ylabel("flux diff %")
    ax[0,0].text(0.2,0.9,'Med='+"% 6.4f" %(meandiffflux),transform=ax[0,0].transAxes,
                 horizontalalignment='center',verticalalignment='center')
    ax[1,0].plot(sources1["xcentroid"][gmatch],diffmag[gmatch],'o')
    ax[1,0].axhline(meandiffmag,color='b',linestyle='dashed')
    ax[1,0].set_xlabel("x-obeam (pix)")
    ax[1,0].set_ylabel("mag diff %")
    ax[1,0].text(0.2,0.9,'Med='+"% 6.4f" %(meandiffmag),transform=ax[1,0].transAxes,
                 horizontalalignment='center',verticalalignment='center')
    ax[0,1].plot(sources1["ycentroid"][gmatch],diffflux[gmatch],'o')
    ax[0,1].axhline(meandiffflux,color='b',linestyle='dashed')
    ax[1,1].plot(sources1["ycentroid"][gmatch],diffmag[gmatch],'o')
    ax[1,1].axhline(meandiffmag,color='b',linestyle='dashed')
    ax[1,1].set_xlabel("y-obeam (pix)")
    if savefile is not None:            
        plt.savefig(savefile+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-fluxdiff.png') 

    fig,ax = plt.subplots(1,2,figsize=(9,6))
    ax[0].plot(sources1["flux"][gmatch],sources2["flux"][match2][gmatch],'o')
    ll = [np.min(sources1["flux"][gmatch]),np.max(sources1["flux"][gmatch])]
    ax[0].plot(ll,ll,'b--',linewidth=0.5)
    ax[0].set_xlabel("flux1")
    ax[0].set_ylabel("flux2")
    ax[1].plot(sources1["mag"][gmatch],sources2["mag"][match2][gmatch],'o')
    ll = [np.min(sources1["mag"][gmatch]),np.max(sources1["mag"][gmatch])]
    ax[1].plot(ll,ll,'b--',linewidth=0.5)
    ax[1].set_xlabel("mag1")
    ax[1].set_ylabel("mag2")
    if savefile is not None:            
        plt.savefig(savefile+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-fluxcomp.png')     
        
    ##plot images with circles
    fig = plt.figure(figsize=(12,7))
    fig.add_subplot(1,2,1)
    img1,img2 = np.arcsinh(image1),np.arcsinh(image2)
    #wfin1, wfin2 = np.isfinite(img1),np.isfinite(img2)
    #l1, l2 = (np.percentile(img1[wfin1].flatten(), 10), np.percentile(img1[wfin1].flatten(), 90))
    #interval = MinMaxInterval()
    #vmin, vmax = interval.get_limits(img1)
    #norm1 = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())

    #norm1 = ImageNormalize(img1, interval=MinMaxInterval(),
    #                      stretch=SqrtStretch())
    l1 = np.percentile(img1[np.isfinite(img1)].flatten(), 5)
    l2 = np.percentile(img1[np.isfinite(img1)].flatten(), 99.5)
    plt.imshow(img1,clim=(l1,l2))#norm=norm1,origin='lower')
    plt.plot(sources1["xcentroid"][gmatch],sources1["ycentroid"][gmatch],'o',mfc='none')
    plt.xlabel("x-obeam (pix)")
    plt.ylabel("y-obeam (pix)")
    plt.ylim(plt.ylim()[::-1])
    fig.add_subplot(1,2,2)
    l1 = np.percentile(img2[np.isfinite(img2)].flatten(), 5)
    l2 = np.percentile(img2[np.isfinite(img2)].flatten(), 99.5)
    plt.imshow(img2,clim=(l1,l2))
    plt.plot(sources2["xcentroid"][match2][gmatch],sources2["ycentroid"][match2][gmatch],'o',mfc='none')
    plt.xlabel("x-ebeam (pix)")
    plt.ylabel("y-ebeam (pix)")
    plt.ylim(plt.ylim()[::-1])
    if savefile is not None:            
        plt.savefig(savefile+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-image.png') 
    plt.close('all')
        
    ##print in file
    if savefile is not None:
        ascii.write(sources1,savefile+'-obeam-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-xycoord.dat',overwrite=True)
        ascii.write(sources2,savefile+'-ebeam-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-xycoord.dat',overwrite=True)
        ascii.write(pars,savefile+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-quadpars.dat',
                    overwrite=True)

    ##final ydiff shift
    ny,nx = np.shape(image2)
    y = np.linspace(0,ny-1,ny)
    ydiff = np.round(quadfunc(pars,y),1)

    return ydiff

## -------------------------------------------------------------------------
## ---------- FUNCTION TO FIND STARS OF AN IMAGE----------------------------
## -------------------------------------------------------------------------
## PURPOSE: Find stars in images. It uses DAOPHOT from photutils.
## INPUT:
##       image
## OPTIONAL INPUT:
##       - fwhm: Full width half max to search for stars (def: 5)
##       - threshold: The absolute image value above which to select sources (def: 5)
## OUTPUT: sources dictionary with 'flux','mag'... see daophotfinder from photutils
## DEPENDENCIES: photutils,astropy,scipy (python)

def find_stars(image,fwhm=5.0,threshold=5.0):

    from astropy.stats import sigma_clipped_stats
    #from astropy.visualization import (MinMaxInterval, SqrtStretch,
    #                                   ImageNormalize)
    from photutils import DAOStarFinder

    ny,nx = np.shape(image)
    
    ##get background
    mean, median, std = sigma_clipped_stats(image,sigma=3.0,iters=5)
 
    ##find stars
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)    
    sources = daofind(image - median)

    return sources
    
## --------------------------------------------------------------------------------
## ---------- FUNCTION TO CALCULATE PSF PHOT--------------------------------------
## --------------------------------------------------------------------------------
## PURPOSE: Calculate PSF photometry of a point source (uses photutils)
## INPUT:
##        1. image where source is located
##        2. error in image (see e.g. routine 'noise')
##        3. mask of image (where values are ok)
##        4. center (ypix,xpix) where approximately star is found
## OPTIONAL INPUT:
##        savefile: path+basic_file name where to save results
## OUTPUT: photometry,error: PSF photometry and its associated error

## WARNING: Error calculated right now is not a statistical error but a systematic error from
##          choosing different initial PSF guess.
##          Statistical error is implemented in new version of photutils... to be installed
## DEPENDENCIES: photutils,astropy (python)

def psf_phot(image,errimage,mask,center,savefile=None):  

    ##Check if already done
    if (savefile is not None) & (os.path.isfile(savefile+'_psfphot.npy')):
        print("   Found existing phot file: %s" %(savefile+'_psfphot.npy'))
        flux,error = np.load(savefile+'_psfphot.npy')
        return flux,error
    
    rmax = 25
    icenter = np.asarray(center,dtype='int')
    #timage = image[icenter[1]-rmax:icenter[1]+rmax,icenter[0]-rmax:icenter[0]+rmax]
                   
    from photutils import fit_2dgaussian
    from photutils.psf import IntegratedGaussianPRF, DAOGroup, BasicPSFPhotometry
    from photutils.background import MMMBackground #, MADStdBackgroundRMS
    from astropy.modeling.fitting import LevMarLSQFitter
    from astropy.stats import gaussian_sigma_to_fwhm
    from astropy.table import Table
    print("   PSF photometry")
    
    ##preliminary 2d gaussian fit
    fit2g = fit_2dgaussian(image[icenter[1]-rmax:icenter[1]+rmax,
                                 icenter[0]-rmax:icenter[0]+rmax])
    sigma_psf1 = 0.5*(np.abs(fit2g.x_stddev)+np.abs(fit2g.y_stddev))
    sigma_psf2 = np.abs(fit2g.y_stddev)
    
    ##psf phot
    mmm_bkg = MMMBackground()
    fitter = LevMarLSQFitter()
    pos = Table(names=['x_0', 'y_0'], data=[[center[0]],[center[1]]])

    ##psf1
    daogroup1 = DAOGroup(2.0*sigma_psf1*gaussian_sigma_to_fwhm)
    psf_model1 = IntegratedGaussianPRF(sigma=sigma_psf1)
    psf_model1.x_0.fixed = True
    psf_model1.y_0.fixed = True
    photometry1 = BasicPSFPhotometry(group_maker=daogroup1,
                                    bkg_estimator=mmm_bkg,
                                    psf_model=psf_model1,
                                    fitter=LevMarLSQFitter(),
                                    fitshape=(rmax,rmax))
    result1 = photometry1(image=image,positions=pos)
    ##psf2
    daogroup2 = DAOGroup(2.0*sigma_psf2*gaussian_sigma_to_fwhm)
    psf_model2 = IntegratedGaussianPRF(sigma=sigma_psf2)
    psf_model2.x_0.fixed = True
    psf_model1.y_0.fixed = True
    photometry2 = BasicPSFPhotometry(group_maker=daogroup2,
                                    bkg_estimator=mmm_bkg,
                                    psf_model=psf_model2,
                                    fitter=LevMarLSQFitter(),
                                    fitshape=(rmax,rmax))
    result2 = photometry2(image=image,positions=pos)
    #residual_image = photometry.get_residual_image()

    phot1 = np.asarray(result1['flux_fit'],dtype=float)
    phot2 = np.asarray(result2['flux_fit'],dtype=float)
    phot = phot1.reshape(1,1)#np.mean([phot1,phot2]).reshape(1,1)
    #erphot = np.asarray(result1['flux_fit_unc'],dtype=float)
    syserphot = np.std([phot1,phot2]).reshape(1,1)
    erphot=syserphot
    print("      Flux= %f +/- %f" %(phot,erphot))
    print("      Systematic error of diff 2G sigma= +/- %f" %(syserphot))
    
    if (savefile is not None):
        np.save(savefile+'_psfphot.npy',(phot,erphot))

    return (phot,erphot)
    
   
## ---------------------------------------------------------------------------------
## ---------- FUNCTION TO CALCULATE APERTURE PHOT-----------------------------------
## ---------------------------------------------------------------------------------
## PURPOSE: Calculate aperture photometry of a point source (uses photutils)
## INPUT:
##        1. image where source is located
##        2. error in image (see e.g. routine 'noise')
##        3. mask of image (where values are ok)
##        4. center (ypix,xpix) where approximately star is found
## OPTIONAL INPUT:
##        - radpix: Radius at which to do aperture photometry (pix).
##          It can be one of four things:
##                a. None: in this case several radii are calculated and
##                         the user is interactively asked to choose
##                b. A fixed number in pix
##                c. A string stating the percentage of the curve, e.g. '95perc'
##                d. A string stating the number of FWHM, e.g. '1.5852fwhm'
##        - savefile: path+basic_file name where to save results
## OUTPUT: photometry,error: Aperture photometry and its associated error

## Note: Two errors are calculated: a statistical one from photutils (too small)
##       and a systematic (much larger) from choosing slightly different radii.
##       Right now systematic is returned. 

## DEPENDENCIES: photutils,scipy (python)

def aperture_phot(image,errimage,mask,center,radpix=None,savefile=None):

    #icenter = np.asarray(center,dtype='int')
    #image = timage[icenter[1]-50:icenter[1]+50,icenter[0]-50:icenter[0]+50]
    #mask = tmask[icenter[1]-50:icenter[1]+50,icenter[0]-50:icenter[0]+50]

    ##Check if already done
    if (savefile is not None) & (os.path.isfile(savefile+'_aperphot.npy')):
        print("   Found existing phot file: %s" %(savefile+'_aperphot.npy'))
        flux,error,radpix = np.load(savefile+'_aperphot.npy')
        return flux,error,radpix
    
    from photutils import CircularAperture,CircularAnnulus,aperture_photometry,fit_2dgaussian
    from scipy.interpolate import interp1d
    print("   Aperture photometry")
    
    ##1.calculate background with annulus (not needed if radius known, since pol erases)
    ##2.aperture photometry (at different radii if radius not known)

    ##apertures
    perc,nsig = None,None
    if radpix is not None:
        if type(radpix) == str:
            if 'perc' in radpix:
                perc = np.float(radpix.split('perc')[0])/100.0
            elif 'fwhm' in radpix:
                nsig = np.float(radpix.split('fwhm')[0])
            rmin,rmax = 3,26
        else:
            rmin,rmax = np.round(radpix)-3,np.round(radpix)+3
    else:
        rmin,rmax = 3,26
        
    radii = np.arange(rmin,rmax,dtype=float)
    annulus = CircularAnnulus(center,r_in=rmax+5.,r_out=rmax+7.)
    aperture = [CircularAperture(center, r=r) for r in radii]
    aperture.append(annulus)
    
    ##photometry    
    phot_table = aperture_photometry(image,aperture,mask=~mask,error=errimage)

    ##background
    bkg_mean = phot_table['aperture_sum_'+str(radii.size)]/annulus.area()
    bkg_sum = [bkg_mean * a.area() for a in aperture]
    final_sum = np.asarray([phot_table['aperture_sum_'+str(i)]-bkg_sum[i]
                            for i in np.arange(0,radii.size)],dtype=float)
    erfinal_sum = np.asarray([phot_table['aperture_sum_err_'+str(i)]
                              for i in np.arange(0,radii.size)],dtype=float)
    norm_final_sum =np.reshape(final_sum/np.max(final_sum),-1)
    norm_erfinal_sum =np.reshape(erfinal_sum/np.max(final_sum),-1)

    if perc is not None:
        fct  = interp1d(radii,norm_final_sum)
        invfct = interp1d(norm_final_sum,radii)
        percrad = invfct(perc)
        print('      %f perc of radius: %f' %(perc,percrad))
        radpix = percrad
        
    if nsig is not None:
        ##Gaussian 2D fit for the FWHM
        icenter = np.asarray(center,dtype='int')
        fit2g = fit_2dgaussian(image[icenter[1]-rmax:icenter[1]+rmax,
                                     icenter[0]-rmax:icenter[0]+rmax])
        fw = np.sqrt(8*np.log(2))*0.5*(fit2g.x_stddev+fit2g.y_stddev)
        radpix = nsig*fw
        print('      FWHM radius (%f sig): %f '%(nsig,radpix))
    
    ##plot and ask user for best radius
    if radpix == None:

        # 95% perc radius
        fct  = interp1d(radii,norm_final_sum)
        invfct = interp1d(norm_final_sum,radii)
        percrad = invfct(0.95)
        print('      95%% radius: %f' %percrad)
        # Gaussian 2D fit
        icenter = np.asarray(center,dtype='int')
        fit2g = fit_2dgaussian(image[icenter[1]-rmax:icenter[1]+rmax,
                                     icenter[0]-rmax:icenter[0]+rmax])
        fw = np.sqrt(8*np.log(2))*0.5*(fit2g.x_stddev+fit2g.y_stddev)
        optrad = 1.5852*fw
        print('      Optimum radius (1.58sig): %f '%optrad)

        fig,ax = plt.subplots(1,figsize=(8,8))
        ax.errorbar(radii,norm_final_sum,yerr=ernorm_final_sum,fmt='o')
        ax.plot(radii,norm_final_sum)
        if (optrad > 0) and (optrad < rmax): 
            ax.plot([optrad,optrad],[0,1],'--',label='Radius at 1.5852 sig of 2D Gaussian')
        ax.plot([percrad,percrad],[0,1],'--',label='Radius at 95% flux')

        #ax.set_ylim([np.min(norm_final_sum),np.max(norm_final_sum)])
        ax.set_xlabel('Radius (pix)')
        ax.set_ylabel('Normalized Flux')
        ax.legend()
        fig.show()
        
        radpix = np.asarray(raw_input("      Which radius (pix) do you want to choose? "),dtype=float)
        print('      For radius=%f, percentarge flux is %f' %(radpix,fct(radpix)))
        plt.close(fig)
        
        
    flux = final_sum[np.round(radii) == np.round(radpix)]
    error = erfinal_sum[np.round(radii) == np.round(radpix)]
    
    ##systematic error from radpix choice
    syserror = np.std(final_sum[((radii >= radpix-2.0) & (radii <= radpix+2.0))]).reshape(flux.shape)
    error = syserror
    
    print("      Flux = %f +/- %f" %(flux,error))
    print("      Systematic radius error: +/- %f" %syserror)
    
    if (savefile is not None):
        np.save(savefile+'_aperphot.npy',(flux,error,radpix))
    
    return (flux,error,radpix)

   
## ---------------------------------------------------------------------------------
## ---------- FUNCTION TO CALCULATE APERTURE PHOT OF FIELD STARS--------------------
## ---------------------------------------------------------------------------------
## PURPOSE: Calculate aperture photometry of many field point source (uses photutils)
##          First finds stars from summed-angle image and uses those positions
##          for obeam/ebeam of all angles
## INPUT: 1. image
##        2. error in image (see e.g. routine 'noise')
##        3. mask of image (where values are ok)
## OPTIONAL INPUT:
##        - radpix: Radius at which to do aperture photometry (pix).
##          It can be one of three things:
##                a. A fixed number in pix
##                b. A string stating the percentage of the curve, e.g. '95perc'
##                c. A string stating the number of FWHM, e.g. '1.5852fwhm'
##        - fwhm: Full width half max to search for stars (def: 5) [see find_stars]
##        - threshold: The absolute image value above sky to select sources (def: 5) [see find_stars]
##        - sumfile: path+basic_file name where to save/laod summed-ang images
##        - savefile: path+file name where to save results
##        - files1/2: File names of all angles for chip1/2 to get summed-angle image [see 'data_flat']
##        - dobias,bias1/2,docosmic: parameters to get summed-ang image [see 'data_flat']
## OUTPUT: photometry,error: Aperture photometry and its associated error for all found stars
## DEPENDENCIES: photutils,ascii

def field_apphot(img,errimg,mask,savefile=None,sumfile=None,radpix=None,fwhm=5.0,threshold=5.0,\
                 dobias=None,docosmic=None,bias1=None,bias2=None,files1=None,files2=None,bkg=True):


    ## See if file exists
    if (savefile is not None) & (os.path.isfile(savefile+'-fwhm'+str(fwhm)+'-thresh'+\
                                                str(threshold)+'-apflux.dat')):
        print("   Found existing AP field-phot file: %s"
              %(savefile+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-apflux.dat'))
        fread = ascii.read(savefile+'-fwhm'+str(fwhm)+'-thresh'+
                           str(threshold)+'-apflux.dat')
        return fread#['apflux'].reshape((len(fread),1)),fread['aperror'].reshape((len(fread),1))

    from photutils import CircularAperture,CircularAnnulus,aperture_photometry,fit_2dgaussian
    from astropy.stats import gaussian_sigma_to_fwhm
    
    ## I. See if position file already exists
    aux = '-sumang-obeam-merged'
    if (sumfile is not None) & (not os.path.isfile(sumfile+aux+'-fwhm'+str(fwhm)+'-thresh'+
                                                str(threshold)+'-xycoord.dat')): 
        
        ## 1. First get summed images (use data_flat procedure)
        print("   Calculating summed angle image") 
        if files1 is None: print("     Error in field_phot: pass files") 
        retflat = data_flat(files1,files2,savefile=sumfile,#outdir+fname,
                            dobias=dobias,docosmic=docosmic,bias1=bias1,bias2=bias2)
    
        ## 2. Find stars from summed angles (if not already) and save positions
        head,sumimg = read_fits(sumfile+aux+'.fits')
        sources = find_stars(img,fwhm=fwhm,threshold=threshold)
        
        ## write and plot
        ascii.write(sources,sumfile+aux+'-fwhm'+str(fwhm)+'-thresh'+
                    str(threshold)+'-xycoord.dat',overwrite=True)
        fig = plt.figure(figsize=(12,7))
        sumimg1 = sumimg#np.arcsinh(sumimg)
        l1 = np.percentile(sumimg1[np.isfinite(sumimg1)].flatten(), 1)
        l2 = np.percentile(sumimg1[np.isfinite(sumimg1)].flatten(), 99)
        plt.imshow(sumimg1,clim=(l1,l2))#norm=norm1,origin='lower')
        plt.plot(sources["xcentroid"],sources["ycentroid"],'o',mfc='none')
        plt.xlabel("x-obeam (pix)")
        plt.ylabel("y-obeam (pix)")
        plt.gca().invert_yaxis()#ylim(plt.get_ylim()[::-1])
        plt.savefig(sumfile+aux+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-image.png') 
        plt.close(fig)
        
    ## 2. Do photometry at positions
    print("   Doing field source APERTURE photometry") 
    ny,nx = np.shape(img)

    ## Load positions
    sources = ascii.read(sumfile+aux+'-fwhm'+str(fwhm)+'-thresh'+
                         str(threshold)+'-xycoord.dat')
    positions = [(sources["xcentroid"][i],sources["ycentroid"][i]) for i in range(0,len(sources))]

    ## Different radii
    perc,nsig = None,None
    if radpix is not None:
        if type(radpix) == str:
            if 'perc' in radpix:
                perc = np.float(radpix.split('perc')[0])/100.0
            elif 'fwhm' in radpix:
                nsig = np.float(radpix.split('fwhm')[0])
            rmin,rmax = 3,8
        else:
            rmin,rmax = np.round(radpix)-3,np.round(radpix)+3
    else:
        print("   Error in Field_phot: You need some radpix input!")
        rmin,rmax = 3,8
    radii = np.arange(rmin,rmax,dtype=float)
        
    ## Calculate phot at positions at different radii & also annulus at diff radii
    annulus = CircularAnnulus(positions,r_in=rmax+1.,r_out=rmax+3.)
    apertures = [CircularAperture(positions,r=r) for r in radii]
    apertures.append(annulus)
    phot_table = aperture_photometry(img,apertures,mask=~mask,error=errimg)
    
    ## Background
    bkg_mean = phot_table['aperture_sum_'+str(radii.size)]/annulus.area()
    bkg_sum = np.zeros(len(apertures))
    if bkg: bkg_sum = [bkg_mean * a.area() for a in apertures]
    final_sum = np.asarray([phot_table['aperture_sum_'+str(i)]-bkg_sum[i]
                            for i in np.arange(0,radii.size)],dtype=float)
    erfinal_sum = np.asarray([phot_table['aperture_sum_err_'+str(i)]
                              for i in np.arange(0,radii.size)],dtype=float)
    norm_final_sum = final_sum/np.max(final_sum,axis=0)
    norm_erfinal_sum =erfinal_sum/np.max(final_sum,axis=0)

    ## Proper fwhm estimate: Multiple sources - take 40 brightest outside of center
    brightest = np.argsort(sources['flux'])[::-1][0:40]
    
    if perc is not None: 
        iradpix = np.zeros(len(brightest),dtype=float)
        for i in range(0,len(brightest)):
            fct  = interp1d(radii,norm_final_sum[:,brightest[i]])
            invfct = interp1d(norm_final_sum[:,brightest[i]],radii)
            percrad = invfct(perc)
            iradpix[i] = percrad
        radpix = np.median(iradpix)
    if nsig is not None:
        iradpix = np.zeros(len(brightest),dtype=float)
        for i in range(0,len(brightest)):
            ip = brightest[i]
            if np.int(positions[ip][1]) > rmax and np.int(positions[ip][1]) < ny-rmax and \
               np.int(positions[ip][0]) > rmax and np.int(positions[ip][0]) < nx-rmax and \
               np.sqrt(np.int(positions[ip][1])**2.0+np.int(positions[ip][0])**2.0) > 400:
                fit2g = fit_2dgaussian(img[np.int(positions[ip][1])-rmax:np.int(positions[ip][1])+rmax,
                                           np.int(positions[ip][0])-rmax:np.int(positions[ip][0])+rmax])
                psf = 0.5*(np.abs(fit2g.x_stddev)+np.abs(fit2g.y_stddev))
                fw = psf*gaussian_sigma_to_fwhm#np.sqrt(8*np.log(2))*psf
                iradpix[i] = nsig*fw
        radpix = np.median(iradpix[iradpix > 0])

    ## Final fluxes/errors
    flux = final_sum[np.round(radii) == np.round(radpix),:].reshape(-1)
    error = erfinal_sum[np.round(radii) == np.round(radpix),:].reshape(-1)

    ##Save
    ascii.write([sources['id'],sources['xcentroid'],sources['ycentroid'],flux,error],
                savefile+'-fwhm'+str(fwhm)+'-thresh'+\
                str(threshold)+'-apflux.dat',overwrite=True,
                names=['id','xfit','yfit','apflux','aperror'])
    fread = ascii.read(savefile+'-fwhm'+str(fwhm)+'-thresh'+\
                       str(threshold)+'-apflux.dat')
    
    return fread#flux.reshape((len(flux),1)),error.reshape((len(flux),1))

## ---------------------------------------------------------------------------------
## ---------- FUNCTION TO CALCULATE PSF PHOT OF FIELD STARS--------------------
## ---------------------------------------------------------------------------------
## PURPOSE: Calculate psf photometry of many field point source (uses photutils)
##          First finds stars from summed-angle image and uses those positions
##          for obeam/ebeam of all angles
## INPUT: 1. image
##        2. error in image (see e.g. routine 'noise')
##        3. mask of image (where values are ok)
## OPTIONAL INPUT:
##          It can be one of three things:
##                a. A fixed number in pix
##                b. A string stating the percentage of the curve, e.g. '95perc'
##                c. A string stating the number of FWHM, e.g. '1.5852fwhm'
##        - fwhm: Full width half max to search for stars (def: 5) [see find_stars]
##        - threshold: The absolute image value above sky to select sources (def: 5) [see find_stars]
##        - sumfile: path+basic_file name where to save/laod summed-ang images
##        - savefile: path+file name where to save results
##        - files1/2: File names of all angles for chip1/2 to get summed-angle image [see 'data_flat']
##        - dobias,bias1/2,docosmic: parameters to get summed-ang image [see 'data_flat']
## OUTPUT: Table with photometry 'psfflux',error 'psferror'...: PSF photometry and its associated error for all found stars
## DEPENDENCIES: photutils,ascii

def field_psfphot(img,errimg,mask,savefile=None,sumfile=None,fwhm=5.0,threshold=5.0,\
                  bbox=-1,dobias=None,docosmic=None,bias1=None,bias2=None,files1=None,files2=None,
                  posfree=True,bkg=True):

    pos,fw = '',''
    if posfree:
        pos='-posfree'

    if bbox > 0:
        fw = '-bbox'+str(bbox)
    else:
        fw = '-fwhm'+str(fwhm)
          
    ## See if file exists
    if (savefile is not None) & (os.path.isfile(savefile+fw+'-thresh'+\
                str(threshold)+pos+'-psfflux.dat')):
        print("   Found existing PSF field-phot file: %s"
              %(savefile+fw+'-thresh'+str(threshold)+pos+'-psfflux.dat'))
        fread = ascii.read(savefile+fw+'-thresh'+\
                str(threshold)+pos+'-psfflux.dat')
        return fread#['psfflux'].reshape((len(fread),1)),fread['psferror'].reshape((len(fread),1))

    from photutils import fit_2dgaussian
    from photutils.detection import DAOStarFinder,IRAFStarFinder
    from photutils.psf import IntegratedGaussianPRF,prepare_psf_model
    from photutils.psf import DAOGroup,IterativelySubtractedPSFPhotometry,DAOPhotPSFPhotometry,BasicPSFPhotometry
    from photutils.background import MMMBackground,MADStdBackgroundRMS
    from astropy.modeling.fitting import LevMarLSQFitter
    from astropy.stats import gaussian_sigma_to_fwhm
    from astropy.table import Table
    
    ## I. See if position file already exists (used if posfree=False and for fwhm estimate)
    aux = '-sumang-obeam-merged'
    if (sumfile is not None) & (not os.path.isfile(sumfile+aux+fw+'-thresh'+
                                                       str(threshold)+'-xycoord.dat')): 
        
        ## 1. First get summed images (use data_flat procedure)
        print("   Calculating summed angle image") 
        if files1 is None: print("     Error in field_phot: pass files") 
        retflat = data_flat(files1,files2,savefile=sumfile,#outdir+fname,
                            dobias=dobias,docosmic=docosmic,bias1=bias1,bias2=bias2)
    
        ## 2. Find stars from summed angles (if not already) and save positions [DAOStarFinder]
        head,sumimg = read_fits(sumfile+aux+'.fits')
        sources = find_stars(img,fwhm=fwhm,threshold=threshold)

        ## write and plot
        ascii.write(sources,sumfile+aux+fw+'-thresh'+
                    str(threshold)+'-xycoord.dat',overwrite=True)
        fig = plt.figure(1,figsize=(12,7))
        ax = fig.add_subplot(111)
        sumimg1 = sumimg#np.arcsinh(sumimg)
        l1 = np.percentile(sumimg1[np.isfinite(sumimg1)].flatten(), 1)
        l2 = np.percentile(sumimg1[np.isfinite(sumimg1)].flatten(), 99)
        ax.imshow(sumimg1,clim=(l1,l2))#norm=norm1,origin='lower')
        ax.plot(sources["xcentroid"],sources["ycentroid"],'o',mfc='none')
        ax.set_xlabel("x-obeam (pix)")
        ax.set_ylabel("y-obeam (pix)")
        ax.invert_yaxis()
        fig.savefig(sumfile+aux+fw+'-thresh'+str(threshold)+'-image.png') 
        plt.close(fig)

    ## Load positions
    sources = ascii.read(sumfile+aux+fw+'-thresh'+
                         str(threshold)+'-xycoord.dat')
    positions = [(sources["xcentroid"][i],sources["ycentroid"][i]) for i in range(0,len(sources))]
            
    ## 2. Do photometry (at positions)
    print("   Doing field source PSF photometry") 
    ny,nx = np.shape(img)

    ## rmax
    rmax = 7
        
    ## Proper fwhm estimate: Multiple sources - take 40 brightest outside of center
    brightest = np.argsort(sources['flux'])[::-1][0:40]
    allxpsf,allypsf = np.zeros(len(brightest),dtype=float),np.zeros(len(brightest),dtype=float)
    allpsf = np.zeros(len(brightest),dtype=float)
    for i in range(0,len(brightest)):
        ip = brightest[i]
        if np.int(positions[ip][1]) > rmax and np.int(positions[ip][1]) < ny-rmax and \
           np.int(positions[ip][0]) > rmax and np.int(positions[ip][0]) < nx-rmax and \
           np.sqrt(np.int(positions[ip][1])**2.0+np.int(positions[ip][0])**2.0) > 400:
            fit2g = fit_2dgaussian(img[np.int(positions[ip][1])-rmax:np.int(positions[ip][1])+rmax,
                                       np.int(positions[ip][0])-rmax:np.int(positions[ip][0])+rmax])
            allxpsf[i] = np.abs(fit2g.x_stddev)
            allypsf[i] = np.abs(fit2g.y_stddev)
            allpsf[i] = 0.5*(np.abs(fit2g.x_stddev)+np.abs(fit2g.y_stddev))
    sigma_psf = np.median(allpsf)
    print("      FWHM: %f" %(sigma_psf*gaussian_sigma_to_fwhm))
    
    ## psf phot
    mmm_bkg = MMMBackground()
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(img[mask])
    print("      Background RMS: %f" %std)
    bkg_estimator = mmm_bkg if bkg else None
    fitter = LevMarLSQFitter()
    iraffinder = IRAFStarFinder(threshold=threshold*std,fwhm=sigma_psf*gaussian_sigma_to_fwhm)#,
    #                            minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
    #                            sharplo=0.0, sharphi=2.0)
    daofinder = DAOStarFinder(threshold=threshold*std,fwhm=sigma_psf*gaussian_sigma_to_fwhm)
    daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    #from astropy.convolution import Gaussian2DKernel
    #gaussian_2D_kernel = Gaussian2DKernel(np.median(allxpsf),y_stddev=np.median(allypsf))
    #psf_model = prepare_psf_model(gaussian_2D_kernel)
    
    ## pos fixed
    if not posfree:
        position = Table()
        position['x_0'] = [p[0] for p in positions]
        position['y_0'] = [p[1] for p in positions]
        photometry = BasicPSFPhotometry(group_maker=daogroup,
                                        bkg_estimator=bkg_estimator,
                                        psf_model=psf_model,
                                        fitter=LevMarLSQFitter(),
                                        fitshape=(rmax,rmax))
        psf_model.x_0.fixed = True
        psf_model.y_0.fixed = True
        result = photometry(img,positions=position)
        
    ## pos free
    else:
        #photometry = DAOPhotPSFPhotometry(crit_separation=2.0*sigma_psf*gaussian_sigma_to_fwhm,
        #                                  threshold=threshold*std,fwhm=sigma_psf*gaussian_sigma_to_fwhm,
        #                                 fitshape=(rmax,rmax),
        #                                  psf_model=psf_model,niters=3)

        photometry = IterativelySubtractedPSFPhotometry(group_maker=daogroup,
                                                        finder=iraffinder,#daofinder,#
                                                        bkg_estimator=bkg_estimator,
                                                        psf_model=psf_model,
                                                        fitter=LevMarLSQFitter(),
                                                        fitshape=(rmax,rmax),niters=10)           
        result = photometry(img)

    res_img = photometry.get_residual_image()
    x,y = np.asarray(result['x_fit'],dtype=float),np.asarray(result['y_fit'],dtype=float)
    phot = np.asarray(result['flux_fit'],dtype=float)
    error = 0.1*phot###OJO: should get photutils higher version for error!!!

    ## Plot image & residual
    fits.writeto(savefile+fw+'-thresh'+str(threshold)+pos+'-residualimage.fits',
                 res_img,clobber=True)
    with open(savefile+fw+'-thresh'+str(threshold)+pos+'-pol.pkl', 'wb') as f: #python3: 'wb'
        pickle.dump([photometry,result,res_img], f)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig,ax = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    fig.subplots_adjust(wspace=0.3)
    lo = np.percentile(img[np.isfinite(img)].flatten(), 5)
    up = np.percentile(img[np.isfinite(img)].flatten(), 95)
    im = ax[0].imshow(img, cmap='viridis',clim=(lo,up),aspect=1,interpolation='nearest',
                   origin='lower')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=6) 
    ax[0].set_title('Image')
    #ax[0].colorbar(orientation='vertical', fraction=0.046, pad=0.04)
    #plt.subplot(1 ,2, 2)
    lo = np.percentile(res_img[np.isfinite(res_img)].flatten(), 5)
    up = np.percentile(res_img[np.isfinite(res_img)].flatten(), 95)
    im = ax[1].imshow(res_img, cmap='viridis', clim=(lo,up),aspect=1,
               interpolation='nearest', origin='lower')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=6) 
    ax[1].set_title('Residual Image')
    #ax[1].colorbar(orientation='vertical', fraction=0.046, pad=0.04)
    fig.savefig(savefile+fw+'-thresh'+str(threshold)+pos+'-resimage.png') 
    
    ## Final fluxes
    ascii.write([result['id'],x,y,phot,error],
                savefile+fw+'-thresh'+\
                str(threshold)+pos+'-psfflux.dat',overwrite=True,
                names=['id','xfit','yfit','psfflux','psferror'])
    fread = ascii.read(savefile+fw+'-thresh'+\
                       str(threshold)+pos+'-psfflux.dat')
    
    return fread#x.reshape((len(x),1)),y.reshape((len(x),1)),phot.reshape((len(phot),1)),error.reshape((len(phot),1))

## ----------------------------------------------------------------------
## ---------- FUNCTION TO FIT ISOPHOT ELLIPSES TO GALAXY ---------------
## ----------------------------------------------------------------------
## PURPOSE: Fit elliptical model to galaxy (image should be combined and hopefully without strips)
##          to obtain ultimately mask to get background polarization correction
## INPUT: 1. savefile: full path+savefile of file to investigate
##        2. center: x,y values as initial estimate
##        3,4. ra, dec: Ra,Dec at center of image
## OPTIONAL INPUT:
##       - galradius: initial semimajor axis guess (in between min and max in pixel)
##       - ellipticity: initial guess of ellipiticity
##       - pa: initial position angle guess (degrees)
## OUPUT: fit mask (largest ellipse) for background fit
##        Creates plots of center position vs semimajor axis, intensity/total flux vs sma,
##                         elliptical model and residual
##        Creates file of fit result of ellipse model (use this last line to get mask)

def galisophot(savefile,center,ra,dec,galradius=100.0,ellipticity=0.7,pa=140.0):

    print("  Isophot for %s" %savefile)
    sfile = savefile.replace('.fits','')

    ## see if file already done
    if os.path.exists(sfile+'-galmask.npy'):
        fitmask,galradius = np.load(sfile+'-galmask.npy')
        return fitmask,galradius

    from photutils.isophote import EllipseGeometry, Ellipse
    from photutils import EllipticalAperture
    
    ## read file
    head,img = read_fits(savefile)
    
    ## mask for strip
    #mask = (img > 0)
    #image = np.ma.masked_array(img,mask=(~mask))
    #image[~mask] = np.nan#median(image[mask])

    ## interpolation for strip (mask does not work)
    from scipy.interpolate import interp2d,RectBivariateSpline,SmoothBivariateSpline,griddata
    image = np.zeros(img.shape,dtype=float)
    ny,nx = img.shape
    x,y = np.arange(0,nx),np.arange(0,ny)
    xx,yy = np.meshgrid(x,y)
    f2 = interp2d(x,[y[1000:1023],y[1042:1065]],[img[1000:1023,:],img[1042:1065,:]])
    image = img.copy()
    image[1023:1042,:] = f2(x,y[1023:1042])

    ## initial ellipse estimate (rad=200, pa=140)
    geometry = EllipseGeometry(x0=center[0],y0=center[1],sma=galradius,
        eps=ellipticity,pa=pa*np.pi/180.)

    ## to plot
    aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
   	 	    geometry.sma*(1 - geometry.eps),geometry.pa)
    lo,up = np.percentile(image.flatten(), 5),np.percentile(image.flatten(), 95)
    plt.imshow(image, origin='lower',clim=(lo,up))
    aper.plot(color='white')
    plt.show()

    # fit
    ellipse = Ellipse(image, geometry)   
    isolist = ellipse.fit_image(maxsma=700,fflag=0.4,maxgerr=1.0)
    pdb.set_trace()
    np.savetxt(sfile+'-galisophot.dat',(isolist.to_table()),fmt="%20s",
    	header="RA=%f DEC=%f" %(ra,dec)+"\n sma intens intens_err ellipticity ellipticity_err pa pa_err "
    	+"grad_rerr ndata flag niter stop_code")

    # mask
    bsma = len(isolist)-1
    aper = EllipticalAperture((isolist[bsma].x0, isolist[bsma].y0), isolist[bsma].sma,
   	 	              isolist[bsma].sma*(1 - isolist[bsma].eps),isolist[bsma].pa)
    smask = aper.to_mask(method='center')[0].data
    my,mx = smask.shape
    cy,cx = np.int(aper.positions[0][0]),np.int(aper.positions[0][1])
    fitmask = np.zeros(img.shape,dtype=bool)
    fitmask[cy-my/2:cy+my/2,cx-mx/2:cx+mx/2] = smask
    np.save(sfile+'-galmask.npy',(fitmask,np.int(isolist[bsma].sma)))
    
    # plot curve of growth
    fig,(ax1,ax2) = plt.subplots(figsize=(14, 5), nrows=1, ncols=2)
    ax1.errorbar(isolist.sma,isolist.intens,isolist.int_err,fmt='o')
    ax1.set_xlabel('Semimajor axis (pix)')
    ax1.set_ylabel('Mean intensity in ellipse')
    ax2.scatter(isolist.sma,isolist.tflux_e/(np.pi*isolist.sma**2.0*(1.0-isolist.ellipticity)))
    ax2.set_xlabel('Semimajor axis (pix)')
    ax2.set_ylabel('Flux within ellipse/Area')
    fig.savefig(sfile+'-galcurve.png')

    # plot centers
    fig,(ax1,ax2) = plt.subplots(figsize=(14, 5), nrows=1, ncols=2)
    ax1.errorbar(isolist.sma,isolist.x0,isolist.x0_err,fmt='o')
    ax1.set_xlabel("Semimajor axis (pix)")
    ax1.set_ylabel("x0")
    ax2.errorbar(isolist.sma,isolist.y0,isolist.y0_err,fmt='o')
    ax2.set_xlabel("Semimajor axis (pix)")
    ax2.set_ylabel("y0")
    fig.savefig(sfile+'-galcenter.png')

    #model image (not really needed)
    from photutils.isophote import build_ellipse_model
    model_image = build_ellipse_model(image.shape, isolist)
    residual = image - model_image

    #plot ellipses
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 5), nrows=1, ncols=3)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
    lo,up = np.percentile(image.flatten(), 5),np.percentile(image.flatten(), 95)
    c = ax1.imshow(image, origin='lower',clim=(lo,up))
    ax1.set_title('Data')
    smas = np.linspace(10, np.max(isolist.sma), 10)
    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        ax1.plot(x, y, color='white')

    c2 = ax2.imshow(model_image, origin='lower',clim=(lo,up))#
    ax2.set_title('Ellipse Model')

    c3 = ax3.imshow(residual, origin='lower',clim=(lo,up))
    ax3.set_title('Residual')
    plt.savefig(sfile+'galisophot.png')
   
    plt.close("all")
    return fitmask,np.int(isolist[bsma].sma)


## ---------------------------------------------------------------------
## ---------- FUNCTION TO BIN POINTS (clipped averge) ------------
## ---------------------------------------------------------------------
## PURPOSE: Bin given scattered points to larger boxes
## INPUT:   1. x positions
##          2. y position
##          3. Value at x,y positions
## OPTIONAL INPUT:
##          - radpix: Half of the box size to do binning in pixels, def: 40
##          - sigmaclip: # Sigma to do clipping in each box (def: 3.0)
##          - savefile: path+basic_name to save results
##          - center: If you want to make sure binning goes through particular point
##          - fullbin: Fill all pixels within bin with same center value (def: False)
## OUPTUT:  outimage,outerimage: Final binned image and associated error
## DEPENDENCIES: astropy (python)

def bin_points(xpts,ypts,zpts,radpix=40,sigmaclip=3.0,percclip=0.1,mask=None,
               ypix=2064,xpix=2049,savefile=None,center=None,fullbin=False):

    ##Check if already done
    if (savefile is not None):
        if (os.path.isfile(savefile+'.fits')):
            print("   Found binning field points: %s" %(savefile+'.fits')) 
            binfile = fits.open(savefile+'.fits')
            img,erimg = binfile[0].data
            return img,erimg
            
    from astropy.stats import sigma_clipped_stats
    from photutils import make_source_mask
    print("   Binning points") 
    xini,yini = 0,0
    if center is not None:
        xstep,ystep = center[0]/radpix,center[1]/radpix
        xini,yini = center[0]-xstep*radpix,center[1]-ystep*radpix

    tmask = None
    if mask is not None: mask = mask.reshape(-1)
    tzpts = zpts.reshape(-1)    
    outimage = np.zeros((ypix,xpix),dtype='float')
    outerimage,outnimage = np.zeros((ypix,xpix),dtype='float') ,np.zeros((ypix,xpix),dtype='int') 
    for i in range(radpix+xini,xpix-radpix,radpix):
        for j in range(radpix+yini,ypix-radpix,radpix):
            inpts = np.argwhere((xpts > i-radpix) & (xpts <= i+radpix-1) & \
                                (ypts > j-radpix) & (ypts <= j+radpix-1)).reshape(-1)
            if mask is not None:
                tmask = ~mask[inpts]

            #print(len(inpts))
            if len(inpts) > 0:
                #sigma clipping
                mean,median,std = sigma_clipped_stats(tzpts[inpts],mask=tmask,
                                                      sigma=sigmaclip,iters=10)
                outnimage[j,i] = len(inpts)
                
                outimage[j,i] = median
                outerimage[j,i] = std
            else:
                outimage[j,i] = np.nan
                
    if fullbin:
        for i in range(radpix+xini,xpix-radpix,radpix):
            for j in range(radpix+yini,ypix-radpix,radpix):
                outnimage[j-radpix/2:j+radpix/2+1,i-radpix/2:i+radpix/2+1] = outnimage[j,i]
                outimage[j-radpix/2:j+radpix/2+1,i-radpix/2:i+radpix/2+1] = outimage[j,i]
                outerimage[j-radpix/2:j+radpix/2+1,i-radpix/2:i+radpix/2+1] = outerimage[j,i]
                #outnimage[j-radpix/2:j+radpix/2,i-radpix/2:i+radpix/2] = outnimage[j,i] ## with lines
                #outimage[j-radpix/2:j+radpix/2,i-radpix/2:i+radpix/2] = outimage[j,i]
                #outerimage[j-radpix/2:j+radpix/2,i-radpix/2:i+radpix/2] = outerimage[j,i]
    
    if savefile is not None:
        fits.writeto(savefile+'.fits',np.asarray((outimage,outerimage),dtype=float),clobber=True)

    return outimage,outerimage
                
## ---------------------------------------------------------------------
## ---------- FUNCTION TO BIN IMAGE (clipped average Pat06) ------------
## ---------------------------------------------------------------------
## PURPOSE: Bin given image to larger boxes in order to increase S/N
## INPUT:   1. image to do binning to
##          2. mask of good values of image
## OPTIONAL INPUT:
##          - radpix: Half of the box size to do binning in pixels, def: 15
##          - sigmaclip: # Sigma to do clipping in each box (def: 3.0)
##          - savefile: path+basic_name to save results
##          - center: If you want to make sure binning goes through particular point
##          - fullbin: Fill all pixels within bin with same center value (def: False)
## OUPTUT:  outimage,outerimage: Final binned image and associated error
## DEPENDENCIES: astropy, photutils (python)

def bin_image(image,mask,radpix=15,sigmaclip=3.0,percclip=0.1,savefile=None,center=None,fullbin=False):

    ##Check if already done
    if ((savefile is not None) &
        (os.path.isfile(savefile+'-bin'+str(radpix)+'pix'+str(sigmaclip)+'sig.fits'))):
        print("   Found existing bin file: %s"
              %(savefile+'-bin'+str(radpix)+'pix'+str(sigmaclip)+'sig.fits'))
        hdu = fits.open(savefile+'-bin'+str(radpix)+'pix'+str(sigmaclip)+'sig.fits')
        outimage,outerimage =  hdu[0].data
        return outimage,outerimage

    from astropy.stats import sigma_clipped_stats
    from photutils import make_source_mask
    print("   Binning image")

    ## This is a 'mean' filter
    #selem = disk(radpix)#square(radpix)
    #outimage = rank.mean_percentile(image.astype(dtype=np.uint16),selem,
    #                                mask=mask.astype(int),
    #                                p0=percclip,p1=1.0-percclip)

    ##This is a rebinning changing shape
    #sh = shape[0],image.shape[0]//shape[0],shape[1],image.shape[1]//shape[1]
    #outimage = image.reshape(sh).mean(-1).mean(1)


    ##USE NDDATA - block reduce!!?
    
    ## -- SG's way:
    xini,yini = 0,0
    if center is not None:
        xstep,ystep = center[0]/radpix,center[1]/radpix
        xini,yini = center[0]-xstep*radpix,center[1]-ystep*radpix
    (ypix,xpix) = np.shape(image)
    outimage = np.zeros((ypix,xpix),dtype='float')
    outerimage,outnimage = np.zeros((ypix,xpix),dtype='float') ,np.zeros((ypix,xpix),dtype='float') 
    for i in range(radpix+xini,xpix-radpix,radpix):
        for j in range(radpix+yini,ypix-radpix,radpix):
            img = image[j-radpix:j+radpix-1,i-radpix:i+radpix-1]
            ind = mask[j-radpix:j+radpix-1,i-radpix:i+radpix-1]
            if img[ind].size > 0.6*(2*radpix*2*radpix):

                #make first star masking
                #stmask = make_source_mask(img, snr=2, npixels=5, mask=~ind)#, dilate_size=11)

                #sigma clipping
                #vals = sigma_clip(img[ind],sigma=sigmaclip)
                mean,median,std = sigma_clipped_stats(img,mask=~ind,sigma=sigmaclip,iters=10)#,mask=~ind)#*stmask)
                outimage[j,i] = median#vals.mean()
                outerimage[j,i] = std#vals.std()

                #number of rejected
                outliers = np.abs(img - median)/std > sigmaclip
                outnimage[j,i] = float(len(img[outliers]))/float(len(img[ind]))
                                  
    if fullbin:
        for i in range(radpix+xini,xpix-radpix,radpix):
            for j in range(radpix+yini,ypix-radpix,radpix):
                outimage[j-radpix/2:j+radpix/2+1,i-radpix/2:i+radpix/2+1] = outimage[j,i]
                outerimage[j-radpix/2:j+radpix/2+1,i-radpix/2:i+radpix/2+1] = outerimage[j,i]
                outnimage[j-radpix/2:j+radpix/2+1,i-radpix/2:i+radpix/2+1] = outnimage[j,i]
                #outimage[j-radpix/2:j+radpix/2,i-radpix/2:i+radpix/2] = outimage[j,i]#w lines
                #outerimage[j-radpix/2:j+radpix/2,i-radpix/2:i+radpix/2] = outerimage[j,i]#w line

    print("    Average outliers: %f" %np.median(outnimage[mask]))
                
    if savefile is not None:
        fits.writeto(savefile+'-bin'+str(radpix)+'pix'+str(sigmaclip)+'sig.fits',
                     np.asarray((outimage,outerimage),dtype=float),clobber=True)

    return outimage,outerimage

## -------------------------------------------------------------------------------
## ------------------------ FUNCTION TO INDICATE INITIAL CENTER -----------------
## -------------------------------------------------------------------------------
## PURPOSE:  find initial estimate of center interactively
## INPUT:    image
## OUTPUT:   center (ypix,xpix)

def inicenter(image):

    global stars1
    stars1 = []
    fig = plt.figure(1,figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.imshow(np.arcsinh(image))
    cid = fig.canvas.mpl_connect('button_press_event',onclick1)
    plt.show(1)
    plt.close(1)
    return stars1[0]


## -------------------------------------------------------------------
## ----------- FIND CENTROID: 2D gaussian, or center of mass ---------
## ---------------------------------------------------------------------
## PURPOSE: Find a centroid of an image based on 2D gaussian fit
## INPUT:   1. image
##          2. mask of good values
## OPTIONAL INPUT:
##          - inicenter: initial guess of the center
##          - radpix: box size in pixels within image,
##                    if you want to restrict the search
## OUTPUT:  centroid (ypix,xpix)
## DEPENDENCIES: photutils (python)

def centroid(matrix,mask,inicenter=None,radpix=50):

    from photutils import centroid_com, centroid_1dg, centroid_2dg

    if inicenter is not None:
        inicenter = np.ndarray.astype(np.round(inicenter),dtype='int')
        matrix = matrix[inicenter[1]-radpix:inicenter[1]+radpix,
                        inicenter[0]-radpix:inicenter[0]+radpix]
        mask = mask[inicenter[1]-radpix:inicenter[1]+radpix,
                    inicenter[0]-radpix:inicenter[0]+radpix]
        
        
    center = centroid_2dg(matrix,mask=~mask)
        
    #sx,sy = np.shape(matrix)
    #x,y = np.arange(sx,dtype='float'),np.arange(sy,dtype='float')
    #xone,yone = np.ones(sx,dtype='float'),np.ones(sy,dtype='float')
    #tot = np.sum(matrix[np.isfinite(matrix)])
    #xout,yout = matrix * np.outer(x,yone),matrix * np.outer(xone,y)
    #xc = np.sum(xout[np.isfinite(xout)])/tot
    #yc = np.sum(yout[np.isfinite(yout)])/tot

    if inicenter is not None:
        center = center + inicenter - [radpix,radpix]
        #xc,yc = xc+inicenter[0]-radpix, yc+inicenter[1]-radpix
    return center

## -------------------------------------------------------------------------------
## ------------------ ALIGN: simple aligh based on RAs/DECs assumed horiz/vert---
## -------------------------------------------------------------------------------
## PURPOSE:  Align two images interactively
## INPUT:    1. image1: image to be aligned
##           2. ra1,dec1: positions of reference
##           3. ra2,dec2: positions of image
## OPTIONAL INPUT:
##           - pixscale (def: 0.126)
##           - binning  (def: 2)
##           - fill_val filling value for outside the shifting range (def:nan)
## OUTPUT:   shifted image1

def align(img,ra0,dec0,ra,dec,pixscale=0.126,binning=2,fill_val=np.nan):
    
    from astropy.coordinates import SkyCoord
    pos = SkyCoord(ra0,dec0,frame='fk5',unit='deg')
    newpos = SkyCoord(ra,dec,frame='fk5',unit='deg')
    sepra,sepdec = newpos.spherical_offsets_to(pos)
    pixsepra = np.int(sepra.value/pixscale*3600/binning)
    pixsepdec = np.int(sepdec.value/pixscale*3600/binning)
    newimg = shiftim(img,pixsepra,pixsepdec,fill_val=fill_val)
    return newimg
    
## -------------------------------------------------------------------------------
## --------------------- MANUALALIGN: FUNCTION to align two images interactively ------------
## -------------------------------------------------------------------------------
## PURPOSE:  Align two images interactively
## INPUT:    1. image1: image to be aligned
##           2. image2: reference image
## OPTIONAL INPUT: savefile: path+basic_file name to save output
## OUTPUT:   shifted image1

## --- These are auxiliary functions
def onclick1(event):
    if event.button != 1:
        print("     Stopping mouse interaction, close window")
        global fig1,cid
        fig1.canvas.mpl_disconnect(cid)
    else:
        global stars1
        print('     star1=%i: x=%d, y=%d, xdata=%f, ydata=%f'%(
            len(stars1)+1, event.x, event.y, event.xdata, event.ydata))
        stars1.append((np.round(event.xdata), np.round(event.ydata)))
    return stars1

def onclick2(event):
    if event.button != 1:
        print("     Stopping mouse interaction")
        global fig2,cid2
        fig2.canvas.mpl_disconnect(cid2)
    else:
        global stars2
        print('     star2=%i, x=%d, y=%d, xdata=%f, ydata=%f'%(
            len(stars2)+1, event.x, event.y, event.xdata, event.ydata))
        stars2.append((np.round(event.xdata), np.round(event.ydata)))
    return stars2    


def manualalign(shiftimg,refimg,savefile=None):

    ##Check if already done
    if (savefile is not None) & (os.path.isfile(savefile+'-align.fits')):
        print("   Found existing align file: %s" %(savefile+'-align.fits'))
        hdu = fits.open(savefile+'-align.fits')
        newim =  hdu[0].data
        return newim
        
    img1,img2 = shiftimg,refimg
    img1[img1 == 0] = np.NAN
    ##img1[img1 == 1400] = np.NAN
    plimg1,plimg2 = np.arcsinh(img1),np.arcsinh(img2)

    global stars1,stars2,fig1,fig2,cid,cid2
    stars1,stars2 = [],[]
    fig1,fig2 = plt.figure(1,figsize=(10,10)),plt.figure(2,figsize=(10,10))
    ax1,ax2 = fig1.add_subplot(111),fig2.add_subplot(111)
    lo = np.percentile(plimg1[np.isfinite(plimg1)].flatten(), 50)
    up = np.percentile(plimg1[np.isfinite(plimg1)].flatten(), 99.9)
    ax1.imshow(plimg1,clim=(lo,up))
    #print((lo,up))
    lo2 = np.percentile(plimg2[np.isfinite(plimg2)].flatten(), 5)
    up2 = np.percentile(plimg2[np.isfinite(plimg2)].flatten(), 99.5)
    ax2.imshow(plimg2,clim=(lo2,up2))
    #print((lo2,up2))
    cid = fig1.canvas.mpl_connect('button_press_event', onclick1)
    cid2 = fig2.canvas.mpl_connect('button_press_event', onclick2)
    plt.show(1)
    plt.show(2)
    plt.close(1)
    plt.close(2)

    boxsize = 20
    nstars = len(stars1)
    x1,x2 = np.zeros(nstars,dtype='float'),np.zeros(nstars,dtype='float')
    y1,y2 = np.zeros(nstars,dtype='float'),np.zeros(nstars,dtype='float')
    for i in range(0,nstars):
        coords1,coords2 = stars1[i],stars2[i]
        #print(i,coords1[0],coords1[1],boxsize,np.shape(img1))
        im1 = img1[int(coords1[1])-boxsize/2:int(coords1[1])+boxsize/2,
                   int(coords1[0])-boxsize/2:int(coords1[0])+boxsize/2]
        im2 = img2[int(coords2[1])-boxsize/2:int(coords2[1])+boxsize/2,
                   int(coords2[0])-boxsize/2:int(coords2[0])+boxsize/2]
        cent1,cent2 = centroid(im1,(im1 > 0)),centroid(im2,(im2 > 0))
        x1[i],y1[i] = cent1[0]+coords1[0]-boxsize/2,cent1[1]+coords1[1]-boxsize/2
        x2[i],y2[i] = cent2[0]+coords2[0]-boxsize/2,cent2[1]+coords2[1]-boxsize/2
        #print("cx1 = %f, cx2 = %f, cy1 = %f, cy2 = %f" %(coords1[1],coords2[1],coords1[0],coords2[0]))
        print("     x1-x2 = %f, y1-y2 = %f" %(x1[i]-x2[i],y1[i]-y2[i]))

    shx,shy = np.median(x2-x1),np.median(y2-y1)
    print("     Median: %f,%f" %(shx,shy))

    #SHIFT
    newim = shiftim(img1,np.int(np.round(shx)),np.int(np.round(shy)))

    if savefile is not None:
        fits.writeto(savefile+'-align.fits',newim,clobber=True)
        wfile = open(savefile+"-align.dat","w")
        for i in range(0,nstars):
            wfile.write('  star field1 Nr %i: x=%d, y=%d \n' %(i,x1[i],y1[i]))
            wfile.write('  star field2 Nr %i: x=%d, y=%d \n' %(i,x2[i],y2[i]))
            wfile.write('  diff star12 Nr %i: dx=%d, dy=%d \n' %(i,x1[i]-x2[i],y1[i]-y2[i]))
        wfile.write('Median difference x1-x2 = %d, y1-y2 = %d\n' %(shx,shy))
        wfile.close()
    return newim

   
## -------------------------------------------------------------------------------
## ------------------------ FUNCTION TO DO ASTROMETRY (astrometry.net) ----------
## -------------------------------------------------------------------------------
## PURPOSE: Find astrometry of an image (uses astrometry.net)
## INPUT:   1. and 2. RA/DEC of center of image (e.g. from header) 
##          3. Path+file name of image to do astrometry
## OPTIONAL INPUT:
##          - radius: radius in deg around RA/DEC where to search (def: 0.5)
##          - outdir: output directory where to save results (def: 'astrometry')
##          - outfile: output file name where write results (def: 'newastrom.fits')
## OUTPUT:  header,newimage
## DEPENDENCIES: astrometry.net

def astrometry(ra,dec,imgfile,radius=0.5,outdir='astrometry',outfile='newastrom.fits'):

    #Use astrometry.net to get solution within radius
    #  (saves a bunch of files with 'new' being new file)
    if ~os.path.isfile(outdir+outfile): 
        os.system("solve-field --config ~/crisp/astrometry/astrometry.cfg --ra "+str(ra)+" --dec "+str(dec)+
                  " --radius "+str(radius)+" --dir "+outdir+" --new-fits "+outdir+outfile+
                  " --cpulimit 1800 "+imgfile)#--overwrite

    #Load result
    hdu = fits.open(outdir+outfile)
    header = hdu[0].header
    data = hdu[0].data
    return header,data

## -------------------------------------------------------------------------------
## ------------------------ FUNCTION TO DO COSMIC RAYS (lacosmic) ---------------
## -------------------------------------------------------------------------------
## PURPOSE:
##        Correct image for cosmic rays (uses lacosmic)
## INPUT:
##        1. image
##        2. header of image
## OPTIONAL INPUT (mostly from lacosmic)
##        - gain: CCD gain (def 2.2)
##        - readnoise: Readout noise (def: 10)
##        - sigclip (def: 5)
##        - sigfrac (def 0.3)
##        - objlim (def: 5.0)
##        - outfile: path+basic_file name where to save results
## OUTPUT
##        cleaned image
## DEPENDENCIES: ncosmics (python)

def cosmic_rays(array,header,gain=2.2,readnoise=10.0,sigclip=5.0,sigfrac=0.3,objlim=5.0,
                outfile=None):

    ##Check if already done
    if (outfile is not None) and (os.path.isfile(outfile+'-cosmic.fits')):
        hdu = fits.open(outfile+'-cosmic.fits')
        return hdu[0].data
    
    print("   Cleaning cosmic-rays")
    import ncosmics as cosmics
    c = cosmics.cosmicsimage(array, gain = gain, readnoise = readnoise, sigclip = sigclip,
                             sigfrac = sigfrac, objlim = objlim)
    c.run(maxiter=4)
    if outfile != None:
        cosmics.tofits(outfile+'-cosmic.fits',np.transpose(c.cleanarray),header)
    return c.cleanarray

## -------------------------------------------------------------------------------
## --------------------------- FUNCTION to read fits files ----------------------
## -------------------------------------------------------------------------------
## PURPOSE: read fits file
## INPUT: file
## OUTPUT: header,image

def read_fits(tfile):
    hdu = fits.open(tfile)
    data = hdu[0].data
    header = hdu[0].header
    hdu.close()
    return header,data

## --------------------------------------------------------------------
## -------------- SHIFTIM: SHIFT IMAGE BY OFFSET ------------------------------
## ---------------------------------------------------------------------
## PURPOSE:  Shift an image by a scalar offset
## INPUT:    1. image
##           2. shift in x (pix)
##           3. shift in y (pix)
## OPTIONAL:
##          fill_val: filling value for outside the shifting range (def:nan)
## OUTPUT:   shited image

def shiftim(img,xs,ys,fill_val=np.nan):

    if (xs == 0) & (ys == 0): return img
    nx,ny = np.shape(img)
    nimg = np.roll(img,ys,axis=0)
    nimg = np.roll(nimg,xs,axis=1)
    if xs > 0: nimg[:,0:xs] = fill_val
    elif xs < 0: nimg[:,nx+xs:nx] = fill_val
    if ys > 0: nimg[0:ys,:] = fill_val
    elif ys < 0:nimg[ny+ys:ny,:] = fill_val

    return nimg

## -------------------------------------------------------------------------------
## ------------------------------------EBEAM_SHIFT ---------------------------
## -------------------------------------------------------------------------------
## PURPOSE:
##         Shift an image (extraordinary beam) continuously (not scalar)
##           by interpolating y_beam according to fit of routine 'find_stars' 
## INPUT:  1. image (extraordinary beam)
##         2. y-difference (output from 'find_stars')
## OPTIONAL INPUT: -savefile: path+basic-file name to save output
## OUTPUT: shifted image
## DEPENDENCIES: scipy (python)

def ebeam_shift(img,ydiff,mask=None,savefile=None):

    if mask is not None: domask=True
    else: domask=False
    
    ##Check if already done
    if (savefile is not None) and (os.path.isfile(savefile+'-shifted.fits')):
        hdu = fits.open(savefile+'-shifted.fits')
        newimg = hdu[0].data
        if domask:
            mask = (newimg > 0)
            #shiftimg,mask = hdu[0].data
            return newimg,mask
        else:
            return newimg
            
    print("   Shifting image")   

    from scipy.ndimage.interpolation import shift as interpshift
    from scipy.interpolate import griddata,RectBivariateSpline
    ny,nx = np.shape(img)
    x,y = np.linspace(0,nx-1,nx), np.linspace(0,ny-1,ny)
    xx,yy = np.meshgrid(x,y)
    newimg = img
    if domask:
        nmask = np.zeros(np.shape(mask),dtype=float)
        nmask[mask] = 1
        yy_mask,xx_mask = np.ma.masked_where(~mask,yy),np.ma.masked_where(~mask,xx)
        newimg_mask = np.ma.masked_where(~mask,newimg)
        nmask_mask = np.ma.masked_where(~mask,nmask)
        
    #tydiff = ydiff.astype(int)
    tydiff = ydiff
    for i in range(0,ny):
        shift = tydiff[i]
        if (shift != 0):
            if tydiff.dtype == int:
                newimg[i:ny,:] = shiftim(newimg[i:ny,:],0,shift)
                if domask: nmask[i:ny,:] = shiftim(nmask[i:ny,:],0,shift)
            else:
                newimg[i:ny,:] = interpshift(newimg[i:ny,:],[shift,0])
                if domask: nmask[i:ny,:] = interpshift(nmask[i:ny,:],[shift,0])
            if i < ny-1:
                tydiff[i+1:ny] = tydiff[i+1:ny]-shift

            #interpolate all this line
            if (i > 2) and (i < ny-2):
                
                if domask:
                    timg = np.concatenate((newimg[i-2:i,:],newimg[i+1:i+3,:]))
                    tnmask = np.concatenate((nmask[i-2:i,:],nmask[i+1:i+3,:]))
                    tmask = (tnmask > 0.6)
                    txx,tyy = np.concatenate((xx[i-2:i,:],xx[i+1:i+3,:])),np.concatenate((yy[i-2:i,:],yy[i+1:i+3,:]))
                    
                    if np.all(np.sum(tmask,axis=1)) > 0:
                        newimg[i,:] = griddata((tyy[tmask],txx[tmask]),timg[tmask],
                                               (yy[i,:],xx[i,:]),method='linear',fill_value=0)
                        nmask[i,:] = griddata((tyy[tmask],txx[tmask]),tnmask[tmask],
                                              (yy[i,:],xx[i,:]),method='linear',fill_value=0)
                        
                else:    
                    f = RectBivariateSpline(np.concatenate((y[i-2:i],y[i+1:i+3])),x,
                                            np.concatenate((newimg[i-2:i,:],newimg[i+1:i+3,:])))
                    newimg[i,:] = f(y[i],x)
                
                #if domask:
                #    fm = RectBivariateSpline(np.concatenate((y[i-2:i],y[i+1:i+3])),x,
                #                            np.concatenate((nmask[i-2:i,:],nmask[i+1:i+3,:])))
                #    nmask[i,:] = fm(y[i],x)
                    
                #    fm = SmoothBivariateSpline(np.concatenate((y[i-2:i],y[i+1:i+3])),x,
                #                               np.concatenate((nmask[i-2:i,:],nmask[i+1:i+3,:])),
                #                               w=np.concatenate((nmask[i-2:i,:],nmask[i+1:i+3,:])))
                #    nmask[i,:] = fm(y[i],x)
        
    if domask:
        newmask = (nmask > 0.6)# value checked in histogram
        newimg[~newmask] = 0
        #fig,ax =plt.subplots(1)
        #ax.hist(nmask.reshape(-1))
        #plt.savefig(savefile+'-histmask.png')
    
    if (savefile is not None):
        #if domask: fits.writeto(savefile+'-shifted.fits',np.asarray([newimg,newmask]),clobber=True)            
        #else: fits.writeto(savefile+'-shifted.fits',newimg,clobber=True)
        fits.writeto(savefile+'-shifted.fits',newimg,clobber=True)
        
    if domask: return newimg,newmask
    else: return newimg


## -------------------------------------------------------------------------------
## --------------------------- FUNCTION to stick chip1/chip2  ---------------------
## -------------------------------------------------------------------------------
## PURPOSE:
##         Combine chips 1 and 2 into a single image
## INPUT:
##         1. image from CCD1
##         2. image from CCD2
##         3. header from CCD1
##         4. header from CCD2
## OPTIONAL INPUT:
##         - savefile: path+basic_file where to save results
##         - rot: perform rotation (boolean), def: True
##         - shift1: scalar shift (x,y) to image1, def: None
##         - shift2: scalar shift (x,y) to image2, def: None
## OUTPUT:
##         combined image
## DEPENDENCIES: scipy

def stick_chips(data1,data2,header,header2,savefile=None,ebeam=False,
                shift1=None,shift2=None,rot=True,mask1=None,mask2=None):

    if mask1 is not None: domask=True
    else: domask=False
    
    ##Check if already done
    if (savefile is not None) and (os.path.isfile(savefile+'-merged.fits')):
        print("   Found existing merged file: %s" %(savefile+'-merged.fits'))
        hdu = fits.open(savefile+'-merged.fits')
        data = hdu[0].data
        mask = (data > 0)
        #if domask:
        #    mhdu = fits.open(savefile+'-merged-mask.fits')
        #    return hdu[0].data,mhdu[0].data
        #else:
        if domask: return data,mask
        else: return data
        
    print("   Sticking chips")
    from scipy.interpolate import RectBivariateSpline,SmoothBivariateSpline,griddata
    
    ##geometry info
    #pixscale = header['HIERARCH ESO INS PIXSCALE']#0.126 arcsec/pix
    xpixbin = header['HIERARCH ESO DET WIN1 BINX'] #2
    ypixbin = header['HIERARCH ESO DET WIN1 BINY'] #2
    xpixsize = header['HIERARCH ESO DET CHIP1 PSZX']*xpixbin #15 microns (w/o bin)
    ypixsize = header['HIERARCH ESO DET CHIP1 PSZY']*ypixbin #15 microns (w/o bin)
    xgap = header['HIERARCH ESO DET CHIP1 XGAP']# 30 microns
    ygap = header['HIERARCH ESO DET CHIP1 YGAP']# 480 microns (32pix)
    ny1,nx1 = header['HIERARCH ESO DET OUT1 NY'],header['HIERARCH ESO DET OUT1 NX']# 1024,2048
    ny2,nx2 = header2['HIERARCH ESO DET OUT1 NY'],header2['HIERARCH ESO DET OUT1 NX']# 1024,2048
    xpos1,ypos1 = header['HIERARCH ESO DET OUT1 X'],header['HIERARCH ESO DET OUT1 Y']# 1,1024
    xpos2,ypos2 = header2['HIERARCH ESO DET OUT1 X'],header2['HIERARCH ESO DET OUT1 Y']#1,1
    rotgap = header['HIERARCH ESO DET CHIP1 RGAP']*np.pi/180 #0.08278deg
    naddy = np.int(ygap/ypixsize)
    naddx = np.int(xgap/xpixsize)
    
    ##pre/overscan regions
    diffy = np.size(data1[:,0]) - ny1
    if (diffy > 0):
        dat1 = data1[diffy/2:ny1+diffy/2,:]
        dat2 = data2[diffy/2:ny2+diffy/2,:]
        if domask:
            mas1 = mask1[diffy/2:ny1+diffy/2,:]
            mas2 = mask2[diffy/2:ny2+diffy/2,:]
        else:
            mas1 = (dat1 > 0)
            mas2 = (dat2 > 0)
            
    x1 = np.arange((xpos1-1)*xpixsize,(nx1+(xpos1-1))*xpixsize,xpixsize,dtype='float')
    y1 = np.arange((ypos1)*ypixsize,(ny1+(ypos1))*ypixsize,ypixsize,dtype='float')##ypos1-1
    x2 = np.arange((xpos2-1)*xpixsize,(nx2+(xpos2-1))*xpixsize,xpixsize,dtype='float')
    y2 = np.arange((ypos2-1)*ypixsize,(ny2+(ypos2-1))*ypixsize,ypixsize,dtype='float')
       
    #rotation
    if rot:
        newdat2 = np.zeros(np.shape(dat2),dtype='float')
        xx2,yy2 = np.meshgrid(x2,y2) #1d arrays into 2D matrix covering all
        newxx2 = xx2*np.cos(rotgap) - yy2*np.sin(rotgap) #- xgap
        newyy2 = xx2*np.sin(rotgap) + yy2*np.cos(rotgap) #- ygap
        
        newdat2 = griddata((newyy2[mas2],newxx2[mas2]),dat2[mas2],
                           (newyy2,newxx2),method='cubic')

        #f2 = SmoothBivariateSpline(newyy2[mas2].reshape(-1),newxx2[mas2].reshape(-1),dat2[mas2].reshape(-1))
        #f2 = RectBivariateSpline(newyy2[:,0],newxx2[0,:],dat2) #Watch out for NAN!
        #f2 = interp2d(newyy2[:,0],newxx2[0,:],dat2)
        #newdat2 = f2(y2,x2)
    else: newdat2 = dat2

    #valid indices
    ind = np.where(dat2 == 0)
    newdat2[ind] = 0
    ind2 = np.where(~np.isfinite(newdat2))
    newdat2[ind2] = 0

    ##gain
    gain1 = header['HIERARCH ESO DET OUT1 GAIN']
    gain2 = header2['HIERARCH ESO DET OUT1 GAIN']
    newdat2 = newdat2/gain2*gain1
    
    #translation (instead of doing x2-xgap, y2-ygap, do x1+xgap, y1+ygap):
    newx1, newy1 = x1+xgap, y1+ygap

    #new matrix
    data = np.zeros((ny1+ny2+naddy,nx1+naddx),dtype='float')
    data[naddy+ypos1:naddy+ny1+ypos1,naddx+xpos1-1:naddx+nx1+xpos1-1] = dat1##ypos1-1
    data[ypos2-1:ny2+ypos2-1,xpos2-1:nx2+xpos2-1] = newdat2
     
    ##mask
    if domask:
        mask = (data > 0)
    #mask = np.zeros((ny1+ny2+naddy,nx1+naddx),dtype='bool')
    #mask[:] = False
    #mask[naddy+ypos1-1:naddy+ny1+ypos1-1,naddx+xpos1-1:naddx+nx1+xpos1-1] = mas1
    #mask[ypos2-1:ny2+ypos2-1,xpos2-1:nx2+xpos2-1] = mas2

    
    ##ebeam shift
    #if shift1 is not None:
    #    data1 = shiftim(data1,shift1[0],shift1[1])
    #if shift2 is not None:
    #    data2 = shiftim(data2,shift2[0],shift2[1])
        
    #save file
    if (savefile is not None):
        fits.writeto(savefile+'-merged.fits',data,clobber=True)
        #if domask:
        #    fits.writeto(savefile+'-merged-mask.fits',mask,clobber=True)
    if domask: return data,mask
    else: return data
    
    
## ------------------------------------------------------------------------------------
## -------------------- SEPARATE CHIP1 AND CHIP2 INTO BEAMS/EBEAMS --------------------
## ------------------------------------------------------------------------------------
## PURPOSE:
##         Separate CCD1/2 images into ordinary and extraordinary beams
## INPUT:
##         1. image from CCD1
##         2. image from CCD2
## OPTIONAL INPUT:
##         - savefile1: path+basic_file where to save results for chip1
##         - savefile2: path+basic_file where to save results for chip2
##         - default: use default values when calling 'get_strips' (boolean), def: True
## OUTPUT:
##         Returns 4 images: ordinary of chip1, extraordinary chip1,
##                           ordinary chip2, extraordinary chip2

def separate_beams(data1,data2,savefile1=None,savefile2=None,default=True,dcountdy=None):
    
    ##Check if already done
    if (savefile1 is not None) & (os.path.isfile(savefile1+'-chip1-obeam.fits')):
        print("   Found existing separated file: %s" %savefile1+'-chip1-obeam.fits')
        beam1 = fits.open(savefile1+'-chip1-obeam.fits')
        #mask1 = fits.open(savefile1+'-chip1-omask.fits')
        ebeam1 = fits.open(savefile1+'-chip1-ebeam.fits')
        #emask1 = fits.open(savefile1+'-chip1-emask.fits')
        beam2 = fits.open(savefile2+'-chip2-obeam.fits')
        #mask2 = fits.open(savefile2+'-chip2-omask.fits')
        ebeam2 = fits.open(savefile2+'-chip2-ebeam.fits')
        #emask2 = fits.open(savefile2+'-chip2-emask.fits')
        return beam1[0].data,ebeam1[0].data,beam2[0].data,ebeam2[0].data
            #(mask1[0].data,emask1[0].data,mask2[0].data,emask2[0].data)

    print("   Separating beams")


    #beam,ebeam
    beam1,ebeam1 = np.zeros(np.shape(data1),dtype=float),np.zeros(np.shape(data1),dtype=float)
    mask1,emask1 = np.zeros(np.shape(data1),dtype=bool),np.zeros(np.shape(data1),dtype=bool)
    beam2,ebeam2 = np.zeros(np.shape(data2),dtype=float),np.zeros(np.shape(data2),dtype=float)
    mask2,emask2 = np.zeros(np.shape(data2),dtype=bool),np.zeros(np.shape(data2),dtype=bool)
    #beam1[:],ebeam1[:] = np.NAN,np.NAN#np.median(data1)#np.NAN
    #beam2[:],ebeam2[:] = np.NAN,np.NAN#np.median(data2)#np.NAN
    
    #separation strips
    if not default:
        strips1 = get_strips(image=data1,plotfile=savefile1+'-chip1',dycut=dcountdy)
        strips2 = get_strips(image=data2,plotfile=savefile2+'-chip2',dycut=dcountdy)
    else:
        strips1,strips2 = get_strips(plotfile=savefile1)

    #xmin,xmax (by eye)
    xmin1,xmax1,xmin2,xmax2 = 187,1861,188,1864 
        
    #chip1
    for s in range(0,len(strips1)-1):
        if s%2 == 0:
            beam1[strips1[s,1]+1:strips1[s+1,0],xmin1:xmax1+1] = data1[strips1[s,1]+1:strips1[s+1,0],xmin1:xmax1+1]
            mask1[strips1[s,1]+1:strips1[s+1,0],xmin1:xmax1+1] = True
        else:
            ebeam1[strips1[s,1]+1:strips1[s+1,0],xmin1:xmax1+1] = data1[strips1[s,1]+1:strips1[s+1,0],xmin1:xmax1+1]
            emask1[strips1[s,1]+1:strips1[s+1,0],xmin1:xmax1+1] =  True
            
    #chip2
    for s in range(0,len(strips2)-1):
        if s%2 != 0:
            beam2[strips2[s,1]+1:strips2[s+1,0],xmin2:xmax2+1] = data2[strips2[s,1]+1:strips2[s+1,0],xmin2:xmax2+1]
            mask2[strips2[s,1]+1:strips2[s+1,0],xmin2:xmax2+1] = True
        else:
            ebeam2[strips2[s,1]+1:strips2[s+1,0],xmin2:xmax2+1] = data2[strips2[s,1]+1:strips2[s+1,0],xmin2:xmax2+1]
            emask2[strips2[s,1]+1:strips2[s+1,0],xmin2:xmax2+1] = True
            
    #old:
    #    ostrips1,estrips1,ostrips2,estrips2  = get_strips()
    #    for strip in ostrips1: beam1[strip[0]:strip[1],:] = data1[strip[0]:strip[1],:]
    #    for estrip in estrips1: ebeam1[estrip[0]:estrip[1],:] = data1[estrip[0]:estrip[1],:]
    #    #chip2
    #    for strip in ostrips2: beam2[strip[0]:strip[1],:] = data2[strip[0]:strip[1],:]
    #    for estrip in estrips2: ebeam2[estrip[0]:estrip[1],:] = data2[estrip[0]:estrip[1],:]

    
    if savefile1 is not None:
        fits.writeto(savefile1+'-chip1-obeam.fits',beam1,clobber=True)
        #fits.writeto(savefile1+'-chip1-omask.fits',mask1,clobber=True)
        fits.writeto(savefile1+'-chip1-ebeam.fits',ebeam1,clobber=True)
        #fits.writeto(savefile1+'-chip1-emask.fits',emask1,clobber=True)
      
    if savefile2 is not None:
        fits.writeto(savefile2+'-chip2-obeam.fits',beam2,clobber=True)
        #fits.writeto(savefile2+'-chip2-omask.fits',mask2,clobber=True)
        fits.writeto(savefile2+'-chip2-ebeam.fits',ebeam2,clobber=True)
        #fits.writeto(savefile2+'-chip2-emask.fits',emask2,clobber=True)
    
    return beam1,ebeam1,beam2,ebeam2#),(mask1,emask1,mask2,emask2)

## ------------------------------------------------------------------------------------
## -----------READ_REFERENCE: Fct to read template images without pol and do astrometry
## PURPOSE: Read file from headdata/template.dat
## INPUT: path+file
## OUTPUT: ref dict

def read_reference(datadir,reffile,rawdir):

    refname = '_refimage'
    ## template file
    if not os.path.isfile(reffile):
        print(" WARNING: No template.dat file with reference images for astrometry")
        refinfo = None
    else:    
        print(" Found template file %s: will use that" %reffile)
        
        # read like polfiles
        reffiles = np.loadtxt(reffile,
            dtype={'names':('file','galaxy','target','ra','dec',
                            'filter','angle','exptime','mjd','chip','moon'),
                   'formats':('O','O','O','f','f','O','f','f','f8','O','f')})
        reffiles = reffiles[np.argsort(reffiles['mjd'])]
        filters = np.unique(reffiles['filter'])
        refinfo = np.zeros(len(filters),dtype={'names':('file','filter','ra','dec'),'formats':('O','O','f','f')})

        for i,filt in enumerate(filters):
            freffiles1 = reffiles[(reffiles['filter'] == filt) & (reffiles['chip'] == 'CHIP1')]
            freffiles2 = reffiles[(reffiles['filter'] == filt) & (reffiles['chip'] == 'CHIP2')]
            nffiles = len(freffiles1)
            if len(freffiles2) != nffiles: 
                print("Error in template.dat file")
                raise

            # already done (tempo?)
            refinfo['file'][i] = datadir+filt+refname+'-merged.fits'
            refinfo['filter'][i] = filt
            refinfo['ra'][i] = freffiles1['ra'][0]
            refinfo['dec'][i] = freffiles1['dec'][0]         

            if os.path.isfile(datadir+filt+refname+'-merged.fits'): continue

            for f in range(0,nffiles):
                if nffiles > 1: extname = '_'+np.str(f)
                else: extname='' 
                
                #Stick chips
                h1,d1 = read_fits(rawdir+freffiles1['file'][f])
                h2,d2 = read_fits(rawdir+freffiles2['file'][f])
                ref_data12 = stick_chips(d1,d2,h1,h2,savefile=datadir+filt+refname+extname)

                #Astrometry
                ra,dec = freffiles1["ra"][f],freffiles1["dec"][f]
                ref_head,ref_data = astrometry(ra,dec,datadir+filt+refname+extname+'-merged.fits',
                                       outdir=datadir,outfile=filt+refname+extname+'-merged-astrom.fits')

            #Align several files with same filter??

            #Galfit

        #refinfo = {}
        #with open(reffile) as f:
        #    content = f.readlines()
        #    refinfo['file1'] = rawdir+content[0].split()[0]
        #    refinfo['file2'] = rawdir+content[1].split()[0]
        #refinfo['head1'],refinfo['data1'] = read_fits(refinfo['file1'])
        #refinfo['head2'],refinfo['data2'] = read_fits(refinfo['file2'])
        
    return refinfo


## ------------------------------------------------------------------------------------
## -----------READ_OFFSET: Fct to read offsets
## PURPOSE: Read offset file from headdata/observation.dat
## INPUT: path+file
## OUTPUT: offsetinfo dict

def read_offset(offfile):

    #before
    #try:
    #    offsetinfo = np.loadtxt(datadir+'observation.dat',dtype='str')
    #    print("Found offset file: will use that")
    #except:
    #    offsetinfo = None
    #    print("No file with offset info: assuming no offsets were taken")
    
    if not os.path.isfile(offfile):
        print("No file with offset info: finding them automatically")
        offsetinfo = None
    else:    
        print("Found offset file %s: will use that" %offfile)
        offsetinfo = {}
        with open(offfile) as f:
            for line in f:
                if line[0] == '#': continue
                line = line.strip()
                arrsplit = line.split()
                nsp = len(arrsplit)
                offsetinfo[arrsplit[0]] = np.asarray(arrsplit[1:nsp],dtype=np.float32)
    return offsetinfo

## ------------------------------------------------------------------------------------
## -----------GET_OFFSET: Fct to get multiple offset/iteration info and folders ---------------------
##                    (& before: to prepare data for esoreflex changing headers)
## ------------------------------------------------------------------------------------
## PURPOSE: Get offset info file from i) observation.dat file or ii) direclty from the files
##          Sometimes there are no offsets but several iteration of same target/filter/angle.
##          This is also returned
## INPUT:  offsetinfo     Either info from target/headata/observation.dat or none
##                         'observation.dat' has Filter and RA/DEC of each offset
##         files           File names within target folder
##         filter          Name of filter under consideration
## OPTIONAL INPUT:
##         dir              where to write auxiliary files
## OUTPUT: offset file with: 'ra' and 'dec' of each offset
##         and type 'off' (for offset) or 'it' (for iteration)

def get_offset(offsetinfo,opolfiles,thisfilter,polangles,dir=dir):
    
    if 'offit' not in opolfiles.dtype.names:
        from numpy.lib import recfunctions as rfn
        polfiles = rfn.append_fields(opolfiles,'offit',np.full(len(opolfiles),-1,dtype=int))
    else: polfiles = opolfiles
        
    tol = 1e-8 #tolerance for RA/DEC comparison
    
    ## multiple offsets
    if offsetinfo is not None:
        if thisfilter not in offsetinfo.keys():
            print("ERROR in get_offset: filter %s not found" %thisfilter)
        noffset = np.int(offsetinfo[thisfilter].shape[0]/2)
        offset = np.zeros(noffset,dtype=[('ra',float),('dec',float),('type',object)])
        offset['ra'] = offsetinfo[thisfilter][0::2]
        offset['dec'] = offsetinfo[thisfilter][1::2]
        offset['type'] = 'off'
        radec = [[o['ra'],o['dec']] for o in offset]
        uradec,uind,ind,rep = np.unique(radec,axis=0,return_index=True,return_inverse=True,
                                        return_counts=True)
        nrep = rep[ind]
           
    ## find offsets ourselves OR iterations with same offset
    else: 
        fpolfiles = polfiles[polfiles['filter'] == thisfilter]
        radec = [[p['ra'],p['dec']] for p in fpolfiles]
        uniqradec = np.unique(radec,axis=0)#,return_index=True,return_inverse=True,return_counts=True)
        nuniqradec = len(uniqradec)
        nrep = np.ones(nuniqradec)
        if nuniqradec > 1: #offset ourselves
            noffset = nuniqradec
            uniqradec = np.reshape(uniqradec,(noffset,2))
            offset = np.zeros(noffset,dtype=[('ra',float),('dec',float),('type',object)])
            offset['ra'],offset['dec'] = uniqradec[:,0], uniqradec[:,1]
            offset['type'] = 'off'
        
            ## write observation file
            obsfile = open(dir+'observation_'+thisfilter+'.dat','w+')
            obsfile.write("#Filter RA1 DEC1(pos1) RA2 DEC2(pos2)....\n")
            linestr = ''
            for i in range(0,len(uniqradec)):
                linestr += ("%.8f %.8f " %(uniqradec[i,0],uniqradec[i,1]))
            obsfile.write(linestr)
            obsfile.close()
            

        else: #iterations
            tfiles = polfiles['file'][(polfiles['angle'] == 67.5) &
                                      (polfiles['filter'] == thisfilter) &
                                      (polfiles['chip'] == 'CHIP1')]
            noffset = len(tfiles)
            offset = np.zeros(noffset,dtype=[('ra',float),('dec',float),('type',object)])
            offset['ra'],offset['dec'] = polfiles['ra'][0],polfiles['dec'][0]
            offset['type'] = 'it'

    noffset = len(offset)
    if offset['type'][0] == 'off':
        for i in range(0,noffset):
            ffiles1 = polfiles[(polfiles['filter'] == thisfilter) & 
                               (polfiles['chip'] == 'CHIP1') & 
                               (np.isclose(polfiles['ra'],offset['ra'][i],rtol=tol)) &
                               (np.isclose(polfiles['dec'],offset['dec'][i],rtol=tol))]
            ffiles2 = polfiles[(polfiles['filter'] == thisfilter) & 
                               (polfiles['chip'] == 'CHIP2') & 
                               (np.isclose(polfiles['ra'],offset['ra'][i],rtol=tol)) &
                               (np.isclose(polfiles['dec'],offset['dec'][i],rtol=tol))]

                        
            ##check for errors
            if ((offset['type'][i] == 'off') & (len(ffiles1) != len(polangles))):
                medexptime,stdexptime = np.median(ffiles1['exptime']),np.std(ffiles1['exptime'])
                if stdexptime > 0.5:
                    print(" Some files have large exposure time divergence and shouldn't be considered!")
                    print(ffiles1['file'][(np.abs(ffiles1['exptime']-medexptime) > 0.5)])
                    pdb.set_trace()
                else:
                    if (nrep[i] > 1) & (len(ffiles1) == nrep[i]*len(polangles)):
                        print(" Some files have repeated offset!")
                        ti = np.argwhere(ind == ind[i]).reshape(-1)
                        ni = np.sum(ti < i)
                     
                        ## Assign offset
                        okvals = np.argwhere((polfiles['filter'] == thisfilter) & 
                                  (np.isclose(polfiles['ra'],offset['ra'][i],rtol=tol)) &
                                  (np.isclose(polfiles['dec'],offset['dec'][i],rtol=tol))).reshape(-1)
                        polfiles['offit'][okvals[ni*2*len(polangles):(ni+1)*2*len(polangles)]] = i
             
                    else:
                        if (len(ffiles1) > len(polangles)):
                            print(" Warning: more files than expected with similar exposure time and position")
                            print(ffiles1)
                            docomb = raw_input(" Do you want to continue by doing a median image from all? (Y/N) ")
                            if docomb.lower() == 'y':
                                for tang in polangles:
                                    affiles1 = dir+ffiles1[(ffiles1['angle'] == tang)]['file']
                                    affiles2 = dir+ffiles2[(ffiles2['angle'] == tang)]['file']
                                    pdb.set_trace()
                                    if len(affiles1) > 0:
                                        combine_images(affiles1,affiles1[0].replace('.fits','_med.fits'))
                                        combine_images(affiles2,affiles2[0].replace('.fits','_med.fits'))
                                print(" You should now change your input filemap.dat and re-rerun. ")
                                pdb.set_trace()
                            else:    
                                pdb.set_trace()
                        else:
                            print(" Warning: less files than expected with similar exposure time and position")
                            print(ffiles1)
                            print(" You should probably get rid of these. ")
                            pdb.set_trace()
            else:
                ## Assign offset
                polfiles['offit'][(polfiles['filter'] == thisfilter) & 
                                  (np.isclose(polfiles['ra'],offset['ra'][i],rtol=tol)) &
                                  (np.isclose(polfiles['dec'],offset['dec'][i],rtol=tol))] = i

    elif offset['type'][0] == 'it':

        ffiles1 = polfiles[(polfiles['filter'] == thisfilter) & 
                           (polfiles['chip'] == 'CHIP1')]

        if (len(ffiles1) != len(polangles)*len(offset)):
            print(" More exposure times than iterations!")
            pdb.set_trace()
        
        ##organize per exptime
        exptime = np.round(polfiles['exptime'],decimals=1)
        fexptime = np.round(ffiles1['exptime'],decimals=1)
        uniqexptime = np.unique(fexptime)
        if len(uniqexptime) > noffset:
            print(" More exposure times than iterations!")
            pdb.set_trace()
        it = 0
        for t in uniqexptime:
            tfiles = polfiles[(polfiles['filter'] == thisfilter) &
                              (polfiles['chip'] == 'CHIP1') &
                              (exptime == t)]
            if np.mod(len(tfiles),len(polangles)) != 0:
                print(" Some files have exposure time divergence and shouldn't be considered!")
                print(tfiles)
                pdb.set_trace()
            intrait = len(tfiles)/len(polangles)
            for p in polangles:
                ttfiles = polfiles[(polfiles['filter'] == thisfilter) & (polfiles['angle'] == p) &
                                    (exptime == t)]
                if len(ttfiles) != 2*intrait:
                   print(" Some sort of error")
                   print(ttfiles)
                   pdb.set_trace()
                polfiles['offit'][(polfiles['filter'] == thisfilter) & (polfiles['angle'] == p) &
                                  (polfiles['chip'] == 'CHIP1') & (exptime == t)] = it+np.arange(0,intrait)
                polfiles['offit'][(polfiles['filter'] == thisfilter) & (polfiles['angle'] == p) &
                                  (polfiles['chip'] == 'CHIP2') & (exptime == t)] = it+np.arange(0,intrait)
            it+=intrait
                
        if it != noffset:
            print(" Some sort of error!")
            pdb.set_trace()

    return offset,polfiles

## ------------------------------------------------------------------------------------
## -----------PREPARE_FOLDERS: Fct to arrange data and folders ---------------------
##                    (& before: to prepare data for esoreflex changing headers)
## ------------------------------------------------------------------------------------
## PURPOSE:
##       This routine organizes raw data downloaded from ESO website
##       E.g. given observation from Vela1 on a MJD night '57811', then the rawdata should be in:
##            'home/Vela1/57811/rawdata/'
##       This will create a new directory with only science data in:
##            'home/Vela1/57811/headdata/'
##       At the same time it will put all bias frames in:
##            'home/bias/57811/'
##       At the same time it will put all flat frames in:
##            'home/flat/57811/'
##       Additionnally it creates following useful ascii files with info on files:
##            'home/Vela1/57811/headdata/filemap.dat'
##            'home/bias/57811/biasmap.dat'
##            'home/flat/57811/flatmap.dat'
##       NOTE: Originally it used to organize it into ESO REFLEX standard
##        (changing some things in the header so that REFLEX would assume it's photometry)
## INPUT:
##       list of file names to be arranged
## OPTIONAL INPUT:
##       - indir: folder where rawdata is found, e.g. 'home/Vela1/57811/rawdata/'
##       - outdir: folder where organized data will be found, e.g. 'home/Vela1/57811/headdata/'
##       - target: Name of target, e.g. 'Vela1'. If this set and not 'indir/outdir', then
##                 indir = home+"/crisp/FORS2-POL/"+target+"/rawdata/"
##                 outdir = home+"/crisp/FORS2-POL/"+target+"/headdata/"
##       - nocopy: Only creates filemap.dat but does not mv files around (def: False)
## OUTPUT:
#        This routine does not return anything but organizes files into folders
##        and creates map files for later use


### FOR FORS2
def prepare_folder(files, indir = None, outdir= None, target='NGC-3351', nocopy=False,
                   location='paranal'):

    from moon import moon_pol
    
    #dirs
    if indir == None:
        indir = home+"/crisp/FORS2-POL/"+target+"/rawdata/"
    if outdir == None:
        outdir = home+"/crisp/FORS2-POL/"+target+"/headdata/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if os.path.isfile(outdir+'filemap.dat'):
        return
    biasdir = home+"/crisp/FORS2-POL/bias/"
    flatdir = home+"/crisp/FORS2-POL/flat/"
    stddir = home+"/crisp/FORS2-POL/"
    
    #map of files
    filemap = np.zeros(np.size(files), \
                   dtype=[('file','S40'),('object','S20'),('target','S20'),\
                          ('ra','S20'),('dec','S20'),\
                          ('filter','S10'),('angle','S8'),('exptime','S15'),\
                          ('mjdobs','S15'),('extname','S15'),('moon','S15')])
    
    #loop files
    i = -1
    for f in files:

        print("  File: %s" %f)
        i += 1
        filemap[i]['file'] = f

        #initialize
        filemap[i]['filter'] = '--'
        filemap[i]['angle'] = 'NaN'
        filemap[i]['extname'] = '--'
        filemap[i]['target'] = '--'
        filemap[i]['object'] = '--'
        filemap[i]['mjdobs'] = '--'
        filemap[i]['exptime'] = '--'
        filemap[i]['moon'] = '--'
        filemap[i]['ra'] = '--'; filemap[i]['dec'] = '--'
        
        #make sure it's fits
        if not f.endswith('.fits'):
            print("   --Not fits file!")
            continue

        #make sure it's not already done: comment cause of filemap
        #if os.path.isfile(outdir+f):
        #    print("   --Already done!")
        #    continue
    
        #open fits file
        hdu = fits.open(indir+f)
        header = hdu[0].header
    
        filemap[i]['object'] = header['OBJECT']
        filemap[i]['mjdobs'] = header['MJD-OBS']
        filemap[i]['exptime'] = header['EXPTIME']
        if 'RA' in header.keys():
            filemap[i]['ra'] = header['RA']; filemap[i]['dec'] = header['DEC']
        print(header['OBJECT'])
        if 'EXTNAME' in header.keys():
            filemap[i]['extname'] = header['EXTNAME']
        if 'ESO INS FILT1 NAME' in header.keys():
            filemap[i]['filter'] = header['HIERARCH ESO INS FILT1 NAME']
    
        #IPOL instead of IMG: change target name
        if ('FORS2' in indir):
            if (header['HIERARCH ESO INS MODE'] == 'IPOL'):
                header['HIERARCH ESO INS MODE'] = 'IMG'
                print("   --Changed INS MODE")
                if 'ESO INS RETA2 ROT' in header.keys():
                    targname = header['HIERARCH ESO OBS TARG NAME']
                    filemap[i]['target']=targname.replace(' ','-')
                    angle = str(header['HIERARCH ESO INS RETA2 ROT'])
                    header['OBJECT'] = targname #+ '--' + angle #(FOR REFLEX!!)
                    header['HIERARCH ESO OBS NAME'] = targname #+ '--' + angle#(FOR REFLEX!!)
                    header['HIERARCH ESO OBS TARG NAME'] = targname #+'--'+angle#(FOR REFLEX!!)
                    filemap[i]['angle'] = angle
                    print("   --Changed OBJECT/TARG NAME/OBS NAME")
            
                if header['ESO DPR TECH'] == 'POLARIMETRY':
                    header['HIERARCH ESO DPR TECH'] = 'IMAGE'
                    print("   --Changed DPR TECH")


                #GET MOON ILLUMINATION
                moon,mang,mQ,mU =\
                       	moon_pol(header['DATE-OBS'],header['RA'],header['DEC'],
                                 radecsys=header['RADECSYS'],location='paranal',
                                 moon_ra=header['HIERARCH ESO TEL MOON RA'],
                                 moon_dec=header['HIERARCH ESO TEL MOON DEC'])
                filemap[i]['moon'] = np.str(round(moon,6))
                            
        #if ('CAFOS' in indir):
        #    filemap[i]['target'] = header['OBJECT'].replace(' ','_')
        #    if header['IMAGETYP'] == 'bias':
        #        targname = 'BIAS'
        #    elif header['IMAGETYP'] == 'flat':
        #        targname = 'FLAT'
        #    else: targname = header['OBJECT'].replace(' ','_')
        #    filemap[i]['object'] = targname
        #    if header['INSPOFPI'] != 'FREE':
        #        filemap[i]['angle'] = str(round(header['INSPOROT'],1))
        #    filemap[i]['filter'] = header['INSFLNAM']
                
            
        #Write new fits file
        if target in filemap[i]['target']:
            hdu.writeto(outdir+f, output_verify = 'ignore')
            hdu.close()

    #Order per MJD
    sort = np.argsort(filemap['mjdobs'])
    filemap = filemap[sort]
            
    #Save full map
    np.savetxt(indir+'fullmap.dat',(filemap),fmt='%20s')     

    #Bias map (to new dir)
    biasmap = filemap[(filemap['object'] == "BIAS")]
    for bmap in biasmap:
        if not os.path.exists(biasdir+bmap['mjdobs'][0:5]):
            os.mkdir(biasdir+bmap['mjdobs'][0:5])
        os.system("cp "+indir+bmap['file']+" "+biasdir+bmap['mjdobs'][0:5])
    if len(biasmap) > 0:
        print("   %%-- BIAS: %i files -- %%" %(len(biasmap)))
        np.savetxt(biasdir+'biasmap.'+target+'.dat',(biasmap),fmt='%20s')
        
    #Flat map (to new dir)
    flatmap = filemap[(filemap['object'] == "FLAT") | (filemap['object'] == "FLAT,SKY")]
    for fmap in flatmap:
        if not os.path.exists(flatdir+fmap['mjdobs'][0:5]):
            os.mkdir(flatdir+fmap['mjdobs'][0:5])
        os.system("cp "+indir+fmap['file']+" "+flatdir+fmap['mjdobs'][0:5])
    if len(flatmap) > 0:
        print("   %%-- FLATS: %i files -- %%" %(len(flatmap))) 
        np.savetxt(flatdir+'flatmap.'+target+'.dat',(flatmap),fmt='%20s')

    #STD map (to new dir)
    stdmap = filemap[(filemap['object'] == "STD")]
    stdstars = np.unique(stdmap['target'])
    for s in range(0,len(stdstars)):
        if not os.path.exists(stddir+stdstars[s]):
            os.mkdir(stddir+stdstars[s])
        tstdmap = stdmap[(stdmap['target'] == stdstars[s])]
        print("   %%-- STD: %s with %i files-- %%" %(stdstars[s],len(tstdmap)))
        for smap in tstdmap:
            if not os.path.exists(stddir+stdstars[s]+'/'+smap['mjdobs'][0:5]):
                os.mkdir(stddir+stdstars[s]+'/'+smap['mjdobs'][0:5])
                os.mkdir(stddir+stdstars[s]+'/'+smap['mjdobs'][0:5]+'/headdata/')
            os.system("cp "+indir+smap['file']+" "+stddir+stdstars[s]+'/'+smap['mjdobs'][0:5]+'/headdata')
        np.savetxt(stddir+stdstars[s]+'/stdmap-'+target+'.dat',(tstdmap),fmt='%20s')

    
    #OBJ in this dir
    nfilemap = filemap[(filemap['object'] != "BIAS") & (filemap['object'] != "FLAT") &
                      (filemap['object'] != "FLAT,SKY") & (filemap['object'] != "STD") &
                      (filemap['object'] != "--")]
    #nfilemap = np.asarray([fmap for fmap in filemap if target in fmap['target']])
    
    
    #write info map
    np.savetxt(outdir+'filemap.dat',(nfilemap),fmt='%20s')#,header='   '.join(filemap.dtype.names))
    print("   %%-- %s: %i files -- %%" %(target,len(nfilemap)))

### FOR CAFOS
# YES: OBJECT, MJD-OBS
# NOT: EXTNAME
# filter: 'INSFLNAM'
# pol?: 'INSPOFPI' - Wollaston, FREE
# angle: 'INSPOROT' 
# type: 'IMAGETYP' -science,bias


## ------------------------------------------------------------------------------
## ---------------- COMBINE_OFFSETS --------------------------------------------
## ------------------------------------------------------------------------------
## PURPOSE: If several offsets for extended data of the same filter in the same night,
##          then it combines all images (Q,U) to then calculate P,ang
## INPUT:
##       The files are read from the following information. 
##       1. offset info
##       2. filter string
##       3. output path
##       4. additional filename info
## OUTPUT:
##       The output is not returned but saved on file.
def combine_offsets(offset,filt,outdir,savefile,center=None):

    noffset = len(offset)
  
    ## Q
    filename = [outdir+filt+'-'+offset['type'][i]+np.str(i)+savefile+'-QStokes.fits' for i in range(0,noffset)]
    Q,erQ = combine_images(filename,outdir+filt+'-alloff'+savefile+'-QStokes.fits',align=True,
                           erstd=True,method='absmedian',#'weighted',
                           ra=offset['ra'],dec=offset['dec'])
        
    ## U
    filename = [outdir+filt+'-'+offset['type'][i]+np.str(i)+savefile+'-UStokes.fits' for i in range(0,noffset)]
    U,erU = combine_images(filename,outdir+filt+'-alloff'+savefile+'-UStokes.fits',align=True,
                           method='absmedian',#'weighted',
                           ra=offset['ra'],dec=offset['dec'])

    ## Pol
    pol,ang = QUpolarization(Q,U,filt,savefile=outdir+filt+'-alloff'+savefile,errQ=erQ,errU=erU)
    ##plot variation of median vs offset?
    
    ## Errors
    h0,erpol = read_fits(outdir+filt+'-alloff'+savefile+'-erpol0.fits')
    h0,erangle = read_fits(outdir+filt+'-alloff'+savefile+'-erangle0.fits')

    ## Plot
    Qp,Up,Qm,Um = plotstokes(Q,U,savefile=outdir+filt+'-alloff'+savefile,
                             scatter=('bin' in savefile),center=center)#x=x,y=y,
    plotpol(pol,ang,center=center,savefile=outdir+filt+'-alloff'+savefile,
            erpol=erpol,erangle=erangle)
    

## ------------------------------------------------------------------------------
## ---------------- SUM_OFFSETS --------------------------------------------
## ------------------------------------------------------------------------------
## PURPOSE: If several offsets for field star data of the same filter/HWP-angle in the same night,
##          then it adds all stars to do pol/angle values
## INPUT:
##       The photometry files are read from the following information. 
##       1. offset info
##       2. filter string
##       3. output path
## OPTIONAL INPUT
##       binpts:     Box size to bin points (def: 40)
##       signois:    minimal S/N to accept points
##       For file load:
##       - cosmic   True/False
##       - posfree  True/False
## OUTPUT:
##       The output is not returned but saved on file.

def sum_offsets(offset,filt,outdir,binpts=20,cosmic=False,posfree=False,center=None,
                signois=-1,ap=False,xname='',sigmaclip=2,parfit=False,bias=True,method='P14'):

    psf='psf'
    if ap: psf='ap' 
    noffset = len(offset)
    cos = '-cosmic' if cosmic else ''
    pos = '-posfree' if posfree else ''
    print("   -- Including all offsets for field star plots")

    ## load all polarization & x/y files
    for i in range(0,noffset):
        polfile = outdir+filt+cos+'-'+offset['type'][i]+np.str(i)
        Q = ((fits.open(polfile+'-'+psf+'field'+xname+'-QStokes.fits'))[0].data)[0]
        U = ((fits.open(polfile+'-'+psf+'field'+xname+'-UStokes.fits'))[0].data)[0]
        pol = (fits.open(polfile+'-'+psf+'field'+xname+'-pol.fits'))[0].data
        erpol = (fits.open(polfile+'-'+psf+'field'+xname+'-erpol.fits'))[0].data
        
        angle = (fits.open(polfile+'-'+psf+'field'+xname+'-angle.fits'))[0].data
        xyfile = outdir+filt+cos+'-'+offset['type'][i]+np.str(i)+'-'+psf+'field'+pos+'-flux.dat'
        ores = ascii.read(xyfile)
        nstars = len(ores)        
        x,y = ores['xfit'].reshape(nstars,1),ores['yfit'].reshape(nstars,1)
        
        if i > 0:
            allQ = np.concatenate((allQ,Q),axis=0)
            allU = np.concatenate((allU,U),axis=0)
            allpol = np.concatenate((allpol,pol),axis=0)
            allerpol = np.concatenate((allerpol,erpol),axis=0)
            allangle = np.concatenate((allangle,angle),axis=0)
            allx = np.concatenate((allx,x),axis=0)
            ally = np.concatenate((ally,y),axis=0)
        else:
            allpol,allerpol,allangle = pol,erpol,angle
            allQ,allU,allx,ally = Q,U,x,y       

    ##signois
    ##force signois
    #sn = allpol.reshape(-1)/allerpol.reshape(-1)
    #signois = np.percentile(sn,30)
    #print(" signois: " %signois)
    #import pdb;pdb.set_trace()
    if signois > 0:
        sn = allpol.reshape(-1)/allerpol.reshape(-1)
        indsn = np.argwhere(sn >= signois).reshape(-1)
        allx,ally = allx[indsn],ally[indsn]
        allpol,allerpol,allangle = allpol[indsn],allerpol[indsn],allangle[indsn]

    indsn = np.argwhere(allpol.reshape(-1) < 0.02).reshape(-1)
    allx,ally = allx[indsn],ally[indsn]
    allpol,allerpol,allangle = allpol[indsn],allerpol[indsn],allangle[indsn]
    if bias: allpol = polbias(allpol,allerpol,method=method)
    
    ##Plot polarization of field stars
    savefile = outdir+filt+cos+'-alloff'
    xyplotpol(allx,ally,allpol,allangle,center=center,savefile=savefile+'-'+psf+'field'+xname)

    ##Bin stars QU
    allQmap,erallQmap = bin_points(allx,ally,allQ,fullbin=True,center=center,sigmaclip=sigmaclip,
                                   savefile=savefile+'-'+psf+'field-binQ'+xname,radpix=binpts)
    allUmap,erallUmap = bin_points(allx,ally,allU,fullbin=True,center=center,sigmaclip=sigmaclip,
                                   savefile=savefile+'-'+psf+'field-binU'+xname,radpix=binpts)
    plotstokes(allQmap,allUmap,savefile=savefile+'-'+psf+'field-bin'+xname,
               parfit=parfit,scatter=True,center=center)
    
    ## Bin stars POL and plot
    allpolmap,erallpolmap = bin_points(allx,ally,allpol,fullbin=True,center=center,sigmaclip=sigmaclip,
                                       savefile=savefile+'-'+psf+'field-binpol'+xname,radpix=binpts)
    allanglemap,erallanglemap = bin_points(allx,ally,allangle,fullbin=True,center=center,sigmaclip=sigmaclip,
                                           savefile=savefile+'-'+psf+'field-binangle'+xname,radpix=binpts)
    plotpol(allpolmap,allanglemap,erpol=erallpolmap,erangle=erallanglemap,polrange=[0.006,0.011],
            step=int(binpts/2),center=center,savefile=savefile+'-'+psf+'field-bin'+xname)       
    radius_dependence(allpolmap,allanglemap,savefile+'-'+psf+'field-bin'+xname,radfit=True,scatter=True,
                      parfit=parfit,erpol=erallpolmap,filt=filt)#,center=center)        
       
## ------------------------------------------------------------------------------
## ---------------- AVERAGE PHOT ITERATIONS ------------------------------
## ------------------------------------------------------------------------------
## PURPOSE: If several photometrical iterations of the same filter/HWP-angle 
##          of a science point source in the same night,
##          then it does a weighted average of all pol/angle values
## INPUT:
##       The photometry files are read from the following information. 
##       1. photometry type (string): either 'psf' or 'aper'
##       2. Iteration array, e.g: [0,1]
##       3. angles where to do average, e.g. [0,22.5,45.0,67.5]
##       4. output path
##       5. filter string
##       6. string of cosmic ray or not, e.g. '-cosmic'
## OUTPUT:
##       The output is not returned but saved on file.

def average_iterations(ph,iters,angles,outdir,fname,cname):

    niters,nangles = len(iters),len(angles)
    totpol,toterpol = np.zeros(niters,dtype=float),np.zeros(niters,dtype=float)
    totangle,toterangle = np.zeros(niters,dtype=float),np.zeros(niters,dtype=float)
    totQ,toterQ = np.zeros(niters,dtype=float),np.zeros(niters,dtype=float)
    totU,toterU = np.zeros(niters,dtype=float),np.zeros(niters,dtype=float)
    totF,toterF = np.zeros((niters,nangles),dtype=float),np.zeros((niters,nangles),dtype=float)
    
    for i in range(0,niters):
        tname = fname+'-'+iters['type'][i]+np.str(i)+cname
        pol,angle,Q,U,F,erpol,erangle,erQ,erU,erF = np.load(outdir+tname+"_"+ph+"phot.npy")
        totpol[i],toterpol[i] = pol,erpol
        totangle[i],toterangle[i] = angle,erangle
        totQ[i],toterQ[i] = Q,erQ
        totU[i],toterU[i] = U,erU
        totF[i,:],toterF[i,:] = F[:,0,0],erF[:,0,0]

    pol = np.average(totpol,weights=1/toterpol**2).reshape(1,1)
    erpol = 1/np.sqrt(np.sum(1/toterpol**4)).reshape(1,1)
    angle = np.average(totangle,weights=1/toterangle**2).reshape(1,1)
    erangle = 1/np.sqrt(np.sum(1/toterangle**4)).reshape(1,1)
    U = np.average(totU,weights=1/toterU**2).reshape(1,1)
    erU = 1/np.sqrt(np.sum(1/toterU**4)).reshape(1,1)
    Q = np.average(totQ,weights=1/toterQ**2).reshape(1,1)
    erQ = 1/np.sqrt(np.sum(1/toterQ**4)).reshape(1,1)
    F = np.average(totF,axis=0,weights=1/toterF**2).reshape(nangles,1,1)
    erF = 1/np.sqrt(np.sum(1/toterF**4,axis=0)).reshape(nangles,1,1)
    np.save(outdir+fname+cname+'_'+ph+'phot.npy',(pol,angle,Q,U,F,erpol,erangle,erQ,erU,erF))

    

## ---------------------------------------------------------------------------
## ---------------- ANALYZE STRIPS VS ANGLE----------------------------------
## ---------------------------------------------------------------------------
## PURPOSE:
##       Analyse strip positions with respect to HWP angle
##       Strip positions are the ones found with function 'get_strips'
## INPUT:
##       1. filter, e.g. 'R_SPECIAL'
##       2. angles to consider, e.g [0,22.5,45.0,67.5]
##       3. output directory
##       4. object type, 'GAL' or 'STD'
##       5. offset/iteration array, eg. [0,1]
## OUTPUT:
##       It does not return anything but creates a plot of strip positions vs angle
##       ('...strips-ang.png') and ASCII file ('...strips-ang.dat').

def analyse_angstrips(filt,angles,odir,obj,its):

    #its
    nits = len(its)
    
    #strips
    n1,n2 = 12,9
    labels,lst,mul = np.zeros(n1+n2,dtype=object),np.zeros(n1+n2,dtype=object),np.zeros(n1+n2,dtype=int)
    labels[0:n1],lst[0:n1],mul[0:n1] = 'chip1-','-',1.0
    labels[n1:n1+n2],lst[n1:n1+n2],mul[n1:n1+n2] = 'chip2-','--',1.0

    import matplotlib.cm as cm
    colors = cm.gist_rainbow(np.linspace(0,1,n1+n2))
    
    #read them
    N = len(angles)
    offsets = np.zeros((nits,N,n1+n2),dtype=float)
    strips = np.zeros((nits,N,n1+n2,2),dtype=float)
    midstrips = np.zeros((nits,N,n1+n2),dtype=float)
    for i in range(0,len(its)):
        tname=''
        if len(its) > 1: tname = '-'+its['type'][i]+np.str(i)

        for p,angle in enumerate(angles):
                fname = odir+filt+'-ang'+str(angle)+tname
                strips1 = np.loadtxt(fname+'-chip1-strips.dat')
                strips2 = np.loadtxt(fname+'-chip2-strips.dat')
                strips[i,p,0:n1,:] = strips1
                strips[i,p,n1:n1+n2,:] = strips2
                midstrips[i,p,0:n1] = [0.5*(strip[0]+strip[1]) for strip in strips1]
                offsets[i,p,0:n1] = [strip[1]-strip[0] for strip in strips1]
                midstrips[i,p,n1:n1+n2] = [0.5*(strip[0]+strip[1]) for strip in strips2]
                offsets[i,p,n1:n1+n2] = [strip[1]-strip[0] for strip in strips2]
        
        #plot them
        fig,ax = plt.subplots(2,figsize=(10,10))
        fig.subplots_adjust(hspace=0)
        for s in range(n1+n2): 
            ax[0].plot(angles,mul[s]*(midstrips[i,:,s]-midstrips[i,0,s]),'o'+lst[s],
                       label=labels[s]+str(midstrips[i,0,s])+'-pix',color=colors[s])
        ax[0].set_ylabel("STRIP POSITION (y-pix)")
        ax[0].legend(loc='lower left',fontsize='xx-small')
        for s in range(n1+n2): 
            ax[1].plot(angles,offsets[i,:,s]-offsets[i,0,s],'o'+lst[s],color=colors[s])    
        ax[1].set_ylabel("STRIP OFFSET (pix)")
        ax[1].set_xlabel("ANGLE HWP")
        ax[1].set_xticks(angles)
        plt.savefig(odir+filt+tname+'-strips-ang.png')
        plt.close(fig)
         #fig.show()
         
    #average    
    foffsets = np.mean(offsets,axis=0)
    fmidstrips = np.mean(midstrips,axis=0)
    fstrips = np.mean(strips,axis=0)
    if len(its) > 1:
        fig,ax = plt.subplots(2,figsize=(10,10))
        fig.subplots_adjust(hspace=0)
        for s in range(0,np.size(fmidstrips[0,:])):
            ax[0].plot(angles,mul[s]*(fmidstrips[:,s]-fmidstrips[0,s]),'o'+lst[s],
                       label=labels[s]+str(fmidstrips[0,s])+'-pix',color=colors[s])
        ax[0].set_ylabel("STRIP POSITION (y-pix)")
        ax[0].legend(loc='lower left',fontsize='xx-small')
        for s in range(0,np.size(foffsets[0,:])): 
            ax[1].plot(angles,(foffsets[:,s]-foffsets[0,s]),'o'+lst[s],color=colors[s])    
        ax[1].set_ylabel("STRIP OFFSET (pix)")
        ax[1].set_xlabel("ANGLE HWP")
        ax[1].set_xticks(angles)
        plt.savefig(odir+filt+'-strips-ang.png')
        plt.close(fig)

    #median
    finoffsets = np.median(foffsets,axis=0)
    finmidstrips = np.median(fmidstrips,axis=0)
    finstrips = np.median(fstrips,axis=0)
    #erfinstrips = np.std(fstrips,axis=0)
    wfile = open(odir+filt+'-strips-ang.dat','w')
    wfile.write('Y1-STRIP   Y2-STRIP\n')
    [wfile.write("%f    %f\n" %(finstrips[f][0],finstrips[f][1])) for f in range(0,len(finstrips))]
    #[wfile.write("%f    %f     %f     %f\n"
    #             %(finstrips[f][0],finstrips[f][1],erfinstrips[f][0],erfinstrips[f][1])) for f in range(0,len(finstrips))]
    wfile.close()

## ---------------------------------------------------------------------------
## ---------------- ANALYZE STRIPS VS FILTER-----------------------------------
## ---------------------------------------------------------------------------
## PURPOSE:
##       Analyse strip positions with respect to wavelength, i.e. filter
##       Strip positions are the ones found with function 'get_strips'
## INPUT:
##       1. filters, e.g. [b_HIGH,v_HIGH,R_SPECIAL,I_BESS]
##       2. output directory
## OUTPUT:
##       It does not return anything but creates a plot ('..strips-filt.png')
##       of strip positions vs filter and ASCII file ('..strips-filt.dat').

def analyse_filtstrips(filts,odir):

    n1,n2 = 12,9
    filtswaves = {'b_HIGH':4413,'v_HIGH':5512,'R_SPECIAL':6586,'I_BESS':8060}
    filtshort = {'b_HIGH':'B','v_HIGH':'V','R_SPECIAL':'R','I_BESS':'I'}
    labels,lst = np.zeros(n1+n2,dtype=object),np.zeros(n1+n1,dtype=object)
    mul = np.zeros(n1+n2,dtype=int)
    labels[0:n1],lst[0:n1],mul[0:n1] = 'chip1-','-',1.0
    labels[n1:n1+n2],lst[n1:n1+n2],mul[n1:n1+n2] = 'chip2-','--',1.0
    import matplotlib.cm as cm
    colors = cm.gist_rainbow(np.linspace(0,1,n1+n2))
    
    #read them
    offsets = np.zeros((len(filts),n1+n2),dtype=float)
    #strips = np.zeros((len(filts),n1+n2),dtype=float)
    strips = np.zeros((len(filts),n1+n2,2),dtype=float)
    allwaves = np.zeros(len(filts),dtype=float)
    for f,filt in enumerate(filts):
        fname = odir+filt+'-strips-ang.dat'
        #strips[f,:],offsets[f,:] = np.loadtxt(fname,skiprows=1,unpack=True)
        strips[f,:] = np.loadtxt(fname,skiprows=1,unpack=True).transpose()
        allwaves[f] = filtswaves[filt]
        
    #plot them
    ifilts = np.linspace(1,len(filts),len(filts))
    sort = np.argsort(allwaves)
    import pdb;pdb.set_trace()
    fig,ax = plt.subplots(2,figsize=(10,10))
    fig.subplots_adjust(hspace=0)
    for s in range(0,n1+n2): 
        ax[0].plot(ifilts,mul[s]*(strips[sort,s,0]-strips[sort[0],s,0]),'o'+lst[s],
                   label=labels[s]+str(strips[0,s,0])+'-pix',color=colors[s])
        #ax[0].plot(ifilts,mul[s]*(strips[sort,s]-strips[sort[0],s]),'o'+lst[s],
        #           label=labels[s]+str(strips[0,s])+'-pix',color=colors[s])
    ax[0].set_ylabel("STRIP POSITION (y-pix)")
    ax[0].legend(loc='lower left',fontsize='xx-small')
    for s in range(0,n1+n2): 
        ax[1].plot(ifilts,(strips[sort,s,1]-strips[sort[0],s,0]),'o'+lst[s],color=colors[s])
        #ax[1].plot(ifilts,(offsets[sort,s]-offsets[sort[0],s]),'o'+lst[s],color=colors[s])
    ax[1].set_ylabel("STRIP OFFSET (pix)")
    ax[1].set_xlabel("FILTER")
    ax[1].set_xticks(ifilts)
    ax[1].set_xticklabels(filtshort[filts[sort]])
    plt.savefig(odir+'strips-filt.png')
    plt.close(fig)

    fig,ax = plt.subplots(2,figsize=(10,10))
    fig.subplots_adjust(hspace=0)

    for s in range(0,n1): 
        ax[0].plot(ifilts,mul[s]*(strips[sort,s,0]-strips[sort[0],s,0]),'^'+lst[s],
                   label='strip-'+str(s),color=colors[s])
        ax[0].plot(ifilts,mul[s]*(strips[sort,s,1]-strips[sort[0],s,1]),'v'+lst[s],
                   color=colors[s])
    ax[0].set_ylabel("CCD1 $\Delta$(y) [pix]")
    ax[0].legend(loc='lower left',fontsize='xx-small')
    for s in range(n1,n1+n2):
        ax[1].plot(ifilts,mul[s]*(strips[sort,s,0]-strips[sort[0],s,0]),'^'+lst[s],
                   label='strip-'+str(s-n1),color=colors[s])
        ax[1].plot(ifilts,mul[s]*(strips[sort,s,1]-strips[sort[0],s,1]),'v'+lst[s],
                   color=colors[s])
    ax[1].set_ylabel("CCD2 $\Delta$(y) [pix]")
    ax[1].legend(loc='lower left',fontsize='xx-small')    
    ax[1].set_xlabel("FILTER")
    ax[1].set_xticks(ifilts)
    ax[1].set_xticklabels(filtshort[filts[sort]])
    plt.savefig(odir+'strips-filt2.png')
    plt.close(fig)
    #fig.show()

        
    #final
    #finoffsets = np.median(offsets,axis=0)
    #finstrips = np.median(strips,axis=0)
    finstrips = np.median(strips,axis=0)
    wfile = open(odir+'strips-filt.dat','w')
    wfile.write('Y1-STRIP   Y2-STRIP\n')
    [wfile.write("%f    %f\n" %(finstrips[f,0],finstrips[f,1])) for f in range(0,len(finstrips))]
    #[wfile.write("%f    %f\n" %(finstrips[f],finoffsets[f])) for f in range(0,len(finoffsets))]
    wfile.close()
        
    
## ---------------------------------------------------------------------------
## ---------------- ANALYZE QUADPARS VS ANGLE--------------------------------
## ---------------------------------------------------------------------------
## PURPOSE:
##       Analyse quadratic parameters of 'find_stars' with respect to HWP-angle
## INPUT:
##       1. filter, e.g. b_HIGH
##       2. angles, e.g. [0,22.5,45.0,67.5]
##       3. output path directory
##       4. object type, 'GAL' or 'STD'
##       5. offset/iteration array, eg. [0,1]
## OPTIONAL INPUT:
##       - fwhm: Same FWHM used when running 'find_stars'
##       - threshold: Same threshold used when running 'find_stars'
## OUTPUT:
##       It does not return anything but creates plot ('..quadpars-ang.png')
##       of quad pars vs angle and ASCII file ('..quadpars-ang.dat').

def analyse_angquadpars(filt,angles,odir,obj,its,fwhm=5.0,threshold=5.0):

    #npars
    npars=4
    #its
    nits = len(its)
    
    #read them
    allpars = np.zeros((nits,len(angles),npars),dtype=float)
    for i in range(0,len(its)):
        tname=''
        if len(its) > 1: tname = '-'+its['type'][i]+np.str(i)

        for p,angle in enumerate(angles):
            fname = odir+filt+'-ang'+str(angle)+tname
            allpars[i,p,:] = np.loadtxt(fname+'-chip12-fwhm'+str(fwhm)+
                                      '-thresh'+str(threshold)+'-quadpars.dat',skiprows=1)
        #plot them    
        fig,ax = plt.subplots(4,figsize=(10,10))
        fig.subplots_adjust(hspace=0)
        for p in range(0,npars): 
            ax[p].plot(angles,allpars[i,:,p],'o-')
            ax[p].set_ylabel("PAR-"+str(p+1))
        ax[npars-1].set_xlabel("ANGLE HWP")
        ax[npars-1].set_xticks([0,22.5,45,67.5])
        #fig.show()
        plt.savefig(odir+filt+tname+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-quadpars-ang.png')
        
    #avg
    fallpars = np.mean(allpars,axis=0)
    if len(its) >1:
        fig,ax = plt.subplots(4,figsize=(10,10))
        fig.subplots_adjust(hspace=0)
        for p in range(0,npars): 
            ax[p].plot(angles,fallpars[:,p],'o-')
            ax[p].set_ylabel("PAR-"+str(p+1))
        ax[npars-1].set_xlabel("ANGLE HWP")
        ax[npars-1].set_xticks([0,22.5,45,67.5])
        #fig.show()
        plt.savefig(odir+filt+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-quadpars-ang.png')
        
    #median
    finpars = np.median(fallpars,axis=0)
    erfinpars = np.std(fallpars,axis=0)
    np.savetxt(odir+filt+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-quadpars-ang.dat',
               (finpars,erfinpars),fmt='%20s')
    plt.close('all')
    
## ---------------- ANALYZE QUADPARS VS FILTER------------------------------
## ---------------------------------------------------------------------------
## PURPOSE:
##       Analyse quadratic parameters of 'find_stars' with respect to wavelength, i.e. filter
## INPUT:
##       1. filters, e.g.[b_HIGH,v_HIGH,R_SPECIAL,I_BESS]
##       2. output path directory
## OPTIONAL INPUT:
##       - fwhm: Same FWHM used when running 'find_stars'
##       - threshold: Same threshold used when running 'find_stars'
## OUTPUT:
##       It does not return anything but creates plot ('..quadpars-filt.png')
##       of quad pars vs angle and ASCII file ('..quadpars-filt.dat').

def analyse_filtquadpars(filts,odir,fwhm=5.0,threshold=5.0):

    npars=4
    filtswaves = {'b_HIGH':4413,'v_HIGH':5512,'R_SPECIAL':6586,'I_BESS':8060}
    
    #read them
    allpars = np.zeros((len(filts),npars),dtype=float)
    allwaves = np.zeros(len(filts),dtype=float)
    for f,filt in enumerate(filts):
        fname = odir+filt+'-fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-quadpars-ang.dat'
        allpars[f,:] = np.loadtxt(fname)
        allwaves[f] = filtswaves[filt]
    #plot them    
    ifilts = np.linspace(1,len(filts),len(filts))
    sort = np.argsort(allwaves)
    fig,ax = plt.subplots(4,figsize=(6,8))
    fig.subplots_adjust(hspace=0)
    for p in range(0,npars): 
        ax[p].plot(ifilts,allpars[sort,p],'o-')
        ax[p].set_ylabel("PAR-"+str(p+1))
    ax[npars-1].set_xlabel("FILTER")
    ax[npars-1].set_xticks(ifilts)
    ax[npars-1].set_xticklabels(filts[sort])
    ax[0].set_title('Quadpars vs filter')
    plt.savefig(odir+'fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-quadpars-filt.png')
    plt.close(fig)
    
    #median
    #finpars = np.median(allpars,axis=0)
    np.savetxt(odir+'fwhm'+str(fwhm)+'-thresh'+str(threshold)+'-quadpars-filt.dat',
               (allpars[sort,:]),fmt='%20s')


## ---------------------------------------------------------------------------
## ---------------- ANALYZE POLARIZATION/ANGLE VS FILTER ----------------------
## ---------------------------------------------------------------------------
## PURPOSE:
##       Analyse polarization/angle of point sources with respect to wavelength, i.e. filter
## INPUT:
##       1. filters, e.g.[b_HIGH,v_HIGH,R_SPECIAL,I_BESS]
##       2. target name, e.g. 'Vela1'
##       3. output path directory
## OPTIONAL INPUT:
##       - fit: Perform Serkowski fit (boolean), def: True
## OUTPUT:
##       It does not return anything but creates plots
##           ('..pol-filt.png', '..pol-filt_Serkowski.png')
## DEPENDENCIES: collections

def analyse_filtphpol(filts,target,dir,fit=None):

    from collections import OrderedDict
    filtshorts = {'b_HIGH':'B','v_HIGH':'V','R_SPECIAL':'R','I_BESS':'I'}
    filtswaves = {'b_HIGH':4413,'v_HIGH':5512,'R_SPECIAL':6586,'I_BESS':8060}

    nfilts = len(filts)
    filtappol,filtpsfpol = np.zeros(nfilts,dtype=float),np.zeros(nfilts,dtype=float)
    erfiltappol,erfiltpsfpol = np.zeros(nfilts,dtype=float),np.zeros(nfilts,dtype=float)
    filtapang,filtpsfang = np.zeros(nfilts,dtype=float),np.zeros(nfilts,dtype=float)
    erfiltapang,erfiltpsfang = np.zeros(nfilts,dtype=float),np.zeros(nfilts,dtype=float)

    #re-order
    waves = np.asarray([filtswaves[filt] for filt in filts])
    sort = np.argsort(waves)
    filts,waves = filts[sort],waves[sort]
    
    ##read photometry pol files
    for f,filt in enumerate(filts):
        pol,angle,Q,U,F,erpol,erangle,erQ,erU,erF = np.load(dir+filt+"_aperphot.npy")
        filtappol[f],filtapang[f] = pol*100,angle
        erfiltappol[f],erfiltapang[f] = erpol*100,erangle
        pol,angle,Q,U,F,erpol,erangle,erQ,erU,erF = np.load(dir+filt+"_psfphot.npy")
        filtpsfpol[f],filtpsfang[f] = pol*100,angle
        erfiltpsfpol[f],erfiltpsfang[f] = erpol*100,erangle

    ##read literature pol file
    litfile = home+'/crisp/FORS2-POL/Information/STDSTARS//literature_'+target+'.dat'
    litinfo = np.loadtxt(litfile,dtype={'names':('source','filter','pol','erpol','angle','erangle'),
                                        'formats':('O','O','f','f','f','f')})
    source = np.unique(litinfo['source']) 
    srccolor = ['aqua','coral','olive','plum']
    #from random_colors import *
    #srccolor = random_colors(len(source)
        
    ##plot pol/angle vs filt
    ifilts = np.linspace(1,len(filts),len(filts))
    fig,(ax1,ax2) = plt.subplots(2,figsize=(6,8))
    fig.subplots_adjust(hspace=0)
    ax1.errorbar(ifilts,filtappol,yerr=erfiltappol,fmt='o',label='APER')
    ax1.errorbar(ifilts+0.1,filtpsfpol,yerr=erfiltpsfpol,fmt='s',label='PSF')
    ax2.errorbar(ifilts,filtapang,yerr=erfiltapang,fmt='o')
    ax2.errorbar(ifilts+0.1,filtpsfang,yerr=erfiltpsfang,fmt='s')
    for f,filt in enumerate(filts):
       for si in np.arange(0,len(source)):
           find = np.argwhere((litinfo['filter'] == filtshorts[filt]) &
                              (litinfo['source'] == source[si]))
           
           if len(find) < 1: continue
           uppol = litinfo['pol'][find]+litinfo['erpol'][find]
           lopol = litinfo['pol'][find]-litinfo['erpol'][find]
           ax1.plot(np.array([ifilts[f]-0.5,ifilts[f]+0.5]),
                    np.array([litinfo['pol'][find],litinfo['pol'][find]]).reshape(-1),
                    color=srccolor[si],linewidth=0.5)
           ax1.fill_between(np.array([ifilts[f]-0.5,ifilts[f]+0.5]),
                            np.array([uppol,uppol]).reshape(-1),
                            np.array([lopol,lopol]).reshape(-1),alpha=0.5,facecolor=srccolor[si])

           upang = litinfo['angle'][find]+litinfo['erangle'][find]#-180
           loang = litinfo['angle'][find]-litinfo['erangle'][find]#-180
           ax2.plot(np.array([ifilts[f]-0.5,ifilts[f]+0.5]),
                    np.array([litinfo['angle'][find],litinfo['angle'][find]]).reshape(-1),
                    #np.array([litinfo['angle'][find]-180,litinfo['angle'][find]-180]).reshape(-1),
                    color=srccolor[si],linewidth=0.5)
           if np.isfinite(upang) and np.isfinite(loang): 
               ax2.fill_between(np.array([ifilts[f]-0.5,ifilts[f]+0.5]),
                                np.array([upang,upang]).reshape(-1),
                                np.array([loang,loang]).reshape(-1),alpha=0.5,facecolor=srccolor[si],
                                label=source[si])
           
           #del find,uppol,lopol,upang,loang

    ax1.legend()
    ax1.set_ylabel("POL %")
    ax2.set_ylabel("ANGLE ")
    handles, labels = ax2.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(),loc='lower left')
    ax2.set_xlabel("FILTER")
    ax2.set_xticks(ifilts)
    ax2.set_xticklabels(filts)
    plt.savefig(dir+'pol-filt.png')
    plt.close(fig)
    
    ##fit Serkowski law
    if (fit is not None) and (len(filts) >= 4):
        ##read from literature
        lines = tuple(open(litfile, 'r'))
        litpars = [line.split() for line in lines if 'Serkowski' in line]

        ##fit
        from scipy import optimize
        serkfct = lambda x,a,b,c: a*np.exp(-b*(np.log(c/x))**2)
        serkfunc = lambda p,x: p[0]*np.exp(-p[1]*(np.log(p[2]/x))**2)
        errfunc = lambda p,x,y,yerr: (serkfunc(p,x)-y)/yerr
        p0 = [1.0,1.0,6000]

        ## leastsq: no input errors (output yes)
        #ap_pars,ap_succ = optimize.leastsq(errfunc,p0[:],args=(waves,filtappol,erfiltappol))
        #psf_pars,psf_succ = optimize.leastsq(errfunc,p0[:],args=(waves,filtpsfpol,erfiltpsfpol))

        ## curvefit: input errors (output yes)
        ap_pars, ap_cov = optimize.curve_fit(serkfct,waves,filtappol,p0=p0,sigma=erfiltappol)
        psf_pars, psf_cov = optimize.curve_fit(serkfct,waves,filtpsfpol,p0=p0,sigma=erfiltpsfpol)
        ap_erpars = np.array([np.absolute(ap_cov[i][i])**0.5 for i in range(len(p0))])
        psf_erpars = np.array([np.absolute(psf_cov[i][i])**0.5 for i in range(len(p0))])
        
        wavearr = np.arange(2000,10000,10,dtype=float)
        ap_arr,psf_arr = serkfct(wavearr,*ap_pars), serkfct(wavearr,*psf_pars)
        #ap_arr,psf_arr = serkfunc(ap_pars,wavearr), serkfunc(psf_pars,wavearr)

        
        ##plot pol vs filt and fit
        fig,ax = plt.subplots(1,figsize=(9,6))
        ax.errorbar(waves,filtappol,yerr=erfiltappol,fmt='o',label='APER')
        ax.errorbar(waves+50,filtpsfpol,yerr=erfiltpsfpol,fmt='s',label='PSF')
        ax.plot(wavearr,ap_arr,'-',color='blue',label='APER-Serkowski')
        ax.plot(wavearr,psf_arr,'-',color='orange',label='PSF-Serkowski')
        #plot literature laws
        for si,lpars in enumerate(litpars):
            larr = serkfunc(np.asarray(lpars[2:5],dtype=float),wavearr)
            ax.plot(wavearr,larr,'--',color=srccolor[si],label=lpars[0])
        ax.legend()
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylabel("Polarization [%] ")
        plt.savefig(dir+'pol-filt_Serkowski.png')
        plt.close(fig)
        
## ---------------------------------------------------------------------------
## ---------------- ANALYZE POLARIZATION/ANGLE VS RADIUS ----------------------
## ---------------------------------------------------------------------------
## PURPOSE:
##       Analyse polarization/angle with respect to radius
## INPUT:
##       1.Polarization image
##       2.Angle image
##       3.Dir+name of output plot
## OPTIONAL INPUT:
##   center:  where central optical path is located (def: ny/2,nx/2)
##   scatter: Do scatter plot instead of density plot (def: False
##   radfit: perform cubic radial fit (def: True)
##   parfit: perform paraboloid fit (def: False)
##   inla: perform INLA comparison (precalculated) def: False

def radius_dependence(pol,angle,filename,center=None,scatter=False,erpol=None,inla=False,
                      filt=None,pixscale=0.126,pixbin=2.0,fitradius=None,radfit=True,parfit=False):

    print("   Plotting polarization vs radius ")   
    xi,xf,yi,yf = 188,1862,434,1965    
    ny,nx = np.shape(pol)
    if center is None: center=[ny/2,nx/2]
    x,y = np.arange(0,nx,dtype=float),np.arange(0,ny,dtype=float)
    xx,yy = np.meshgrid(x-center[1],y-center[0])
    rr = np.sqrt(xx**2.0+yy**2.0)
    ax,ay,radius,allpol,allangle = xx.reshape(-1),yy.reshape(-1),rr.reshape(-1),pol.reshape(-1),angle.reshape(-1)
    if erpol is not None: allerpol = erpol.reshape(-1)
    else: allerpol = np.full(np.shape(allpol),0.0001)

    mask = (allpol > 0)
    ind = (np.where(mask))[0]
    fitmask = mask.copy()
    if fitradius is not None:
        fitmask = (fitmask) & (rr > fitradius)
    fitind = (np.where(fitmask))[0]

    ##limits
    lopol,uppol = np.percentile(allpol[ind],5),np.percentile(allpol[ind],95)
    loang,upang = np.percentile(allangle[ind],5),np.percentile(allangle[ind],95) 
    
    if scatter:
        uniq = (np.unique(allpol[fitind],return_index=True))[1]
        fx,fy = ax[fitind[uniq]],ay[fitind[uniq]]
        frad,fpol,ferpol = radius[fitind[uniq]],allpol[fitind[uniq]],allerpol[fitind[uniq]]
    else:
        fx,fy = ax[fitind],ay[fitind]
        frad,fpol,ferpol = radius[fitind],allpol[fitind],allerpol[fitind]
    
    ##fit
    if radfit:
        from scipy import optimize

        ## --1D
        patatpars = {'v_HIGH':[0.0,0.012,0.046,0.002],'I_BESS':[0.0,-0.017,0.105,-0.006]}
        radarr = np.arange(0,1300,1,dtype=float)
        #cubefct0 = lambda r,b,c,d,r0: b*(r-r0)+c*(r-r0)**2+d*(r-r0)**3
        #p0 = [0.0,0.1,0.1,0.0]
        cubefct0 = lambda r,b,c,d: b*r+c*r**2+d*r**3
        p0 = [0.0,0.1,0.1]
        fitpars0,fitcov0 = optimize.curve_fit(cubefct0,frad,fpol,p0=p0,sigma=ferpol)
        fiterpars0 = np.sqrt(np.diag(fitcov0))
        polarr0 = cubefct0(radarr,*fitpars0)
        #cubefct = lambda r,a,b,c,d,r0: a+b*(r-r0)+c*(r-r0)**2+d*(r-r0)**3
        #p = [0.0,0.0,0.1,0.1,0.0]
        cubefct = lambda r,a,b,c,d: a+b*r+c*r**2+d*r**3
        p = [0.0,0.0,0.1,0.1]
        fitpars,fitcov = optimize.curve_fit(cubefct,frad,fpol,p0=p,sigma=ferpol)
        fiterpars = np.sqrt(np.diag(fitcov))
        polarr = cubefct(radarr,*fitpars)
        print("      Radial fit parameters with p(r=0)=0: %.4e,%.4e,%.4e"
              %(fitpars0[0],fitpars0[1],fitpars0[2]))
        print("      Radial fit error parameters with p(r=0)=0: %.4e,%.4e,%.4e"
              %(fiterpars0[0],fiterpars0[1],fiterpars0[2]))
        print("      Radial fit parameters: %.2e,%.2e,%.2e,%.2e" %(fitpars[0],fitpars[1],fitpars[2],fitpars[3]))
        print("      Radial fit error parameters: %.2e,%.2e,%.2e,%.2e"
              %(fiterpars[0],fiterpars[1],fiterpars[2],fiterpars[3]))
        if filt in patatpars.keys():
            patatarr = cubefct(radarr*1.0/60.0*pixscale*pixbin,*patatpars[filt])/100.0

    mask = (np.isfinite(pol) & (pol > 0))
            
    ## --2D:
    if parfit:
        from scipy import optimize
        
        #- elliptical paraboloid
        xarr = np.arange(-int(nx/2),int(nx/2),1,dtype=float)
        yarr = np.arange(-int(nx/2),int(nx/2),1,dtype=float)
        rarr = np.sqrt(xarr**2.0+yarr**2.0)
        paraboloid = lambda xy,a,b,x0,y0: (xy[0]-x0)**2.0/a**2.0 + (xy[1]-y0)**2.0/b**2.0
        pp0 = [1.0,1.0,0.0,0.0]
        #paraboloid = lambda xy,a,b: xy[0]**2.0/a**2.0 + xy[1]**2.0/b**2.0
        #pp0 = [1.0,1.0]

        try:
            fitppars,fitppcov = optimize.curve_fit(paraboloid,(fx,fy),fpol,p0=pp0,sigma=ferpol)
            fiterppars = np.sqrt(np.diag(fitppcov))
        except:
            fitppars,fiterppars = pp0,np.zeros(len(pp0))*np.nan
        rpolarr = paraboloid((xarr,yarr),*fitppars)
        ppolarr = paraboloid((ax,ay),*fitppars)
        ppol = ppolarr.reshape((ny,nx))
        respol = pol-ppol
        
        #respol[mm] = np.nan
        
        print("      Paraboloid fit parameters: %.2e,%.2e" %(fitppars[0],fitppars[1]))
        print("      Paraboloid fit error parameters: %.2e,%.2e" %(fiterppars[0],fiterppars[1]))
        medrespol = np.median(respol[mask])
        stdrespol = np.median(np.abs(respol[mask]-medrespol))
        print("      Residual Paraboloid median/MAD: %.4e,%.4e" %(medrespol,stdrespol)) 
        
        #- rotated paraboloid
        rotparaboloid = lambda xy,a,b,theta,x0,y0:\
                        ((xy[0]-x0)*np.cos(theta)-(xy[1]-y0)*np.sin(theta))**2.0/a**2.0 + \
                        ((xy[0]-x0)*np.sin(theta)+(xy[1]-y0)*np.cos(theta))**2.0/b**2.0
        rpp0 = [1.0,1.0,0.0,0.0,0.0]
        try:
            fitrppars,fitrppcov = optimize.curve_fit(rotparaboloid,(fx,fy),fpol,p0=rpp0,sigma=ferpol)
            fiterrppars = np.sqrt(np.diag(fitrppcov))
        except:
            fitrppars,fiterrppars = rpp0,np.zeros(len(rpp0))*np.nan
        rrpolarr = rotparaboloid((xarr,yarr),*fitrppars)
        rppolarr = rotparaboloid((ax,ay),*fitrppars)
        rppol = rppolarr.reshape((ny,nx))
        resrppol = pol - rppol
        #resrppol[mm] = np.nan
        print("      RotParaboloid fit parameters: %.4e,%.4e,%.4e,%.4e,%.4e"
              %(fitrppars[0],fitrppars[1],fitrppars[2],fitrppars[3],fitrppars[4]))
        print("      RotParaboloid fit error parameters: %.4e,%.4e,%.4e,%.4e,%.4e"
              %(fiterrppars[0],fiterrppars[1],fiterrppars[2],fiterrppars[3],fiterrppars[4]))
        medresrppol = np.median(resrppol[mask])
        stdresrppol = np.median(np.abs(resrppol[mask]-medresrppol))
        print("      Residual RotParaboloid median/MAD: %.4e,%.4e" %(medresrppol,stdresrppol)) 
   
        #- rotated paraboloid plus constant
        crotparaboloid = lambda xy,a,b,theta,x0,y0,cst:\
                        ((xy[0]-x0)*np.cos(theta)-(xy[1]-y0)*np.sin(theta))**2.0/a**2.0 + \
                        ((xy[0]-x0)*np.sin(theta)+(xy[1]-y0)*np.cos(theta))**2.0/b**2.0 + cst
        crpp0 = [1.0,1.0,0.0,0.0,0.0,0.0]
        try:
            fitcrppars,fitcrppcov = optimize.curve_fit(crotparaboloid,(fx,fy),fpol,
                                                       p0=crpp0,sigma=ferpol)
            fitercrppars = np.sqrt(np.diag(fitcrppcov))
        except:
            fitcrppars,fitercrppars = crpp0,np.zeros(len(crpp0))*np.nan
        crrpolarr = crotparaboloid((xarr,yarr),*fitcrppars)
        crppolarr = crotparaboloid((ax,ay),*fitcrppars)
        crppol = crppolarr.reshape((ny,nx))
        rescrppol = pol - crppol
        #resrppol[mm] = np.nan
        print("      CstRotParaboloid fit parameters: %.4e,%.4e,%.4e,%.4e,%.4e,%.4e"
              %(fitcrppars[0],fitcrppars[1],fitcrppars[2],fitcrppars[3],fitcrppars[4],fitcrppars[5]))
        print("      CstRotParaboloid fit error parameters: %.4e,%.4e,%.4e,%.4e,%.4e,%.4e"
              %(fitercrppars[0],fitercrppars[1],fitercrppars[2],fitercrppars[3],fitercrppars[4],fitercrppars[5]))
        medrescrppol = np.median(rescrppol[mask])
        stdrescrppol = np.median(np.abs(rescrppol[mask]-medrescrppol))
        print("      Residual CstRotParaboloid median/MAD: %.4e,%.4e" %(medrescrppol,stdrescrppol)) 

        fits.writeto(filename+'-polmodel.fits',crppol,clobber=True) 
        np.savetxt(filename+'-polmodel.dat',(fitcrppars,fitercrppars),fmt='%8e',
                   header='CstRotParaboloid fit Pol parameters\nResidual median/MAD Pol: %.4e %.4e'
                   %(medrescrppol,stdrescrppol))
        
        #- elliptical paraboloid + plane
        #plane = lambda xy,a,b,c : a + b*xy[0] + c*xy[1]
        #planeparaboloid = lambda xy,pb_a,pb_b,pl_a,pl_b,pl_c,x0,y0: paraboloid(xy,*(pb_a,pb_b,x0,y0))+\
        #                  plane(xy,*(pl_a,pl_b,pl_c))
        #ppp0 = [1.0,1.0,0.0,0.1,0.1,0.0,0.0]
        #fitpppars,fitpppcov = optimize.curve_fit(planeparaboloid,(fx,fy),fpol,p0=ppp0,sigma=ferpol)
        #fiterpppars = np.sqrt(np.diag(fitpppcov))
        #pppolarr = planeparaboloid((ax,ay),*fitpppars)
        #pppol = pppolarr.reshape((ny,nx))
        #respppol = pol - pppol
        ##respppol[mm] = np.nan
        #rppolarr = planeparaboloid((xarr,yarr),*fitpppars)
        #medrespppol = np.median(respppol[mask])
        #stdrespppol = np.median(np.abs(respppol[mask]-medrespppol))
        #print("      Residual PlaneParaboloid median/MAD: %.4e,%.4e" %(medresrppol,stdresrppol)) 
               
        ##rarr,rind = np.unique(radius,return_index=True)
        ##rpolarr,rppolarr = ppolarr[rind],pppolarr[rind]
        ##rpolarr = [np.median(ppolarr[np.argwhere(i==radius)]) for i in rarr]
        ##rppolarr = [np.median(pppolarr[np.argwhere(i==radius)]) for i in rarr]

    ## --running median
    def running_median(x,y,binint=50):#,nbins=None):
        xmin,xmax = np.min(x), np.max(x)
        bins = np.arange(xmin,xmax,binint)
        nbins = len(bins)
        #bins = np.linspace(np.min(x), np.max(x), nbins)
        binmed = [np.median(y[(x <= bins[i+1]) & (x > bins[i])]) for i in range(0,nbins-1)]
        return 0.5*(bins[1:nbins]+bins[0:nbins-1]),binmed
                         
    ## -- plot
    # vs radius
    fs,window = 12,3
    #fig,(ax1,ax2) = plt.subplots(2,figsize=(6,8))
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
    if scatter:
        uniq = (np.unique(allpol[ind],return_index=True))[1]
        ax1.scatter(radius[ind[uniq]],allpol[ind[uniq]],s=6)
        #rmed,bmed = running_median(radius[ind[uniq]],allpol[ind[uniq]],binint=50)
        #ax1.plot(rmed,bmed,'--',color='black',label='median')
        #ax2.scatter(radius[ind[uniq]],allangle[ind[uniq]],s=6)
        ax2.scatter(np.arctan2(ay[ind[uniq]],ax[ind[uniq]])/np.pi*180,allangle[ind[uniq]],s=6)
        ax1.set_ylim([lopol,uppol])
        ax2.set_ylim([loang,upang])
    else:    
        ax1.hist2d(radius[ind],allpol[ind],cmap='rainbow',bins=100,range=[[0,1300],[lopol,uppol]])
        #rmed,bmed = running_median(radius[ind],allpol[ind],binint=5)
        #ax1.plot(rmed,bmed,'--',color='black',label='median')
        #ax2.hist2d(radius[ind],allangle[ind],cmap='rainbow',bins=100,range=[[0,1300],[loang,upang]])
        ax2.hist2d(np.arctan2(ay[ind],ax[ind])/np.pi*180,allangle[ind],cmap='rainbow',bins=100,range=[[-90,90],[loang,upang]])
        

    if radfit:
        ax1.plot(radarr,polarr,'-',color='magenta',label='fit')#: %.2e+%.2e$r$+%.2e$r^2$+%.2e$r^3$'
                 #%(fitpars[0],fitpars[1],fitpars[2],fitpars[3]))
        ax1.plot(radarr,polarr0,'--',color='magenta',label='fit $p(r=0)=0$')
        if filt in patatpars.keys():
            ax1.plot(radarr,patatarr,color='black',label='P&R06')#Patat06')
    if parfit:
        ax1.plot(rarr,rpolarr,'-',color='yellow',label='<Paraboloid fit>')
        ax1.plot(rarr,rrpolarr,'-',color='green',label='<RotParaboloid fit>')
    if radfit or parfit: ax1.legend(fontsize=fs-2)#'x-small')
        #ax2.plot([-00,200],[-200,200],'--',color='black')#WRONG!!
    ax1.set_xlim([0,1300])#np.max(radius)])
    ax1.set_ylabel("Polarization",fontsize=fs)
    ax2.set_ylabel("Angle [deg]",fontsize=fs)
    ax1.set_xlabel("Radius [pix]",fontsize=fs)
    ax2.set_xlabel("atan(y/x) [deg]",fontsize=fs)
    ax1.tick_params(labelsize=fs-2);
    ax2.tick_params(labelsize=fs-2) 
    plt.savefig(filename+'-pol-radius.png')
    plt.close(fig)

    ## -vs xy
    fig,(ax1,ax2) = plt.subplots(2,figsize=(6,8))
    if scatter:
        uniq = (np.unique(allpol[ind],return_index=True))[1]
        ax1.scatter(ax[ind[uniq]],allpol[ind[uniq]],s=6)
        ax2.scatter(ay[ind[uniq]],allpol[ind[uniq]],s=6)
        ax1.set_ylim([lopol,uppol])
        ax2.set_ylim([lopol,uppol])
    else:    
        ax1.hist2d(ax[ind],allpol[ind],cmap='rainbow',bins=100,
                   range=[[np.min(ax),np.max(ax)],[lopol,uppol]])
        ax2.hist2d(ay[ind],allpol[ind],cmap='rainbow',bins=100,
                   range=[[np.min(ay),np.max(ay)],[lopol,uppol]])

    if parfit:
        ax1.plot(x-center[1],np.median(ppol,axis=0),'-',color='yellow',label='Paraboloid fit')
        ax2.plot(y-center[0],np.median(ppol,axis=1),'-',color='yellow',label='Paraboloid fit')
        ax1.plot(x-center[1],np.median(crppol,axis=0),'-',color='green',label='CstRotParaboloid fit')
        ax2.plot(y-center[0],np.median(crppol,axis=1),'-',color='green',label='CstRotParaboloid fit')
        ax1.legend(fontsize='x-small')
    ax1.set_xlim([np.min(ax),np.max(ax)])
    ax2.set_xlim([np.min(ay),np.max(ay)])
    ax1.set_ylabel("POL")
    ax2.set_ylabel("POL")
    ax1.set_xlabel("x (pix)")
    ax2.set_xlabel("y (pix)")

    plt.savefig(filename+'-pol-xy.png')
    plt.close(fig)

    # -2D
    if parfit:
        fs = 8#fontsize

        ## Q/U-POL FIT (previously run)
        HQhyp,Qhyp = read_fits(filename+"-Qmodel.fits")
        HUhyp,Uhyp = read_fits(filename+"-Umodel.fits")
        qupol,quang = polfct(Qhyp,Uhyp)
        resqupol = -(pol - qupol) #OJO - binary
        medresqupol = np.median(resqupol[mask])
        stdresqupol = np.median(np.abs(resqupol[mask]-medresqupol))
        print("      Residual QU-POL (HypRotParaboloid) median/MAD: %.4e,%.4e" %(medresqupol,stdresqupol)) 
        
        fig,axes = plt.subplots(3,1,figsize=(4,7),sharex=True,sharey=True)
        fig.subplots_adjust(hspace=0,wspace=0.1)
        pol[pol == 0] = np.nan
        im1 = axes[0].imshow(pol,clim=(lopol,uppol),cmap='rainbow')
        axes[0].set(adjustable='box-forced', aspect='equal')
        axes[0].set_ylabel('y [pix]',fontsize=fs)
        axes[0].invert_yaxis()
        axes[0].text(0.75, 0.1, 'Polarization', horizontalalignment='center',fontsize=fs,
                       verticalalignment='center', transform=axes[0].transAxes)
        axes[0].tick_params(labelsize=fs-2)
        im2 = axes[1].imshow(qupol,clim=(lopol,uppol),cmap='rainbow')
        axes[1].set_ylabel('y [pix]',fontsize=fs)
        axes[1].set(adjustable='box-forced', aspect='equal')
        axes[1].invert_yaxis()
        axes[1].text(0.65, 0.1, 'POL-Q/U Hyperbolic paraboloids', horizontalalignment='center',fontsize=fs,
                       verticalalignment='center', transform=axes[1].transAxes)
        axes[1].tick_params(labelsize=fs-2)
        loqupol,upqupol = np.percentile(resqupol[mask],5),np.percentile(resqupol[mask],95)
        imres = axes[2].imshow(resqupol,clim=(loqupol,upqupol),cmap='rainbow')#binary OJO
        axes[2].set(adjustable='box-forced', aspect='equal')
        axes[2].set_ylabel('y [pix]',fontsize=fs)
        axes[2].set_xlabel('x [pix]',fontsize=fs)
        axes[2].invert_yaxis()
        axes[2].text(0.75, 0.1, 'Residual', horizontalalignment='center',fontsize=fs,
                       verticalalignment='center', transform=axes[2].transAxes)
        axes[2].tick_params(labelsize=fs-2)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        #fig.colorbar(im1, ax=axes[:,0].ravel().tolist(),aspect=50) ## all with same
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im1, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im2, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(imres, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        plt.savefig(filename+'-QUpolmodel.png')
        plt.close(fig)

        
        ## POL FIT
        #fig,axes = plt.subplots(3,2,figsize=(7,7),sharex=True,sharey=True)
        fig,axes = plt.subplots(3,1,figsize=(4,7),sharex=True,sharey=True)
        fig.subplots_adjust(hspace=0,wspace=0.1)
        pol[pol == 0] = np.nan
        im1 = axes[0].imshow(pol,clim=(lopol,uppol),cmap='rainbow')
        axes[0].set(adjustable='box-forced', aspect='equal')
        axes[0].set_ylabel('y [pix]',fontsize=fs)
        axes[0].invert_yaxis()
        axes[0].text(0.75, 0.1, 'Polarization', horizontalalignment='center',fontsize=fs,
                       verticalalignment='center', transform=axes[0].transAxes)
        axes[0].tick_params(labelsize=fs-2)
        im2 = axes[1].imshow(crppol,clim=(lopol,uppol),cmap='rainbow')
        axes[1].set_ylabel('y [pix]',fontsize=fs)
        axes[1].set(adjustable='box-forced', aspect='equal')
        axes[1].invert_yaxis()
        axes[1].text(0.65, 0.1, 'Rotated Paraboloid', horizontalalignment='center',fontsize=fs,
                       verticalalignment='center', transform=axes[1].transAxes)
        axes[1].tick_params(labelsize=fs-2)
        ##mm = (np.isfinite(pol) & (pol > 0))
        lorpol,uprpol = np.percentile(rescrppol[mask],5),np.percentile(rescrppol[mask],95)
        #lorpol,uprpol = np.percentile(resrppol,1),np.percentile(resrppol,99)
        imres = axes[2].imshow(rescrppol,clim=(lorpol,uprpol),cmap='binary')
        axes[2].set(adjustable='box-forced', aspect='equal')
        axes[2].set_ylabel('y [pix]',fontsize=fs)
        axes[2].set_xlabel('x [pix]',fontsize=fs)
        axes[2].invert_yaxis()
        axes[2].text(0.75, 0.1, 'Residual', horizontalalignment='center',fontsize=fs,
                       verticalalignment='center', transform=axes[2].transAxes)
        axes[2].tick_params(labelsize=fs-2)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        #fig.colorbar(im1, ax=axes[:,0].ravel().tolist(),aspect=50) ## all with same
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im1, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im2, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(imres, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
         
        #im1 = axes[0,1].imshow(pol,clim=(lopol,uppol),cmap='rainbow')
        #axes[0,1].set(adjustable='box-forced', aspect='equal')
        #axes[0,1].set_ylabel('y [pix]')
        #axes[0,1].invert_yaxis()
        #axes[0,1].text(0.8, 0.1, 'Polarization', horizontalalignment='center',fontsize=fs,
        #               verticalalignment='center', transform=axes[0,1].transAxes)
        #im2 = axes[1,1].imshow(rppol,clim=(lopol,uppol),cmap='rainbow')
        #axes[1,1].set_ylabel('y [pix]')
        #axes[1,1].set(adjustable='box-forced', aspect='equal')
        #axes[1,1].invert_yaxis()
        #axes[1,1].text(0.8, 0.1, 'RotParaboloid', horizontalalignment='center',fontsize=fs,
        #               verticalalignment='center', transform=axes[1,1].transAxes)
        #imres = axes[2,1].imshow(resrppol,clim=(lorpol,uprpol),cmap='terrain')
        #axes[2,1].set(adjustable='box-forced', aspect='equal')
        #axes[2,1].set_ylabel('y [pix]')
        #axes[2,1].set_xlabel('x [pix]')
        #axes[2,1].invert_yaxis()
        #axes[2,1].text(0.8, 0.1, 'Residual', horizontalalignment='center',fontsize=fs,
        #            verticalalignment='center', transform=axes[2,1].transAxes)
        #divider = make_axes_locatable(axes[0,1])
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        #cb = fig.colorbar(im1, cax=cax, orientation='vertical')
        #cb.ax.tick_params(labelsize=6) 
        #divider = make_axes_locatable(axes[1,1])
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        #cb = fig.colorbar(im2, cax=cax, orientation='vertical')
        #cb.ax.tick_params(labelsize=6) 
        #divider = make_axes_locatable(axes[2,1])
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        #cb = fig.colorbar(imres, cax=cax, orientation='vertical')
        #cb.ax.tick_params(labelsize=6) 
        
        plt.savefig(filename+'-polmodel.png')
        plt.close(fig)

    # -2D
    if inla:
        #POL DEGREE
        HPi,poli = read_fits(home+"/crisp/FORS2-POL/Information/"+filt+"_inlapol.fits")
        respoli = pol - poli
        medrespoli = np.median(respoli[mask])
        stdrespoli = np.median(np.abs(respoli[mask]-medrespoli))
        print("      Residual INLA pol median/MAD: %.4e,%.4e" %(medrespoli,stdrespoli)) 
        
        fs = 8#fontsize
        #fig,axes = plt.subplots(3,2,figsize=(7,7),sharex=True,sharey=True)
        fig,axes = plt.subplots(3,1,figsize=(4,7),sharex=True,sharey=True)
        fig.subplots_adjust(hspace=0,wspace=0.1)
        #pol[pol == 0] = np.nan
        im1 = axes[0].imshow(pol,clim=(lopol,uppol),cmap='rainbow')
        axes[0].set(adjustable='box-forced', aspect='equal')
        axes[0].set_ylabel('y [pix]',fontsize=fs)
        axes[0].invert_yaxis()
        axes[0].text(0.75, 0.1, 'Polarization', horizontalalignment='center',fontsize=fs,
                       verticalalignment='center', transform=axes[0].transAxes)
        axes[0].tick_params(labelsize=fs-2)
        im2 = axes[1].imshow(poli,clim=(lopol,uppol),cmap='rainbow')
        axes[1].set_ylabel('y [pix]',fontsize=fs)
        axes[1].set(adjustable='box-forced', aspect='equal')
        axes[1].invert_yaxis()
        axes[1].text(0.75, 0.1, 'INLA', horizontalalignment='center',fontsize=fs,
                       verticalalignment='center', transform=axes[1].transAxes)
        axes[1].tick_params(labelsize=fs-2)
        #mm = (np.isfinite(pol) & (pol > 0))
        lorpol,uprpol = np.percentile(respoli[mask],5),np.percentile(respoli[mask],95)
        imres = axes[2].imshow(respoli,clim=(lorpol,uprpol),cmap='binary')
        axes[2].set(adjustable='box-forced', aspect='equal')
        axes[2].set_ylabel('y [pix]',fontsize=fs)
        axes[2].set_xlabel('x [pix]',fontsize=fs)
        axes[2].invert_yaxis()
        axes[2].text(0.75, 0.1, 'Residual', horizontalalignment='center',fontsize=fs,
                       verticalalignment='center', transform=axes[2].transAxes)
        axes[2].tick_params(labelsize=fs-2)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        #fig.colorbar(im1, ax=axes[:,0].ravel().tolist(),aspect=50) ## all with same
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im1, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im2, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(imres, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fs-2) 
         
        plt.savefig(filename+'-polinla.png')
        plt.close(fig)
    
        #POL ANGLE
        HPi,angi = read_fits(home+"/crisp/FORS2-POL/Information/"+filt+"_inlaangle.fits")
        resangi = angle - angi
        medresangi = np.median(resangi[mask])
        stdresangi = np.median(np.abs(resangi[mask]-medresangi))
        print("      Residual INLA angle median/MAD: %.4e,%.4e" %(medresangi,stdresangi)) 
        
        fs = 6#fontsize
        #fig,axes = plt.subplots(3,2,figsize=(7,7),sharex=True,sharey=True)
        fig,axes = plt.subplots(3,1,figsize=(4,7),sharex=True,sharey=True)
        fig.subplots_adjust(hspace=0,wspace=0.1)
        #pol[pol == 0] = np.nan
        loang,upang = np.percentile(angle[mask],5),np.percentile(angle[mask],95)
        im1 = axes[0].imshow(angle,clim=(loang,upang),cmap='rainbow')
        axes[0].set(adjustable='box-forced', aspect='equal')
        axes[0].set_ylabel('y [pix]')
        axes[0].invert_yaxis()
        axes[0].text(0.75, 0.1, 'Angle', horizontalalignment='center',fontsize=fs,
                       verticalalignment='center', transform=axes[0].transAxes)
        im2 = axes[1].imshow(angi,clim=(loang,upang),cmap='rainbow')
        axes[1].set_ylabel('y [pix]')
        axes[1].set(adjustable='box-forced', aspect='equal')
        axes[1].invert_yaxis()
        axes[1].text(0.75, 0.1, 'INLA', horizontalalignment='center',fontsize=fs,
                       verticalalignment='center', transform=axes[1].transAxes)
        #mm = (np.isfinite(pol) & (pol > 0))
        lorang,uprang = np.percentile(resangi[mask],5),np.percentile(resangi[mask],95)
        imres = axes[2].imshow(respoli,clim=(lorang,uprang),cmap='binary')
        axes[2].set(adjustable='box-forced', aspect='equal')
        axes[2].set_ylabel('y [pix]')
        axes[2].set_xlabel('x [pix]')
        axes[2].invert_yaxis()
        axes[2].text(0.75, 0.1, 'Residual', horizontalalignment='center',fontsize=fs,
                       verticalalignment='center', transform=axes[2].transAxes)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        #fig.colorbar(im1, ax=axes[:,0].ravel().tolist(),aspect=50) ## all with same
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im1, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=6) 
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im2, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=6) 
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(imres, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=6) 
         
        plt.savefig(filename+'-angleinla.png')
        plt.close(fig)
## ---------------------------------------------------------------------------
## ---------------- PLOT Ord/EXt & F as a fct of angle ----------------------
## ---------------------------------------------------------------------------
## PURPOSE:
##       Plot ordinary/extraordinary fluxes and F values as a fct of HWP angle
##         for individual parts of the image. Also plot fourier coeff.
## INPUT:
##       1.Ordinary
##       2.Extraordinary
##       3.pol
##       4.angle
## OPTIONAL INPUT:
##       corr         Correct for Q0/U0 each k-component
##                    'none','med','cen' (see QUcorrect)
##       mask,emask
##       savefile
## 

def flux_angle(beam,ebeam,pol,angle,num=16,mask=None,emask=None,
               corr='None',savefile=None,errbeam=None,errebeam=None,fit=True):
    
    ##load Fdiff & Fourier
    if savefile is not None:
        fdiff = fits.open(savefile+'-Fdiff.fits')
        erfdiff = fits.open(savefile+'-erFdiff.fits')
        F,erF = fdiff[0].data,erfdiff[0].data
        afour = fits.open(savefile+'-aFourier.fits')
        erafour = fits.open(savefile+'-eraFourier.fits')
        bfour = fits.open(savefile+'-bFourier.fits')
        erbfour = fits.open(savefile+'-erbFourier.fits')
        inten = fits.open(savefile+'-intensity.fits')
        intensity = inten[0].data
        a,b = afour[0].data,bfour[0].data
        era,erb = erafour[0].data,erbfour[0].data

    print("   Plotting Fdiff vs HWP and Fourier coefficients ")    
    nang = len(F)
    ang = 22.5
    kangle = np.arange(nang)*ang
    pltangle = np.arange(nang*20)*ang/20
    ncols=3

    #masks
    if mask is None:
        tot,etot = np.sum(beam,0),np.sum(ebeam,0)
        tmask = (np.isfinite(tot) & (tot > 0) & np.isfinite(etot) & (etot > 0))
        #if errbeam is not None:
        #    errtot,erretot = np.sum(errbeam,0),np.sum(errebeam,0)
        #    tmask = ((tot > 0) & (etot > 0) & (errtot>0) & (erretot>0))
    else:
        smask = (mask & emask)
        tmask = smask[0]
        for i in np.arange(1,nang): tmask*=smask[i]

    beam,ebeam = np.asarray(beam),np.asarray(ebeam)
    errbeam,errebeam = np.asarray(errbeam),np.asarray(errebeam)
    F,erF = np.asarray(F),np.asarray(erF)
    pol,angle = np.asarray(pol),np.asarray(angle)

    ## -- Plot intensity
    fig,ax = plt.subplots(1)
    timg = np.copy(np.arcsinh(intensity))
    timg[timg == 0] = np.nan
    lo = np.percentile(timg[np.isfinite(timg)].flatten(), 5)
    up = np.percentile(timg[np.isfinite(timg)].flatten(), 95)
    c = ax.imshow(timg,clim=(lo,up))#,cmap='rainbow')
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    #cb = fig.colorbar(c,ax=ax)
    #cb.ax.tick_params(labelsize=6) 
    #cb.set_label('Intensity', rotation=270,labelpad=12)
    ax.set_ylim(ax.get_ylim()[::-1])
    fig.savefig(savefile+'-intensity.png')
    
    
    ## -- Plot Fdiff maps
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    nc = nang/2
    fig,ax = plt.subplots(2,nc,sharey=True,figsize=(14,8))
    fig.subplots_adjust(wspace=0.3,hspace=0.05)
    for ai in range(0,nang):
        i,j = np.unravel_index(ai,(2,nc))
        lo = np.percentile(F[ai,tmask].flatten(), 5)
        up = np.percentile(F[ai,tmask].flatten(), 95)
        im = ax[i,j].imshow(F[ai,:,:],clim=(lo,up),cmap='rainbow')
        divider = make_axes_locatable(ax[i,j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=6) 
        ax[i,j].set_title('Fdiff at HWP = %.1f' %(kangle[ai]))
        ax[i,j].set(adjustable='box-forced', aspect='equal')
    ax[0,0].invert_yaxis()#set_ylim(ax[i,j].get_ylim()[::-1])#only once cause sharey
    fig.savefig(savefile+'-Fdiff.png')
    plt.close(fig)

    ## -- Plot flux vs HWP for some pixels/bins    
    cosfct = lambda x,a,c,d: a*np.cos(4*x/180.0*np.pi+c/180.0*np.pi)+d #b=4
    #fit
    if fit:
        from scipy import optimize
    #p0 = [0.1,-2.0*45,0.0]
    #num inds
    fin = np.where(tmask)
    yfin,xfin = fin[0],fin[1]
    nfin = len(fin[0])
    ind = np.linspace(15,nfin-15,num,dtype=int)
    #ind = np.linspace(15,nfin-15,np.sqrt(num),dtype=int)
    #xind,yind = np.meshgrid(ind,ind)
    #xind,yind = xind.reshape(-1),yind.reshape(-1)
    
    ## PLOT ord & ext, intensity and F vs HWP angle
    fig,ax = plt.subplots(num,ncols,sharex=True,figsize=(10,14))
    fig.subplots_adjust(hspace=0,wspace=0.3)
    #import pdb;pdb.set_trace()
    for i in range(0,num):
        ordin = beam[:,yfin[ind[i]],xfin[ind[i]]]
        extra = ebeam[:,yfin[ind[i]],xfin[ind[i]]]
        erordin = errbeam[:,yfin[ind[i]],xfin[ind[i]]]
        erextra = errebeam[:,yfin[ind[i]],xfin[ind[i]]]
        sumoe = ordin+extra
        fdiff = F[:,yfin[ind[i]],xfin[ind[i]]] 
        erfdiff = erF[:,yfin[ind[i]],xfin[ind[i]]] 
        tpol,tang = pol[yfin[ind[i]],xfin[ind[i]]],angle[yfin[ind[i]],xfin[ind[i]]]
        
        ax[i,0].errorbar(kangle,ordin,yerr=erordin,fmt='v',c='y',markersize=5)
        ax[i,0].errorbar(kangle,extra,yerr=erextra,fmt='^',c='r',markersize=5)
        ax[i,1].errorbar(kangle,sumoe,yerr=np.sqrt(erextra**2+erordin**2),fmt='o',markersize=5)
        ax[i,2].errorbar(kangle,fdiff,yerr=erfdiff,fmt='o',markersize=5)

        p0 = [tpol,-2.0*tang,0.0]#[tpol,4.0,-2.0*tang,0.0]
        if fit:
            #import pdb;pdb.set_trace()
            pars,cov = optimize.curve_fit(cosfct,kangle,fdiff,sigma=erfdiff,p0=p0)
            tsin = cosfct(pltangle,*pars) #actual fit
            #print("  Compare: pol %f vs fitpol %f" %(tpol,pars[0]))
   #         print("  Compare: b %f vs fitb %f" %(4.0,pars[1]))
            #print("  Compare: ang %f vs fitang %f" %(tang,pars[1]/-2.0))
            #print("  Compare: d %f vs fitd %f" %(0,pars[2]))
            ax[i,2].plot(pltangle,tsin,'m--',c='purple')
        
        #tsin = tpol*np.cos(4.0*pltangle/180.0*np.pi-2.0*tang/180.0*np.pi) #eq. 6 Patat06
        tsin0 = cosfct(pltangle,*p0) #eq. 6 Patat06   
        ax[i,2].plot(pltangle,tsin0,'-')

        ax[i,0].set_ylim([np.min([ordin,extra]),np.max([ordin,extra])])
        ax[i,1].set_ylim([np.min(sumoe),np.max(sumoe)])
        if fit: ax[i,2].set_ylim([np.min([np.min(fdiff),np.min(tsin)]),
                          np.max([np.max(fdiff),np.max(tsin)])])
        ax[i,0].text(0.4,0.1,'Pixels: '+str(xfin[ind[i]])+'-'+str(yfin[ind[i]]),
                     color='green',transform=ax[i,0].transAxes)
   
        ax[i,0].ticklabel_format(stype='sci',axis='y',scilimits=(0,0))
        ax[i,1].ticklabel_format(stype='sci',axis='y',scilimits=(0,0))
        ax[i,0].tick_params(axis='both',labelsize=6)
        ax[i,1].tick_params(axis='both',labelsize=6)
        ax[i,2].tick_params(axis='both',labelsize=6)
        
    ax[0,2].set_xticks(kangle)        
    fig.text(0.5, 0.07, 'Angle HWP', ha='center')
    fig.text(0.07, 0.5, 'Ordinary/Extraordinary flux', va='center', rotation='vertical',fontsize=8)
    fig.text(0.35, 0.5, 'Intensity', va='center', rotation='vertical',fontsize=8)
    fig.text(0.63, 0.5, 'F coefficient', va='center', rotation='vertical',fontsize=8)
    
    plt.savefig(savefile+'-flux-HWP.png')
    plt.close(fig)

    ## -- Plot fourier maps
    nk = len(a)
    k = np.arange(0,nk,1)
    a,b,era,erb = np.asarray(a),np.asarray(b),np.asarray(era),np.asarray(erb)
    
    fig,ax = plt.subplots(2,nk,sharey=True,figsize=(14,8))
    fig.subplots_adjust(wspace=0.3,hspace=0.05)
    for ki in range(0,nk):
        if corr.lower() != 'none':
            aaa,bbb,abcorr = QUcorrect(a[ki,:],b[ki,:],corr=corr,mask=tmask)
            a[ki,:],b[ki,:] = aaa,bbb
            
        loa = np.percentile(a[ki,tmask].flatten(), 5)
        upa = np.percentile(a[ki,tmask].flatten(), 95)
        im = ax[0,ki].imshow(a[ki,:,:],clim=(loa,upa),cmap='rainbow')
        divider = make_axes_locatable(ax[0,ki])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=6) 
        ax[0,ki].set_title('$Q_k$ at k = %.1f' %(ki))
        if ki == 0: ax[0,ki].set_ylabel('y [pix]')
        ax[0,ki].set_xlabel('x [pix]')
        ax[0,ki].set(adjustable='box-forced', aspect='equal')
        lob = np.percentile(b[ki,tmask].flatten(), 5)
        upb = np.percentile(b[ki,tmask].flatten(), 95)
        im = ax[1,ki].imshow(b[ki,:,:],clim=(lob,upb),cmap='rainbow')
        divider = make_axes_locatable(ax[1,ki])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=6) 
        ax[1,ki].set_title('$U_k$ at k = %.1f' %(ki))
        if ki == 0: ax[1,ki].set_ylabel('y [pix]')
        ax[1,ki].set_xlabel('x [pix]')
        ax[1,ki].set_ylim(ax[1,ki].get_ylim()[::-1])
        ax[1,ki].set(adjustable='box-forced', aspect='equal')
    ax[0,0].invert_yaxis()   
    fig.savefig(savefile+'-Fourier.png')
    plt.close(fig)

    ## -- Q/U errors
    ksig = nk/2
    totaerr = (np.sum(a,axis=0)-a[ksig,:,:])/(nk-1)
    totberr = (np.sum(b,axis=0)-b[ksig,:,:])/(nk-1)

    fig,ax = plt.subplots(1,sharex=True)#,figsize=(6,10))
    fig.subplots_adjust(hspace=0.3)
    loa = np.percentile(totaerr[tmask].flatten(), 5)
    upa = np.percentile(totaerr[tmask].flatten(), 95)
    im = ax.imshow(totaerr,clim=(loa,upa),cmap='binary')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=12) 
    ax.set_title('$\Delta Q = (Q_0+Q_1+Q_3)/3$',fontsize=16)
    ax.set_ylabel('y [pix]',fontsize=16)
    ax.set_xlabel('x [pix]',fontsize=16)
    ax.set(adjustable='box-forced', aspect='equal')
    ax.tick_params(labelsize=12)
    ax.invert_yaxis()   
    fig.savefig(savefile+'-QErrtot.png')
    plt.close(fig)
    mederr = np.median(totaerr[tmask])
    stderr = np.median(np.abs(totaerr[tmask]-mederr))
    print("      Error (Q0+Q1+Q3)/3 median/MAD: %.4e,%.4e" %(mederr,stderr))

    fig,ax = plt.subplots(1,sharex=True)#,figsize=(6,10))
    fig.subplots_adjust(hspace=0.3)
    lob = np.percentile(totberr[tmask].flatten(), 5)
    upb = np.percentile(totberr[tmask].flatten(), 95)
    im = ax.imshow(totberr,clim=(lob,upb),cmap='binary')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=12) 
    ax.set_title('$\Delta U = (U_0+U_1+U_3)/3$',fontsize=16)
    ax.set_ylabel('y [pix]',fontsize=16)
    ax.set_xlabel('x [pix]',fontsize=16)
    ax.set(adjustable='box-forced', aspect='equal')
    ax.tick_params(labelsize=12)
    ax.invert_yaxis()   
    fig.savefig(savefile+'-UErrtot.png')
    plt.close(fig)
    mederr = np.median(totberr[tmask])
    stderr = np.median(np.abs(totberr[tmask]-mederr))
    print("      Error (U0+U1+U3)/3 median/MAD: %.4e,%.4e" %(mederr,stderr))
    
    ## -- Plot power spectrum map
    fs=8
    power = np.sqrt(a**2.0+b**2.0)
    fig,ax = plt.subplots(nk,sharex=True,figsize=(6,10))
    #import pdb;pdb.set_trace()
    fig.subplots_adjust(hspace=0.3)
    for ki in range(0,nk):
        loa = np.percentile(power[ki,tmask].flatten(), 5)
        upa = np.percentile(power[ki,tmask].flatten(), 95)
        im = ax[ki].imshow(power[ki,:,:],clim=(loa,upa),cmap='rainbow')
        divider = make_axes_locatable(ax[ki])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=6)
        cb.ax.tick_params(labelsize=fs-2)
        ax[ki].set_title('$P_k=(U_k^2+Q_k^2)^{1/2}$ at k = %.1f' %(ki),fontsize=fs)
        ax[ki].set_ylabel('y [pix]',fontsize=fs)
        if ki == nk-1:
            ax[ki].set_xlabel('x [pix]',fontsize=fs)
        ax[ki].set(adjustable='box-forced', aspect='equal')
        ax[ki].invert_yaxis()
        ax[ki].tick_params(labelsize=fs-2)
    fig.savefig(savefile+'-Power.png')
    plt.close(fig)

    ## -- Plot Pol error
    ksig = nk/2
    toterr = (np.sum(power,axis=0)-power[ksig,:,:])/(nk-1)
    err = (np.sum(power,axis=0)-power[ksig,:,:]-power[0,:,:])/(nk-2)
       
    fig,ax = plt.subplots(1,sharex=True)#,figsize=(6,10))
    fig.subplots_adjust(hspace=0.3)
    loe = np.percentile(toterr[tmask].flatten(), 5)
    upe = np.percentile(toterr[tmask].flatten(), 95)
    im = ax.imshow(toterr,clim=(loe,upe),cmap='binary')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=12) 
    ax.set_title('$\Delta P = (P_0+P_1+P_3)/3$',fontsize=16)
    ax.set_ylabel('y [pix]',fontsize=16)
    ax.set_xlabel('x [pix]',fontsize=16)
    ax.set(adjustable='box-forced', aspect='equal')
    ax.tick_params(labelsize=12)
    ax.invert_yaxis()   
    fig.savefig(savefile+'-PowerErrtot.png')
    plt.close(fig)
    mederr = np.median(toterr[tmask])
    stderr = np.median(np.abs(toterr[tmask]-mederr))
    print("      Error (P0+P1+P3)/3 median/MAD: %.4e,%.4e" %(mederr,stderr))
        
    fig,ax = plt.subplots(1,sharex=True)#,figsize=(6,10))
    fig.subplots_adjust(hspace=0.3)
    loe = np.percentile(err[tmask].flatten(), 5)
    upe = np.percentile(err[tmask].flatten(), 95)
    im = ax.imshow(err,clim=(loe,upe),cmap='binary')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=12) 
    ax.set_title('$\Delta P = (P_1+P_3)/2$',fontsize=16)
    ax.set_ylabel('y [pix]',fontsize=16)
    ax.set_xlabel('x [pix]',fontsize=16)
    ax.set(adjustable='box-forced', aspect='equal')
    ax.tick_params(labelsize=12)
    ax.invert_yaxis()   
    fig.savefig(savefile+'-PowerErr.png')
    plt.close(fig)
    mederr = np.median(err[tmask])
    stderr = np.median(np.abs(err[tmask]-mederr))
    print("      Error (P1+P3)/2 median/MAD: %.4e,%.4e" %(mederr,stderr))

    ## Plot angle error
    totangerr = 0.5*np.arctan(toterr/power[ksig,:,:])/np.pi*180.
    fig,ax = plt.subplots(1,sharex=True)#,figsize=(6,10))
    fig.subplots_adjust(hspace=0.3)
    loang = np.percentile(totangerr[tmask].flatten(), 5)
    upang = np.percentile(totangerr[tmask].flatten(), 95)
    im = ax.imshow(totangerr,clim=(loang,upang),cmap='binary')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=12) 
    ax.set_title('$\Delta \chi$',fontsize=16)
    ax.set_ylabel('y [pix]',fontsize=16)
    ax.set_xlabel('x [pix]',fontsize=16)
    ax.set(adjustable='box-forced', aspect='equal')
    ax.tick_params(labelsize=12)
    ax.invert_yaxis()   
    fig.savefig(savefile+'-AngErrtot.png')
    plt.close(fig)
    mederr = np.median(totangerr[tmask])
    stderr = np.median(np.abs(totangerr[tmask]-mederr))
    print("      Error (A0+A1+A3)/3 median/MAD: %.4e,%.4e" %(mederr,stderr))
    
    ## PLOT fourier coefficients vs k
    fig,ax = plt.subplots(num,ncols,sharex=True,figsize=(10,14))
    fig.subplots_adjust(hspace=0,wspace=0.3)
    wi = 0.3
    for i in range(0,num):
        afour = a[:,yfin[ind[i]],xfin[ind[i]]]
        erafour = era[:,yfin[ind[i]],xfin[ind[i]]]
        bfour = b[:,yfin[ind[i]],xfin[ind[i]]]
        erbfour = erb[:,yfin[ind[i]],xfin[ind[i]]]
        power = np.sqrt(afour**2.0+bfour**2.0)
        
        ax[i,0].bar(k,np.abs(afour),wi,yerr=erafour)
        ax[i,1].bar(k,np.abs(bfour),wi,yerr=erbfour)
        ax[i,2].bar(k,power,wi,yerr=(afour*erafour+bfour*erbfour)/power)
        #ax[i,0].set_ylim([np.min(afour),np.max(afour)])
        #ax[i,1].set_ylim([np.min(bfour),np.max(bfour)])
        #ax[i,2].set_ylim([np.min(power),np.max(power)])
        ax[i,0].text(0.4,0.1,'Pixels: '+str(xfin[ind[i]])+'-'+str(yfin[ind[i]]),
                     color='green',transform=ax[i,0].transAxes)
        #ax[i,0].ticklabel_format(stype='sci',axis='y',scilimits=(0,0))
        #ax[i,1].ticklabel_format(stype='sci',axis='y',scilimits=(0,0))
        ax[i,0].tick_params(axis='both',labelsize=6)
        ax[i,1].tick_params(axis='both',labelsize=6)
        ax[i,2].tick_params(axis='both',labelsize=6)
    ax[0,2].set_xticks(k)        
    fig.text(0.5, 0.07, 'k', ha='center',fontsize=12)
    fig.text(0.07, 0.5, '$|U_k|$', va='center', rotation='vertical',fontsize=10)
    fig.text(0.35, 0.5, '$|Q_k|$', va='center', rotation='vertical',fontsize=10)
    fig.text(0.63, 0.5, '$(U_k^2+Q_k^2)^{1/2}$', va='center', rotation='vertical',fontsize=10)
    
    plt.savefig(savefile+'-Fourier-k.png')
    plt.close(fig)
