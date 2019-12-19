## THIS FILE CONTAINS THE "FULL" REDUCTION/ANALYSIS OF A GIVEN TARGET FROM ESO/FORS2/IPOL DATA

## This routine calls multiple functions/routines in the file "ESO_IPOL.py". It is an example on
##  how to use these routines but not the only way. 
## It also reads an user-defined input file (default: "standard.input") with various parameters.
## The input file can be changed and the code run such as:
##   "python multi_eso_ipol my.input" (from terminal)  OR
##   "run multi_eso_ipol my.input" (from ipython console)

# -------------------------------------------------------------
##---------------- STRUCTURE OF THE CODE ----------------------
# -------------------------------------------------------------
## The general structure of the code is the following:

##   1) Read input file which, among others, provides the target under investigation
##   2) Prepare folders for reduction/analysis if not already done (see 'prepare_folder' routine)
##   3) Find master_bias if not already done (see 'master_bias' routine)
##   4) Loop filters this target
##      4.1) Calculate a flat image (angle sum) for this filter from the data, if desired
##        5) Loop angles of this target
##          5.1) Read files (see 'read_fits' routine)
##          5.2) Bias corrections, if desired
##          5.3) Cosmic rays corrections, if desired (see 'cosmic_rays' routine)
##          5.4) Separate beams into ordinary/extraordinary
##               (see 'get_strips' and 'separate_beams' routines)
##          5.5) Combine chips 1 and 2 (see 'stick_chips' routine)*
##          5.6) Match stars of ordinary/extraordinary to find offset between the two
##                (see 'find_stars' and 'ebeam_shift') routines*
##          5.7) Find noise of each image
##          POINT SOURCE:
##             5.8) Find centroid of image (see 'inicenter' and 'centroid' routines)
##             5.9) Calculate aperture photometry (see 'aperture_phot' routine)
##             5.10) Calculate PSF photometry (see 'psf_phot' routine)
##          EXTENDED:
##             5.11) Bin images, if desired (see 'bin_image')
##              WARNING: Maybe better to image after polarization calculation
##       6) Analyse strip position (5.4) as a function of angle, if desired
##           (see 'analyse_angstrips' routine)
##       7) Analyse ord/ext matching (5.6) as a function of angle, if desired
##           (see 'analyse_angquadpars' routine)
##       8) Calculate polarization for this filter (see 'polarization' routine)
##       EXTENDED:
##        9) Calculate polarization also for binned images, if desired
##        10) Plot polarization maps (see 'plotpol' routine)
##       POINT SOURCE:
##        11) Calculate polarization for point source from photometry
##        12) Write and plot results (see 'write_photfile' routine)
##   13) Calculate moon polarization model at this time and place (see 'moon_polmap' routine)
##   14) Analyse strip position (5.4) as a function of filter, if desired
##         (see 'analyse_filtstrips' routine)
##   15) Analyse ord/ext matching (5.6) as a function of filter, if desired
##         (see 'analyse_filtquadpars' routine)
##   POINT SOURCE
##    16) Analyse polarization as a function of filter and Serkowski fit
##        (see 'analyse_filtphpol' routine)
##
## *Part 5.5 and 5.6 in principle are not required for point-sources
##  but this codes does it anyway 

from eso_ipol import *


## ------------- GENERAL VARIABLES ----------

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                    
## %%%%%%%%%%%%       USER INPUT   %%%%%%%%%%%%%%%%%%%%%%%%%%%

## See if input file is given
if len(sys.argv) > 1:
    inputfile = sys.argv[1]
else:
    inputfile = home+"/crisp/FORS2-POL/codes/standard.input"

#read_inputfile(inputfile)
ra,dec = None,None
logfile = ''
obj,target,date,biasdate = '','','',''
output = 'reduced'
dobias,dogain,doflat,dofourier = False,False,False,False
dopix,dobin,dophot,dofphot,dochrom = False,False,False,False,False
calcstrips,findstars,docosmic,bkg = False,False,False,False
polbias,biasmethod = False,''
cenpol,instpol = 'None','None'
doradfit,doparfit,inla = False,False,False
galfit,galfitradius = None,None
rad_aper, noise_box = None,15
dcountdy = 1000.0
filters = ['u_HIGH','b_HIGH','v_HIGH','R_SPECIAL','I_BESS','H_Alpha','OII_8000']
comboff,offsets = False,None

for line in open(inputfile,'r'):
    if line[0] == '#': continue
    strdiv0 = line.split('#')[0]
    strdiv = strdiv0.split()
    if len(strdiv) <= 1: continue
    if 'OBJECT' in strdiv[0].upper(): obj = strdiv[1].upper()
    if 'TARGET' in strdiv[0].upper(): target = strdiv[1]
    if 'DATE:' == strdiv[0].upper(): date = strdiv[1]
    if 'BIASDATE:' == strdiv[0].upper(): biasdate = strdiv[1]
    if 'OUTPUT' in strdiv[0].upper(): output = strdiv[1]
    if 'FILTERS' in strdiv[0].upper():
        filters = np.array(strdiv[1].split(','))
    if 'OFFSETS:' == strdiv[0].upper():
        if ('no' not in strdiv[1].lower()) and ('all' not in strdiv[1].lower()):
            offsets = np.array(strdiv[1].split(','),dtype=int)
    if 'COMBOFFSET:' == strdiv[0].upper():
        if strdiv[1].lower() == 'yes': comboff = True
        if strdiv[1].lower() == 'no': comboff = False
    if 'BIAS:' == strdiv[0].upper():
        if strdiv[1].lower() == 'yes': dobias = True
        if strdiv[1].lower() == 'no': dobias = False
    if 'GAIN' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': dogain = True
        if strdiv[1].lower() == 'no': dogain = False
    if 'COSMIC' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': docosmic = True
        if strdiv[1].lower() == 'no': docosmic = False
    if 'FLAT' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': doflat = True
        if strdiv[1].lower() == 'no': doflat = False
    if 'CALCSTRIPS' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': calcstrips = True
        if strdiv[1].lower() == 'no': calcstrips = False
    if 'DCOUNTDY' in strdiv[0].upper(): dcountdy = np.float(strdiv[1])
    if 'NOISE_BOXSIZE' in strdiv[0].upper():
        noise_box = np.int(strdiv[1])
    if 'PIXPOL' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': dopix = True
        if strdiv[1].lower() == 'no': dopix = False
    if 'BINNING' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': dobin = True
        if strdiv[1].lower() == 'no': dobin = False
    if 'BINSIZE' in strdiv[0].upper(): binsize = np.int(strdiv[1])
    if 'BINSIGMA' in strdiv[0].upper(): binsigma = np.float(strdiv[1])        
    if 'CENTER' in strdiv[0].upper():
        center = np.array(strdiv[1].split(','),dtype='int')
        ecenter = center
    if 'STD-PHOTOMETRY' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': dophot = True
        if strdiv[1].lower() == 'no': dophot = False
    if 'APERRADIUS' in strdiv[0].upper():
        if 'perc' in strdiv[1].lower(): rad_aper = strdiv[1].lower()
        elif 'fwhm' in strdiv[1].lower(): rad_aper = strdiv[1].lower()
        else : rad_aper = np.int(strdiv[1])
    if 'FIELD-PHOTOMETRY' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': dofphot = True
        if strdiv[1].lower() == 'no': dofphot = False
    if 'POSFREE' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': posfree = True
        if strdiv[1].lower() == 'no': posfree = False
    if 'DPIX' in strdiv[0].upper(): dpix = np.float(strdiv[1])
    if 'BKG' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': bkg = True
        if strdiv[1].lower() == 'no': bkg = False
    if 'MAXRADIUS' in strdiv[0].upper(): maxrad = np.int(strdiv[1])
    if 'CHROMATISM' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': dochrom = True
        if strdiv[1].lower() == 'no': dochrom = False
    if 'POLBIAS' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': polbias = True
        if strdiv[1].lower() == 'no': polbias = False
    if 'BIASMETHOD' in strdiv[0].upper(): biasmethod = strdiv[1]
    if 'FINDSTARS' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': findstars = True
        if strdiv[1].lower() == 'no': findstars = False    
    if 'FWHM' in strdiv[0].upper(): fwhm = np.float(strdiv[1])
    if 'BBOX' in strdiv[0].upper(): bbox = np.int(strdiv[1])
    if 'THRESHOLD' in strdiv[0].upper(): threshold = np.float(strdiv[1])
    if 'LOGFILE' in strdiv[0].upper(): logfile = strdiv[1]
    if 'CENPOL' in strdiv[0].upper(): cenpol = strdiv[1]
    if 'INSTPOL' in strdiv[0].upper(): instpol = strdiv[1]
    if 'RADFIT' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': doradfit = True
        if strdiv[1].lower() == 'no': doradfit = False
    if 'PARFIT' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': doparfit = True
        if strdiv[1].lower() == 'no': doparfit = False
    if 'INLA' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': inla = True
        if strdiv[1].lower() == 'no': inla = False    
    if 'FOURIER' in strdiv[0].upper():
        if strdiv[1].lower() == 'yes': dofourier = True
        if strdiv[1].lower() == 'no': dofourier = False
    if 'GALFIT:' == strdiv[0].upper():
        if strdiv[1].lower() != 'none': galfit = strdiv[1]
    if 'GALFITRADIUS:' == strdiv[0].upper():
        if strdiv[1].lower() != 'none': galfitradius = np.float(strdiv[1])

if obj == '': print("ERROR: NO OBJECT INPUT")
if target == '': print("ERROR: NO TARGET INPUT")


##LOG FILE
if logfile is not '':
    orig_stdout = sys.stdout
    logf = open(home+"/crisp/FORS2-POL/"+target+"/"+logfile, 'w')
    sys.stdout = logf
    import datetime
    print(datetime.datetime.now())

print("-------------------------------------")
print("FORS2 POLARIZATION REDUCTION PIPELINE")
print("-------------------------------------")
print("")
print("Investigating %s (%s)" %(target,obj))
if date != '': print("                on date: %s" %date)

if date != '': date='/'+date
rawdir = home+"/crisp/FORS2-POL/"+target+"/rawdata/"
datadir = home+"/crisp/FORS2-POL/"+target+date+"/headdata/"
outdir = home+"/crisp/FORS2-POL/"+target+date+"/"+output+"/"

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## -------- 0) CREATE FOLDERS
if not os.path.isfile(datadir+'filemap.dat'):
    files = [file for file in os.listdir(rawdir) if file.endswith('fits')]
    prepare_folder(files,indir=rawdir,outdir=datadir,target=target)#,nocopy=True)
if not os.path.exists(outdir):
    os.makedirs(outdir)

## ------- 1) REFERENCE NON-POL IMAGE
gfitmask = None
if obj == 'GAL':
    refinfo = read_reference(outdir,datadir+'template.dat',rawdir)

    ## GALFIT: get mask
    if refinfo is not None:
        for rfile in refinfo:
            if galfit == 'ell':
                print(" Finding galaxy elliptical isophot")
                gfitmask,galfitradius = galisophot(rfile['file'],center,rfile['ra'],rfile['dec'])  
            elif (galfit == 'circ') & (galfitradius is not None):
                head,img = read_fits(rfile)
                ny,nx = img.shape
                xx,yy = np.meshgrid(np.arange(0,nx),np.arange(0,ny))
                rr = np.sqrt(xx**2.0+yy**2.0)
                gfitmask = (rr > galfitradius)
                
## --Offset info 
offsetinfo = read_offset(datadir+'observation.dat')

##-------- 2) ALL POL IMAGES
polfiles = np.loadtxt(datadir+'filemap.dat',
                      dtype={'names':('file','galaxy','target','ra','dec',
                                      'filter','angle','exptime','mjd','chip','moon'),
                             'formats':('O','O','O','f','f','O','f','f','f8','O','f')})
#sort
polfiles = polfiles[np.isfinite(polfiles['angle'])]
polfiles = polfiles[np.argsort(polfiles['mjd'])]

polfilters = np.unique(polfiles['filter'])
polangles = np.unique(polfiles['angle'])

#%%%% Master Bias and gain
if dogain or dobias:
    if biasdate == '':
        if date == '': biasdate = '/'+np.str(np.int(np.floor(polfiles['mjd'][0])))
        else: biasdate = '/'+date
    mbias1,mbias2 = master_bias(biasdate)
    if dogain: ## NOT FINISHED!
        find_gain(date,mbias1,mbias2)
        sys.exit('stop')

#%%%% Loop over filters 
for f in range(0,len(polfilters)):

    if polfilters[f] not in filters: continue
    if polfilters[f] == 'u_HIGH': pdb.set_trace()   
    
    print("--FILTER: "+polfilters[f])
    fname = polfilters[f]

    ## offset info & check number filter files 
    offset,polfiles = get_offset(offsetinfo,polfiles,polfilters[f],polangles,dir=datadir)
    noffset = len(offset)
    myoffsets = np.arange(noffset) if offsets is None else offsets
 
    ## FLAT: loop filters & offsets
    fpolfiles = polfiles[(polfiles['filter']==polfilters[f])]
    if doflat:
        for i in range(0,noffset):
            if noffset > 1:
                gname = '-'+offset['type'][i]+np.str(i)
            else: gname=''
            print("   -- OFFSET/ITER: %i" %(i))
            
            ffiles1 = fpolfiles[(fpolfiles['chip'] == 'CHIP1') & (fpolfiles['offit'] == i)]
            ffiles2 = fpolfiles[(fpolfiles['chip'] == 'CHIP2') & (fpolfiles['offit'] == i)]

            ## %%%% Polarimetry flat from data itself! This is a flat (per offset) to multiply ebeam
            retflat = data_flat(datadir+ffiles1['file'],datadir+ffiles2['file'],dobias=dobias,
                                savefile=outdir+fname+gname,dobin=dobin,binsize=binsize,
                                binsigma=binsigma,docosmic=docosmic,bias1=mbias1,bias2=mbias2,
                                doplot=True,center=center,ecenter=ecenter)
            flat = retflat['flat']
            if dobin: binflat = retflat['binflat']

    #%%%% Loop over angles
    alldata1,alldata2,allhead1,allhead2 = [],[],[],[]
    allbeam,allebeam,allerbeam,allerebeam = [],[],[],[]
    allbinbeam,allbinebeam,allerbinbeam,allerbinebeam = [],[],[],[]
    allmask,allemask,allbinmask,allbinemask = [],[],[],[]
    allphot,allephot,allerphot,allerephot = [],[],[],[]
    allpsfphot,allpsfephot,allerpsfphot,allerpsfephot = [],[],[],[]
    allfieldphot,allfieldephot,allafieldphot,allafieldephot  = [],[],[],[]
    
    for a in range(0,len(polangles)):

        #if polangles[a] not in angles: continue
        print(" --ANGLE: %f" %(polangles[a]))
        aname = '-ang'+str(polangles[a])

        for i in range(0,noffset):


            if i not in myoffsets: continue
            
            print("   -- OFFSET/ITER: %i" %(i))
            
            files1 = fpolfiles['file'][(fpolfiles['angle'] == polangles[a]) &
                                       (fpolfiles['filter'] == polfilters[f]) &
                                       (fpolfiles['chip'] == 'CHIP1') & (fpolfiles['offit'] == i)]
                                       #(np.isclose(fpolfiles['ra'],offset['ra'][i],rtol=tol)) &
                                       #(np.isclose(fpolfiles['dec'],offset['dec'][i],rtol=tol))]
                                        
            files2 = fpolfiles['file'][(fpolfiles['angle'] == polangles[a]) &
                                       (fpolfiles['filter'] == polfilters[f]) &
                                       (fpolfiles['chip'] == 'CHIP2') & (fpolfiles['offit'] == i)]
                                       #(np.isclose(fpolfiles['ra'],offset['ra'][i],rtol=tol)) &
                                       #(np.isclose(fpolfiles['dec'],offset['dec'][i],rtol=tol))]
        
            if (len(files1) != len(files2) or (len(files1) == 0)):
                print("ERROR: Something wrong in files for filter %s and angle %f"
                      %(filters[f],polangles[a]))
                sys.exit()
                continue
            
            print(" Investigating following files: %s, %s" %(files1,files2))

            ##-- names
            if noffset > 1: gname = '-'+offset['type'][i]+np.str(i)
            else: gname=''
            tname = fname+aname+gname
            
            ##-- Read reference files
            head1,data1 = read_fits(datadir+files1[0])
            head2,data2 = read_fits(datadir+files2[0])

            ##-- Bias
            if dobias:
                data1 = data1-mbias1
                data2 = data2-mbias2
            
            ##-- Do cosmic ray removal
            if docosmic:
                data1 = cosmic_rays(data1,head1,outfile=outdir+tname+'-chip1')
                data2 = cosmic_rays(data2,head2,outfile=outdir+tname+'-chip2')
                tname += '-cosmic'
                
            ##-- Separate beam/ebeam
            beam1,ebeam1,beam2,ebeam2 = separate_beams(data1,data2,default=(not calcstrips),
                                                       dcountdy=dcountdy,
                                                       savefile1=outdir+tname,savefile2=outdir+tname)
            ##-- Mask1/2
            #mask1, emask1 = (beam1 > 0), (ebeam1 > 0)
            #mask2, emask2 = (beam2 > 0), (ebeam2 > 0)
            
            ##-- Stick chips
            beam = stick_chips(beam1,beam2,head1,head2,savefile=outdir+tname+'-obeam')
            ebeam = stick_chips(ebeam1,ebeam2,head1,head2,savefile=outdir+tname+'-ebeam')

            ##-- Masks 
            mask, emask = (beam > 0), (ebeam > 0)
  
            ##-- Correct ebeam/obeam shift
            #old_ebeam = ebeam
            ydiff = find_shift(beam,ebeam,fwhm=fwhm,threshold=threshold,
                               savefile=outdir+tname+'-chip12',default=(not findstars))
            ebeam,emask = ebeam_shift(ebeam,ydiff,mask=emask,savefile=outdir+tname+'-ebeam-merged')
               
            ##-- Test of flux conservation:
            #ydiff = find_stars(old_ebeam,ebeam,fwhm=fwhm,threshold=threshold,
            #                   savefile=outdir+tname+'-shiftchip12',default=(not findstars))
            
            ##-- Noise maps
            errbeam = noise(beam,mask,noise_box,sigmaclip=binsigma,
                            savefile=outdir+tname+'-obeam-merged')
            errebeam = noise(ebeam,emask,noise_box,sigmaclip=binsigma,
                             savefile=outdir+tname+'-ebeam-merged-shifted')

            ##-- Bin intensity images (after separating o/e and sticking!) 
            if dobin:
                binbeam,errbinbeam = bin_image(beam,mask,radpix=binsize,sigmaclip=binsigma,fullbin=True,
                                               center=center,savefile=outdir+tname+'-obeam-merged')
                binebeam,errbinebeam = bin_image(ebeam,emask,radpix=binsize,sigmaclip=binsigma,fullbin=True,
                                           center=ecenter,savefile=outdir+tname+'-ebeam-merged-shifted')
                binmask,binemask = (binbeam > 0),(binbeam > 0)

                #Binned flat correction
                if doflat: binebeam *= binflat

                allbinbeam.append(binbeam);allbinebeam.append(binebeam)
                allerbinbeam.append(errbinbeam);allerbinebeam.append(errbinebeam)
                allbinmask.append(binmask);allbinemask.append(binemask)
                del binbeam,binebeam,errbinbeam,errbinebeam,binmask,binemask
    
            ##-- Flat correction (pixel by pixel)
            if doflat:
                ebeam *= flat
                #save for fieldphot
                fits.writeto(outdir+tname+'-ebeam-merged-shifted-flat.fits',ebeam,clobber=True)
                
            ##-- Photometry (point sources)
            if dophot:
                
                ##-- Center point source
                print("      Center: "+str(center))
                center = centroid(beam,mask,inicenter=center,radpix=maxrad)
                ecenter = centroid(ebeam,emask,inicenter=ecenter,radpix=maxrad)

                ## -- Aperture
                phot,photerr,trad_aper = aperture_phot(beam,errbeam,mask,center,radpix=rad_aper,
                                                       savefile=outdir+tname+'-obeam')
                ephot,ephoterr,trad_aper = aperture_phot(ebeam,errebeam,emask,ecenter,radpix=trad_aper,
                                                         savefile=outdir+tname+'-ebeam')
                allphot.append(phot);allephot.append(ephot)
                allerphot.append(photerr);allerephot.append(ephoterr)

                ## -- PSF
                phot,photerr = psf_phot(beam,errbeam,mask,center,savefile=outdir+tname+'-obeam')
                ephot,ephoterr = psf_phot(ebeam,errebeam,emask,center,savefile=outdir+tname+'-ebeam') 
                allpsfphot.append(phot);allpsfephot.append(ephot)
                allerpsfphot.append(photerr);allerpsfephot.append(ephoterr)
                del phot,ephot,photerr,ephoterr

             ##-- Photometry (field point sources)
            if dofphot:   
              
                ffiles1 = datadir+fpolfiles['file'][(fpolfiles['chip'] == 'CHIP1') & 
                                            (fpolfiles['ra'] == offset['ra'][i]) &
                                            (fpolfiles['dec'] == offset['dec'][i])]
                ffiles2 = datadir+fpolfiles['file'][(fpolfiles['chip'] == 'CHIP2') & 
                                            (fpolfiles['ra'] == offset['ra'][i]) &
                                            (fpolfiles['dec'] == offset['dec'][i])]
                fphot = field_psfphot(beam,errbeam,mask,files1=ffiles1,files2=ffiles2,
                                      dobias=dobias,docosmic=docosmic,fwhm=fwhm,bbox=bbox,bkg=bkg,
                                      savefile=outdir+tname+'-obeam',threshold=threshold,bias1=mbias1,
                                      bias2=mbias2,sumfile=outdir+fname+gname,posfree=posfree)
                fephot = field_psfphot(ebeam,errebeam,emask,files1=ffiles1,files2=ffiles2,
                                       dobias=dobias,docosmic=docosmic,fwhm=fwhm,bbox=bbox,bkg=bkg,
                                       threshold=threshold,savefile=outdir+tname+'-ebeam',
                                       bias1=mbias1,bias2=mbias2,
                                       sumfile=outdir+fname+gname,posfree=posfree)
                faphot = field_apphot(beam,errbeam,mask,files1=ffiles1,files2=ffiles2,bkg=bkg,
                                      dobias=dobias,docosmic=docosmic,fwhm=fwhm,threshold=threshold,
                                      bias1=mbias1,bias2=mbias2,radpix=rad_aper,
                                      savefile=outdir+tname+'-obeam',sumfile=outdir+fname+gname)
                faephot = field_apphot(ebeam,errebeam,emask,files1=ffiles1,files2=ffiles2,bkg=bkg,
                                       dobias=dobias,bias1=mbias1,bias2=mbias2,docosmic=docosmic,
                                       fwhm=fwhm,threshold=threshold,radpix=rad_aper,
                                       savefile=outdir+tname+'-ebeam',sumfile=outdir+fname+gname)
                allfieldphot.append(fphot);allfieldephot.append(fephot)
                allafieldphot.append(faphot);allafieldephot.append(faephot)
                del fphot,fephot,faphot,faephot
                
            ##-- change beam for ebeam if ang>45
            #if angles[a] >=45:
            #    temp = beam
            #    beam = ebeam
            #    ebeam = temp
            #    os.system("mv "+outdir+tname+"-obeam-merged.fits "+outdir+tname+"-temp.fits")
            #    os.system("mv "+outdir+tname+"-ebeam-merged.fits "+outdir+tname+"-obeam-merged.fits")
            #    os.system("mv "+outdir+tname+"-temp.fits "+outdir+tname+"-ebeam-merged.fits")
            
            #plt.imshow(np.arcsinh(beam))
            #plt.show()

            
            ## -- ALIGN SCIENCE AND REF IMAGES
            #new_beam = manualalign(beam,ref_data,savefile=outdir+tname+'-obeam-merged')
            #new_ebeam = manualalign(ebeam,ref_data,savefile=outdir+tname+'-ebeam-merged')
            
            allbeam.append(beam);allebeam.append(ebeam)
            allerbeam.append(errbeam);allerebeam.append(errebeam)
            allmask.append(mask);allemask.append(emask)
            del beam,ebeam,errbeam,errebeam,mask,emask
                
        #%%%% End loop offsets

        ## -- OFFSET JOIN?

    #%%%% End loop angles

    ## -- ANALYSIS OVER ANGLES
    if calcstrips:
        analyse_angstrips(polfilters[f],polangles,outdir,obj,offset)
    if findstars:
        analyse_angquadpars(polfilters[f],polangles,outdir,obj,offset,fwhm=fwhm,threshold=threshold)
    
        
    ## -- POLARIZATION
    print(" ")
    print("Calculating polarization from following angles: "+str(polangles))
        
    # arrays
    allbeam,allebeam = np.asarray(allbeam),np.asarray(allebeam)
    allerbeam,allerebeam = np.asarray(allerbeam),np.asarray(allerebeam)
    allmask,allemask = np.asarray(allmask),np.asarray(allemask)
    if dobin:
        allbinbeam,allbinebeam = np.asarray(allbinbeam),np.asarray(allbinebeam)
        allerbinbeam,allerbinebeam = np.asarray(allerbinbeam),np.asarray(allerbinebeam)
        allbinmask,allbinemask = np.asarray(allbinmask),np.asarray(allbinemask)
    if dophot:
        allphot,allephot = np.asarray(allphot),np.asarray(allephot)
        allerphot,allerephot = np.asarray(allerphot),np.asarray(allerephot)
        allpsfphot,allpsfephot = np.asarray(allpsfphot),np.asarray(allpsfephot)
        allerpsfphot,allerpsfephot = np.asarray(allerpsfphot),np.asarray(allerpsfephot)
    if dofphot:
        allfieldphot,allfieldephot = np.asarray(allfieldphot),np.asarray(allfieldephot)
        allafieldphot,allafieldephot = np.asarray(allafieldphot),np.asarray(allafieldephot)

    ##loop offsets
    nmyoffsets,ii = len(myoffsets),-1
    for i in range(0,noffset):
        
        if i not in myoffsets: continue
        print("   -- OFFSET/ITER: %i" %(i))
        ii += 1
        
        ##name
        fname = polfilters[f]
        if noffset > 1: gname = '-'+offset['type'][i]+np.str(i)
        else: gname=''
        if dobin: bname = '-bin'+str(binsize)+'pix'+str(binsigma)+'sig'
        cname = '-cosmic' if docosmic else ''
        tname = fname+gname+cname
   
        ##pick indices
        ind = np.arange(ii,len(polangles)*nmyoffsets,nmyoffsets,dtype=np.int)

        ##fitmask
        fitmask = None
        if gfitmask is not None:
            fitmask = align(gfitmask,refinfo['ra'],refinfo['dec'],offset['ra'][i],offset['dec'][i],fill_val=False)
                 
        ## -- Pixel polarization
        if dopix:

            print("   -----> Pixel polarization-")
            ##Stokes parameters
            stok = stokes(allbeam[ind],allebeam[ind],savefile=outdir+tname,
                          mask=allmask[ind],emask=allemask[ind],
                          errbeam=allerbeam[ind],errebeam=allerebeam[ind])
            Q,U,erQ,erU = stok['Q'],stok['U'],stok['erQ'],stok['erU']
            
            pol0,angle0 = QUpolarization(Q,U,polfilters[f])
               
            ##Plot and correct Stokes for center Q0/U0 and field instpol
            Q,U,QUcorr = QUcorrect(Q,U,savefile=outdir+tname,parfit=doparfit,corr=cenpol,#'med','cen'
                                   errQ=erQ,errU=erU,center=center,fitmask=fitmask,
                                   fcorr=instpol,inla=inla,filt=fname)

            ##Polarization from Q/U & error (first without central Q0/U0 corrections & without fieldcorr)
            if cenpol != 'None' or instpol != 'None':
                polraw,angleraw = QUpolarization(QUcorr['Qraw'],QUcorr['Uraw'],polfilters[f])
                plotpol(polraw,angleraw,center=center,savefile=outdir+tname,
                        image=allbeam[ind[0]]+allebeam[ind[0]],fitradius=galfitradius)
            if cenpol != 'None':
                pol0,angle0 = QUpolarization(QUcorr['Qcencorr'],QUcorr['Ucencorr'],polfilters[f])
                plotpol(pol0,angle0,center=center,savefile=outdir+tname+'-corr'+cenpol,
                        image=allbeam[ind[0]]+allebeam[ind[0]],fitradius=galfitradius)
            pol,angle = QUpolarization(Q,U,polfilters[f],errQ=erQ,errU=erU,
                                       savefile=outdir+tname+QUcorr['corrname'])
            pol,erpol,erangle = erpolarization(allbeam[ind],allebeam[ind],pol,angle,
                                               savefile=outdir+tname+QUcorr['corrname'],
                                               mask=allmask[ind],emask=allemask[ind],
                                               bias=polbias,method=biasmethod,
                                               errbeam=allerbeam[ind],errebeam=allerebeam[ind])

            ##Plot polarization
            plotpol(pol,angle,center=center,erpol=erpol,erangle=erangle,savefile=outdir+tname+QUcorr['corrname'],
                    image=allbeam[ind[0]]+allebeam[ind[0]],fitradius=galfitradius)

            ##Plot pol/ang vs radius
            radius_dependence(pol,angle,outdir+tname+QUcorr['corrname'],radfit=doradfit,parfit=doparfit,
                              inla=inla,erpol=erpol,filt=polfilters[f])#,center=center)
        
            ##Plot flux vs HWP angle
            if dofourier:
                flux_angle(allbeam[ind],allebeam[ind],pol0,angle0,savefile=outdir+tname,
                           #fit=False,mask=allmask[ind],emask=allemask[ind],
                           corr=cenpol,errbeam=allerbeam[ind],errebeam=allerebeam[ind])

            ##Polarizaton from Fourier fit
            #pol2,angle2 = Fpolarization(outdir+tname,polfilters[f])
            #compare_polangle(pol0,angle0,pol2,angle2,outdir+tname)      
            ##Plot Fourier polarization
            #plotpol(pol2,angle2,center=center,savefile=outdir+tname+'-F',image=allbeam[ind[0]]+allebeam[ind[0]])

        
        ## --- Bin polarization
        if dobin:

            print("   -----> Binned polarization-")

            ## From binned intensity maps: Stokes & Pol
            binstok = stokes(allbinbeam[ind],allbinebeam[ind],
                             #mask=allbinmask[ind],emask=allbinemask[ind],
                             errbeam=allerbinbeam[ind],errebeam=allerbinebeam[ind],
                             savefile=outdir+tname+bname)
            binQ,binU,erbinQ,erbinU = binstok['Q'],binstok['U'],binstok['erQ'],binstok['erU']

            ## Plot and correct Stokes for center Q0/U0 and field instpol
            binQ,binU,binQUcorr = QUcorrect(binQ,binU,savefile=outdir+tname+bname,scatter=True,
                                            inla=inla,center=center,errQ=erbinQ,errU=erbinU,
                                            fitmask=fitmask,corr=cenpol,#'med',
                                            parfit=doparfit,fcorr=instpol,filt=fname)#'cen'Q0=Qc,U0=Uc,
            ##Polarization from Q/U & error (first without central Q0/U0 corrections & without fieldcorr)
            if cenpol != 'None' or instpol != 'None':
                binpolraw,binangleraw = QUpolarization(binQUcorr['Qraw'],binQUcorr['Uraw'],polfilters[f])
                plotpol(binpolraw,binangleraw,step=binsize,image=allbeam[ind[0]]+allebeam[ind[0]],
                        center=center,savefile=outdir+tname+bname,fitradius=galfitradius)
            if cenpol != 'None':
                binpol0,binangle0 = QUpolarization(binQUcorr['Qcencorr'],binQUcorr['Ucencorr'],polfilters[f])
                plotpol(binpol0,binangle0,step=binsize,image=allbeam[ind[0]]+allebeam[ind[0]],
                        center=center,savefile=outdir+tname+bname+'-corr'+cenpol,fitradius=galfitradius)
            binpol,binangle = QUpolarization(binQ,binU,polfilters[f],errQ=erbinQ,errU=erbinU,
                                             savefile=outdir+tname+bname+binQUcorr['corrname'])
            binpol,erbinpol,erbinangle = erpolarization(allbinbeam[ind],allbinebeam[ind],
                                                        binpol,binangle,bias=polbias,method=biasmethod,
                                                        savefile=outdir+tname+bname+binQUcorr['corrname'],
                                                        mask=allbinmask[ind],emask=allbinemask[ind],
                                                        errbeam=allerbinbeam[ind],errebeam=allerbinebeam[ind])
            
            #plot flux vs HWP angle
            if dofourier:
                flux_angle(allbinbeam[ind],allbinebeam[ind],binpol0,binangle0,savefile=outdir+tname+bname,
                           #fit=False,mask=allbinmask[ind],emask=allbinemask[ind],
                           corr=cenpol,errbeam=allerbinbeam[ind],errebeam=allerbinebeam[ind])
            
            ## BEFORE: Bin Q and U instead of intensity maps (not right!)
                        
            #plot bin polarization
            plotpol(binpol,binangle,erpol=erbinpol,erangle=erbinangle,fitradius=galfitradius,
                    step=binsize,image=allbeam[ind[0]]+allebeam[ind[0]],
                    center=center,savefile=outdir+tname+bname+binQUcorr['corrname'])

            #plot bin polarization vs radius
            radius_dependence(binpol,binangle,outdir+tname+bname+binQUcorr['corrname'],radfit=doradfit,parfit=doparfit,
                              erpol=erbinpol,inla=inla,filt=polfilters[f],scatter=True)#,center=center)

            ##Polarizaton from Fourier fit
            #binpol2,binangle2 = Fpolarization(outdir+tname+bname,polfilters[f])
            #compare_polangle(binpol0,binangle0,binpol2,binangle2,outdir+tname+bname) 
           
    
        #polarization from point source photometry
        if dophot:

            print("   -----> Star (photometry) polarization-")
            #aperture
            apstok  = stokes(allphot[ind],allephot[ind],
                                         errbeam=allerphot[ind],errebeam=allerephot[ind])
            apQ,apU,erapQ,erapU = apstok['Q'],apstok['U'],apstok['erQ'],apstok['erU']

            apQ,apU,apQUcorr = QUcorrect(apQ,apU,savefile=outdir+tname+'_aper',
                                         corr='given',Q0=Qc,U0=Uc,errQ=erapQ,errU=erapU)
            ###correction depends on position!!
            appol,apangle = QUpolarization(apQ,apU,polfilters[f],errQ=erapQ,errU=erapU,
                                         chrom=dochrom)
            appol,erappol,erapangle = erpolarization(allphot[ind],allephot[ind],
                                                     appol,apangle,bias=polbias,method=biasmethod,
                                                     errbeam=allerphot[ind],errebeam=allerephot[ind])
            apstok['Q'],apstok['erQ'],apstok['U'],apstok['erU'] = apQ,erapQ,apU,erapU
            apstok['pol'],apstok['angle'] = appol,apangle
            apstok['erpol'],apstok['erangle'] = erappol,erapangle
            
            write_photfile(outdir+tname+'_aper',allphot[ind],allephot[ind],
                           allerphot[ind],allerephot[ind],apstok)
            print("Iter/offset %i: APER-phot polarization %f and angle %f"
                  %(i,appol,apangle))
               
            #psf
            psfstok = stokes(allpsfphot[ind],allpsfephot[ind],
                             errbeam=allerpsfphot[ind],errebeam=allerpsfephot[ind])
            psfQ,psfU,erpsfQ,erpsfU = psfstok['Q'],psfstok['U'],psfstok['erQ'],psfstok['erU']
            psfQ,psfU,psfQUcorr = QUcorrect(psfQ,psfU,savefile=outdir+tname+'_psf',
                                              corr='given',Q0=Qc,U0=Uc,errQ=erpsfQ,errU=erpsfU)
            psfpol,psfangle = QUpolarization(psfQ,psfU,polfilters[f],chrom=dochrom,
                                           errQ=erpsfQ,errU=erpsfU)
            psfpol,erpsfpol,erpsfangle = erpolarization(allpsfphot[ind],allpsfephot[ind],
                                                        psfpol,psfangle,method=biasmethod,
                                                        errbeam=allerpsfphot[ind],bias=polbias,
                                                        errebeam=allerpsfephot[ind])
            psfstok['Q'],psfstok['erQ'],psfstok['U'],psfstok['erU'] = psfQ,erpsfQ,psfU,erpsfU
            psfstok['pol'],psfstok['angle'] = psfpol,psfangle
            psfstok['erpol'],psfstok['erangle'] = erpsfpol,erpsfangle
            
            write_photfile(outdir+tname+'_psf',allpsfphot[ind],allpsfephot[ind],
                           allerpsfphot[ind],allerpsfephot[ind],psfstok)
            print("Iter/offset %i: PSF-phot polarization %f and angle %f"
                      %(i,psfpol,psfangle))

        #polarization from field source photometry
        if dofphot:
            
            
            ## Match sources PSF
            xsrc,ysrc,fphot,fephot,erfphot,erfephot = match_sources(allfieldphot[ind],
                                                                    allfieldephot[ind],#signois=20.0,
                                                                    dpix=dpix,posfree=posfree,
                                                                    savefile=outdir+tname+'-psffield')

            ## Match sources AP 
            axsrc,aysrc,faphot,faephot,erfaphot,erfaephot = match_sources(allafieldphot[ind],
                                                                          allafieldephot[ind],
                                                                          dpix=dpix,aper=True,
                                                                          savefile=outdir+tname+'-apfield')
            ## Match AP/PSF sources (although they're all the same!)


                     
            
            ## PSF vs AP flux compare & match in flux
            photcompare = False
            if photcompare:
                pamask = compare_flux(fphot,fephot,faphot,faephot,outdir+tname+'-field',maxdiff=0.2)
                xsrc,ysrc,axsrc,aysrc = xsrc[pamask],ysrc[pamask],axsrc[pamask],aysrc[pamask] 
                sh = (len(fphot[:,0]),len(xsrc),1)
                fphot,fephot = fphot[:,pamask].reshape(sh),fephot[:,pamask].reshape(sh)
                erfphot,erfephot = erfphot[:,pamask].reshape(sh),erfephot[:,pamask].reshape(sh)
                faphot,faephot = faphot[:,pamask].reshape(sh),faephot[:,pamask].reshape(sh)
                erfaphot,erfaephot = erfaphot[:,pamask].reshape(sh),erfaephot[:,pamask].reshape(sh)

            
            print("   -----> Field star (PSF photometry) polarization-")
            
            ## Polarization of field stars
            fstok = stokes(fphot,fephot,errbeam=erfphot,errebeam=erfephot,
                           #mask=pamask,emask=pamask,
                           savefile=outdir+tname+'-psffield')
            fQ,fU,erfQ,erfU = fstok['Q'],fstok['U'],fstok['erQ'],fstok['erU']
            fQ,fU,fQUcorr = QUcorrect(fQ,fU,savefile=outdir+tname+'-psffield',center=center,
                                      scatter=True,corr=cenpol,errQ=erfQ,errU=erfU,#Q0=binQc,U0=binUc,
                                      parfit=doparfit,fcorr=instpol,filt=fname,x=xsrc,y=ysrc)
            

            if cenpol != 'None' or instpol != 'None':
                ## Bin stars (Raw QU) and plot
                fQrawmap,erfQrawmap = bin_points(xsrc,ysrc, fQUcorr['Qraw'],fullbin=True,
                                                 center=center,savefile=outdir+tname+'-psffield-binQ')
                fUrawmap,erfUrawmap = bin_points(xsrc,ysrc, fQUcorr['Uraw'],fullbin=True,
                                                 center=center,savefile=outdir+tname+'-psffield-binU')
                plotstokes(fQrawmap,fUrawmap,savefile=outdir+tname+'-psffield-bin',
                           parfit=doparfit,scatter=True,center=center)
                           
                ## Raw Pol
                fpolraw,fangleraw = QUpolarization(fQUcorr['Qraw'],fQUcorr['Uraw'],polfilters[f],
                                                   savefile=outdir+tname+'-psffield')
                fpolraw,erfpolraw,erfangleraw = erpolarization(fphot,fephot,fpolraw,fangleraw,
                                                       errbeam=erfphot,errebeam=erfephot,bias=polbias,
                                                       method=biasmethod,savefile=outdir+tname+'-psffield')
                ## Plot raw polarization of field stars
                xyplotpol(xsrc,ysrc,fpolraw,fangleraw,center=center,#mask=pamask.reshape(-1),
                          savefile=outdir+tname+'-psffield')
                ## Bin stars (pol) and plot
                fpolrawmap,erfpolrawmap = bin_points(xsrc,ysrc,fpolraw,fullbin=True,center=center,#mask=pamask,
                                                     savefile=outdir+tname+'-psffield-binpol')
                fanglerawmap,erfanglerawmap = bin_points(xsrc,ysrc,fangleraw,fullbin=True,center=center,#mask=pamask,
                                                         savefile=outdir+tname+'-psffield-binangle')
                plotpol(fpolrawmap,fanglerawmap,erpol=erfpolrawmap,erangle=erfanglerawmap,
                        step=40,center=center,savefile=outdir+tname+'-psffield-bin')
            if cenpol != 'None':
                ## Bin stars (QU) and plot
                fQ0map,erfQ0map = bin_points(xsrc,ysrc,fQUcorr['Qcencorr'],fullbin=True,
                                             center=center,savefile=outdir+tname+'-psffield-binQ'+'-corr'+cenpol)
                fU0map,erfU0map = bin_points(xsrc,ysrc,fQUcorr['Ucencorr'],fullbin=True,
                                             center=center,savefile=outdir+tname+'-psffield-binU'+'-corr'+cenpol)
                plotstokes(fQ0map,fU0map,savefile=outdir+tname+'-psffield-bin'+'-corr'+cenpol,
                           parfit=doparfit,scatter=True,center=center)
                
                ## Pol
                fpol0,fangle0 = QUpolarization(fQUcorr['Qcencorr'],fQUcorr['Ucencorr'],polfilters[f],
                                               savefile=outdir+tname+'-psffield'+'-corr'+cenpol)
                fpol0,erfpol0,erfangle0 = erpolarization(fphot,fephot,fpol0,fangle0,bias=polbias,
                                                         errbeam=erfphot,errebeam=erfephot,method=biasmethod,
                                                         savefile=outdir+tname+'-psffield'+'-corr'+cenpol)

                ##Plot polarization of field stars 
                xyplotpol(xsrc,ysrc,fpol0,fangle0,center=center,#mask=pamask.reshape(-1),
                          savefile=outdir+tname+'-psffield'+'-corr'+cenpol)
                ## Bin stars (Pol) and plot
                fpol0map,erfpol0map = bin_points(xsrc,ysrc,fpol0,fullbin=True,center=center,#mask=pamask,
                                                     savefile=outdir+tname+'-psffield-binpol'+'-corr'+cenpol)
                fangle0map,erfangle0map = bin_points(xsrc,ysrc,fangle0,fullbin=True,center=center,#mask=pamask,
                                                     savefile=outdir+tname+'-psffield-binangle'+'-corr'+cenpol)
                plotpol(fpol0map,fangle0map,erpol=erfpol0map,erangle=erfangle0map,
                        step=40,center=center,savefile=outdir+tname+'-psffield-bin'+'-corr'+cenpol)

            ## Bin stars (QU)
            fQmap,erfQmap = bin_points(xsrc,ysrc,fQ,fullbin=True,center=center,#mask=pamask,
                                       savefile=outdir+tname+'-psffield-binQ'+fQUcorr['corrname'])
            fUmap,erfUmap = bin_points(xsrc,ysrc,fU,fullbin=True,center=center,#mask=pamask,
                                       savefile=outdir+tname+'-psffield-binU'+fQUcorr['corrname'])
            plotstokes(fQmap,fUmap,savefile=outdir+tname+'-psffield-bin'+fQUcorr['corrname'],
                       parfit=doparfit,scatter=True,center=center)
                            
            ## Final Star Polarization
            fpol,fangle = QUpolarization(fQ,fU,polfilters[f],errQ=erfQ,errU=erfU,chrom=dochrom,
                                         #mask=pamask,emask=pamask,
                                         savefile=outdir+tname+'-psffield'+fQUcorr['corrname'])
                                         ##OJO: name of fwhm?... 
            fpol,erfpol,erfangle = erpolarization(fphot,fephot,fpol,fangle,bias=polbias,
                                             #mask=pamask,emask=pamask,
                                             errbeam=erfphot,errebeam=erfephot,method=biasmethod,
                                             savefile=outdir+tname+'-psffield'+fQUcorr['corrname'])
            
            ## Plot polarization of field stars
            xyplotpol(xsrc,ysrc,fpol,fangle,center=center,#mask=pamask.reshape(-1),
                      savefile=outdir+tname+'-psffield'+fQUcorr['corrname'])
            
            ## Bin stars (pol) and plot
            fpolmap,erfpolmap = bin_points(xsrc,ysrc,fpol,fullbin=True,center=center,#mask=pamask,
                                           savefile=outdir+tname+'-psffield-binpol'+fQUcorr['corrname'])
            fanglemap,erfanglemap = bin_points(xsrc,ysrc,fangle,fullbin=True,center=center,#mask=pamask,
                                               savefile=outdir+tname+'-psffield-binangle'+fQUcorr['corrname'])
            plotpol(fpolmap,fanglemap,erpol=erfpolmap,erangle=erfanglemap,
                    step=40,center=center,savefile=outdir+tname+'-psffield-bin'+fQUcorr['corrname'])
            radius_dependence(fpolmap,fanglemap,outdir+tname+'-psffield-bin'+fQUcorr['corrname'],scatter=True,
                              inla=inla,radfit=doradfit,parfit=doparfit,erpol=erfpolmap,filt=polfilters[f])#,center=center)

            print("   -----> Field star (AP photometry) polarization-")
                
      
            ## Polarization of field stars
            fastok= stokes(faphot,faephot,errbeam=erfaphot,errebeam=erfaephot,
                           savefile=outdir+tname+'-apfield')#mask=pamask,emask=pamask,
            faQ,faU,erfaQ,erfaU = fastok['Q'],fastok['U'],fastok['erQ'],fastok['erU']
            faQ,faU,faQUcorr = QUcorrect(faQ,faU,savefile=outdir+tname+'-apfield',center=center,
                                          scatter=True,corr=cenpol,errQ=erfaQ,errU=erfaU,#Q0=binQc,U0=binUc,
                                          parfit=doparfit,fcorr=instpol,filt=fname,x=axsrc,y=aysrc)
            if cenpol != 'None' or instpol != 'None':
                ## Bin stars (Raw QU) and plot
                faQrawmap,erfaQrawmap = bin_points(axsrc,aysrc,faQUcorr['Qraw'],fullbin=True,
                                                   center=center,savefile=outdir+tname+'-apfield-binQ')
                faUrawmap,erfaUrawmap = bin_points(axsrc,aysrc,faQUcorr['Uraw'],fullbin=True,
                                                   center=center,savefile=outdir+tname+'-apfield-binU')
                plotstokes(faQrawmap,faUrawmap,savefile=outdir+tname+'-apfield-bin',
                           parfit=doparfit,scatter=True,center=center)
                ## Raw Pol
                fapolraw,faangleraw = QUpolarization(faQUcorr['Qraw'],faQUcorr['Uraw'],polfilters[f],
                                                     savefile=outdir+tname+'-apfield')
                fapolraw,erfapolraw,erfaangleraw = erpolarization(faphot,faephot,fapolraw,faangleraw,
                                                                  errbeam=erfaphot,errebeam=erfaephot,
                                                                  bias=polbias,method=biasmethod,
                                                                  savefile=outdir+tname+'-apfield')
                ## Plot raw polarization of field stars
                xyplotpol(axsrc,aysrc,fapolraw,faangleraw,#mask=pamask.reshape(-1),
                      center=center,savefile=outdir+tname+'-apfield')
                ## Bin stars (pol) and plot
                fapolrawmap,erfapolrawmap = bin_points(axsrc,aysrc,fapolraw,fullbin=True,center=center,#mask=pamask,
                                                       savefile=outdir+tname+'-apfield-binpol')
                faanglerawmap,erfaanglerawmap = bin_points(axsrc,aysrc,faangleraw,fullbin=True,center=center,#mask=pamask,
                                                           savefile=outdir+tname+'-apfield-binangle')
                plotpol(fapolrawmap,faanglerawmap,erpol=erfapolrawmap,erangle=erfaanglerawmap,
                        step=40,center=center,savefile=outdir+tname+'-apfield-bin')
            if cenpol != 'None':
                ## Bin stars (cen QU) and plot
                faQ0map,erfaQ0map = bin_points(axsrc,aysrc,faQUcorr['Qcencorr'],fullbin=True,
                                               center=center,savefile=outdir+tname+'-apfield-binQ'+'-corr'+cenpol)
                faU0map,erfaU0map = bin_points(axsrc,aysrc,faQUcorr['Ucencorr'],fullbin=True,
                                               center=center,savefile=outdir+tname+'-apfield-binU'+'-corr'+cenpol)
                plotstokes(faQ0map,faU0map,savefile=outdir+tname+'-apfield-bin'+'-corr'+cenpol,
                           parfit=doparfit,scatter=True,center=center)
                ## cen Pol
                fapol0,faangle0 = QUpolarization(faQUcorr['Qcencorr'],faQUcorr['Ucencorr'],polfilters[f],
                                                 savefile=outdir+tname+'-apfield'+'-corr'+cenpol)
                fapol0,erfapol0,erfaangle0 = erpolarization(faphot,faephot,fapol0,faangle0,method=biasmethod,
                                                     errbeam=erfaphot,errebeam=erfaephot,bias=polbias,
                                                     savefile=outdir+tname+'-apfield'+'-corr'+cenpol)
                ## Plot cen polarization of field stars
                xyplotpol(axsrc,aysrc,fapol0,faangle0,#mask=pamask.reshape(-1),
                          center=center,savefile=outdir+tname+'-apfield'+'-corr'+cenpol)
                ## Bin stars (pol) and plot
                fapol0map,erfapol0map = bin_points(axsrc,aysrc,fapol0,fullbin=True,center=center,#mask=pamask,
                                                   savefile=outdir+tname+'-apfield-binpol'+'-corr'+cenpol)
                faangle0map,erfaangle0map = bin_points(axsrc,aysrc,faangle0,fullbin=True,center=center,#mask=pamask,
                                                       savefile=outdir+tname+'-apfield-binangle'+'-corr'+cenpol)
                plotpol(fapol0map,faangle0map,erpol=erfapol0map,erangle=erfaangle0map,
                        step=40,center=center,savefile=outdir+tname+'-apfield-bin'+'-corr'+cenpol)


            ## Bin stars (QU)
            faQmap,erfaQmap = bin_points(axsrc,aysrc,faQ,fullbin=True,center=center,#mask=pamask,
                                         savefile=outdir+tname+'-apfield-binQ'+fQUcorr['corrname'])
            faUmap,erfaUmap = bin_points(axsrc,aysrc,faU,fullbin=True,center=center,#mask=pamask,
                                         savefile=outdir+tname+'-apfield-binU'+fQUcorr['corrname'])
            plotstokes(faQmap,faUmap,savefile=outdir+tname+'-apfield-bin'+fQUcorr['corrname'],
                       parfit=doparfit,scatter=True,center=center)
            
            ## Final AP polar
            fapol,faangle = QUpolarization(faQ,faU,polfilters[f],errQ=erfaQ,errU=erfaU,chrom=dochrom,
                                           #mask=pamask,emask=pamask,
                                           savefile=outdir+tname+'-apfield'+faQUcorr['corrname'])
            fapol,erfapol,erfaangle = erpolarization(faphot,faephot,fapol,faangle,bias=polbias,
                                               #mask=pamask,emask=pamask,
                                               errbeam=erfaphot,errebeam=erfaephot,method=biasmethod,
                                               savefile=outdir+tname+'-apfield'+faQUcorr['corrname'])

            ## Plot polarization of field stars
            xyplotpol(axsrc,aysrc,fapol,faangle,#mask=pamask.reshape(-1),
                      center=center,savefile=outdir+tname+'-apfield'+faQUcorr['corrname'])

            ## Bin stars and plot
            fapolmap,erfapolmap = bin_points(axsrc,aysrc,fapol,fullbin=True,center=center,
                                             savefile=outdir+tname+'-apfield-binpol'+faQUcorr['corrname'])#,mask=pamask)
            faanglemap,erfaanglemap = bin_points(axsrc,aysrc,faangle,fullbin=True,center=center,
                                                 savefile=outdir+tname+'-apfield-binangle'+faQUcorr['corrname'])#,mask=pamask)
            plotpol(fapolmap,faanglemap,erpol=erfapolmap,erangle=erfaanglemap,
                    step=40,center=center,savefile=outdir+tname+'-apfield-bin'+faQUcorr['corrname'])
            radius_dependence(fapolmap,faanglemap,outdir+tname+'-apfield-bin'+faQUcorr['corrname'],scatter=True,
                              inla=inla,radfit=doradfit,parfit=doparfit,erpol=erfapolmap,filt=polfilters[f])#,center=center)

            ## PSF vs AP
            if photcompare:
                #compare_flux(allfieldphot[ind],allfieldephot[ind],allafieldphot[ind],allafieldephot[ind],
                #             outdir+tname+'-field',maxdiff=0.2)
                compare_polangle(fpol,fangle,fapol,faangle,outdir+tname+'-field',xtit='PSF',ytit='AP')#,mask=pamask)
                compare_polangle(fpolmap,fanglemap,fapolmap,faanglemap,outdir+tname+'-field-bin',
                                 xtit='PSF',ytit='AP')

    #combine offsets
    if comboff and noffset > 1:
        if dopix:
            combine_offsets(offset,polfilters[f],outdir,cname+QUcorr['corrname'],center=center)
        if dobin:
            combine_offsets(offset,polfilters[f],outdir,cname+bname+binQUcorr['corrname'],
                            center=center)
                
    #average offsets
    if dophot and noffset > 1 and offset['type'][0] == 'it':
        average_iterations('aper',offset,polangles,outdir,fname,cname)
        average_iterations('psf',offset,polangles,outdir,fname,cname)

    #field for all offset
    if dofphot and noffset > 1 and offset['type'][0] == 'off':
        if cenpol != 'None' or instpol != 'None':
            sum_offsets(offset,polfilters[f],outdir,cosmic=docosmic,posfree=posfree,
                        center=center,binpts=20,sigmaclip=2.0,parfit=doparfit,
                        bias=polbias,method=biasmethod)
            sum_offsets(offset,polfilters[f],outdir,cosmic=docosmic,center=center,ap=True,#,posfree=posfree,
                        binpts=20,sigmaclip=2.0,parfit=doparfit,bias=polbias,method=biasmethod)
        if cenpol != 'None':
            sum_offsets(offset,polfilters[f],outdir,cosmic=docosmic,posfree=posfree,
                        bias=polbias,method=biasmethod,
                        center=center,binpts=20,sigmaclip=2.0,parfit=doparfit,xname='-corr'+cenpol)
            sum_offsets(offset,polfilters[f],outdir,cosmic=docosmic,center=center,ap=True,
                        bias=polbias,method=biasmethod,
                        binpts=20,sigmaclip=2.0,parfit=doparfit,xname='-corr'+cenpol)
        sum_offsets(offset,polfilters[f],outdir,cosmic=docosmic,posfree=posfree,center=center,bias=polbias,
                    method=biasmethod,binpts=20,sigmaclip=2.0,parfit=doparfit,xname=faQUcorr['corrname'])
        sum_offsets(offset,polfilters[f],outdir,cosmic=docosmic,center=center,ap=True,bias=polbias,
                    method=biasmethod,binpts=20,sigmaclip=2.0,parfit=doparfit,xname=faQUcorr['corrname'])#,posfree=posfree,
        
#%%%% End filter loop    



## --PLOT MOON (wave independent for now)
from moon import *
head,data = read_fits(datadir+polfiles['file'][0])
moon_pol,moon_angle = moon_polmap(head['DATE-OBS'],head['RA'],head['DEC'],savefile=outdir,airmass=True)

## -- ANALYSIS OVER FILTERS
if len(polfilters) > 1:

    ##strips and quadpars
    if calcstrips:
        analyse_filtstrips(polfilters,outdir)
    if findstars:
        analyse_filtquadpars(polfilters,outdir,fwhm=fwhm,threshold=threshold)

    ##pol and angle over filter
    if dophot:
        analyse_filtphpol(polfilters,target,outdir,fit='Serkowski')#gname??

## -- CLOSE LOG FILE
if logfile is not '':
    #sys.stdout = orig_stdout
    logf.close()
