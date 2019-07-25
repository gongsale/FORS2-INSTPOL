# FORS2-INSTPOL

Title: Tips and Tricks in linear imaging polarimetry ofextended sources with FORS2 at the VLT

Authors: S. Gonzalez-Gaitan, A. M. Mourão, F. Patat, J. P. Anderson, A. Cikota, K.Wiersema, A. B. Higgins and K. Silva.

We present here: 

- Pipeline to reduce and analyze linear imaging polarimetry of point/extended source for FORS2-IPOL. 
- Maps of instrumental linear polarization of FORS2 instrument in different filters.

### *Code to reduce and analyze FORS2 IPOL data* 

We provide in the folder *codes* two main Python files: *eso_ipol* with all the necessary routines to do all reduction steps including bias, cosmic rays, flat-fielding as well as analysis routines to calculate and plot Stokes, polarization degree and angle maps, among others. An example on how to put all these routines together is given in the *multi_eso_ipol* file (together with the *standard.input* file). More information is found in each of these respective files.

### *Maps of instrumental polarization* 

We release several maps to correct for the instrumental field polarization in FORS2 in filters BVRI. These are found in the folder *Information* and contain:
- strip positions to separate ordinary and extraordinary beams in each BVRI filter (*default_strips.dat*)
- parameters of the quadratic function to match ordinary and extraordinary beam positions of each BVRI filter (*default_quadpars.dat*).
- parameters of the analytic hyperbolic paraboloids fits for Q and U (*hypparab_instQU.dat*)
- parameters of the analytic paraboloids fits for P (*parab_instpol.dat*)
- non-analytic INLA fits for Q,U,P and CHI in all four filters in fits format (e.g. *R_SPECIAL_inlaQ.fits*)
- binned maps for Q,U,P and CHI in all four filters in fits format (e.g. *R_SPECIAL_instQ.fits*)


