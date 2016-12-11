                                             
/* ********************************************************************
    yee2d.c : 2-D FDTD TE code with UPML absorbing boundary conditions
   ********************************************************************

   A 2-D FDTD TE implementation of UPML ABC
    version 1.0,  4/30/2008,  Doug Neubauer
    From Taflove2005 pp. 297-302 (and see fdtd3d_upml.m by JK Willis)
    Also from Sullivan2000 and Gedney1996

    In this implementation of UPML, the UPML algorithm is evaluated
    over the entire grid, including the main grid (per Taflove2005)

     This program implements the finite-difference time-domain
     solution of Maxwell's curl equations over a two-dimensional
     Cartesian space lattice comprised of uniform square grid cells.

     To illustrate the algorithm, a 6-cm-diameter metal cylindrical
     scatterer in free space is modeled. The source excitation is
     a Gaussian pulse with a carrier frequency of 5 GHz.

     The grid resolution (dx = 3 mm) was chosen to provide 20 samples
     per wavelength at the center frequency of the pulse (which in turn
     provides approximately 10 samples per wavelength at the high end
     of the excitation spectrum, around 10 GHz).

     The computational domain is truncated using the uniaxial perfectly matched
     layer (UPML) absorbing boundary conditions.  The formulation used
     in this code is based on the UPML ABC. The
     PML regions are labeled as shown in the following diagram:

                                                             xSize-1,ySize-1
                                                            /
            ----------------------------------------------.
           |  |                BACK PML                |  |
            ----------------------------------------------
           |L |                                        | R|
           |E |                                        | I|
           |F |                                        | G|
           |T |                                        | H|
           |  |                MAIN GRID               | T|
           |P |                                        |  |
           |M |                                        | P|
           |L |                                        | M|
           |  |                                        | L|
            ----------------------------------------------
           |. |                FRONT PML               |  |
           /----------------------------------------------
         0,0


   Below: Detailed view of the Yee grid...  (where N=xSize, M=ySize, see below)
   Note: an extra column of ey on right edge and an extra row of ex on back edge
   Note: ey at x=0 and x=N are PEC and ex at y=0 and y=M are PEC.

  (0,M)                                                             (N-1,M)
 ___ex___  ___ex___  ___ex___  ___ex___ .. ___ex___  ___ex___  ___ex___
 ......... ......... ......... .........   ......... ......... .........
|        .|        .|        .|        .  |        .|        .|        .|
| 0,M-1  .|        .|        .|        .  |        .|        .| N-1,M-1.|(N,M-1)
ey  hz   .ey  hz   .ey  hz   .ey  hz   .  ey  hz   .ey  hz   .ey  hz   .ey
|        .|        .|        .|        .  |        .|        .|        .|
|___ex___.|___ex___.|___ex___.|___ex___.  |___ex___.|___ex___.|___ex___.|
 ......... ......... ......... ........... ......... ......... .........
|        .|        .|        .|        .  |        .|        .|        .|
|  0,M-2 .|        .|        .|        .  |        .|        .| N-1,M-2.|
ey  hz   .ey  hz   .ey  hz   .ey  hz   .  ey  hz   .ey  hz   .ey  hz   .ey
|        .|        .|        .|        .  |        .|        .|        .|
|___ex___.|___ex___.|___ex___.|___ex___...|___ex___.|___ex___.|___ex___.|
.                                      .  .                             .
.                                      .  .                             .
 ......... ......... ......... ........... ......... ......... .........
|        .|        .|        .|        .  |        .|        .|        .|
|  0,2   .|        .|        .|        .  |        .|        .| N-1,2  .|
ey  hz   .ey  hz   .ey  hz   .ey  hz   .  ey  hz   .ey  hz   .ey  hz   .ey
|        .|        .|        .|        .  |        .|        .|        .|
|___ex___.|___ex___.|___ex___.|___ex___.  |___ex___.|___ex___.|___ex___.|
 ......... ......... ......... ........... ......... ......... .........
|        .|        .|        .|        .  |        .|        .|        .|
|  0,1   .|        .|        .|        .  |        .|        .| N-1,1  .|
ey  hz   .ey  hz   .ey  hz   .ey  hz   .  ey  hz   .ey  hz   .ey  hz   .ey
|        .|        .|        .|        .  |        .|        .|        .|
|___ex___.|___ex___.|___ex___.|___ex___.  |___ex___.|___ex___.|___ex___.|
 ......... ......... ......... ........... ......... ......... .........
|        .|        .|        .|        .  |        .|        .|        .|
|  0,0   .|  1,0   .|   2,0  .|  3.0   .  |  N-3,0 .|  N-2,0 .| N-1,0  .|(N,0)
ey  hz   .ey  hz   .ey  hz   .ey  hz   .  ey  hz   .ey  hz   .ey  hz   .ey
|        .|        .|        .|        .  |        .|        .|        .|
|___ex___.|___ex___.|___ex___.|___ex___...|___ex___.|___ex___.|___ex___.|


     PML Reflection Simulation Results:
     ------------------------------------------
     #layer      r0      gradingOrder     dB
     ------    ------    ------------    -----
       8       1.0e-7        2           -74.4
       8       1.0e-7        3           -81.1   <== use this
       8       1.0e-7        4           -80.1
      10       1.0e-7        3           -90.0
      10       1.0e-8        3           
       6       1.0e-7        3           
       4       1.0e-7        3           -51.8

 The results are pretty much the same as the Split-Field pml results.
 Also, during the cylinder simulation data was compared between the
 split-field version and this upml version results were within -122 dB. 

 In the spirit of the ToyFdtd programs, a point was made to try to heavily 
 comment the source code.

********************************************************************** */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define  ABCSIZECONSTANT    (8)                      // thickness of PML region
#define  MEDIACONSTANT    (2)                        // number of different medias, ie 2: vacuum, metallicCylinder
#define  NUMBEROFITERATIONCONSTANT    (400)          // was 300
#define  BIGLINESIZE    (8192)                       // for creating an output filename

// These global variables are used by ExecuteFdtd()

int  xSize ;                           // number of total grid cells in x-direction  (left+main+right)
int  ySize ;                           // number of total grid cells in y-direction  (front+main+back)
int  maximumIteration;                 // how long to run the simulation
int  xSource, ySource;                 // location of z-directed hard source

double  **ex;      // the fields
double  **dx;      //  ""  (note this Implementation requires dx, dy, and bz. For simplicity they are evaluated over the entire grid.  Note Sullivan2000 has a somewhat different implementation, this version is from Taflove2005)
double  **ey;      //  ""
double  **dy;      //  ""
double  **bz;      //  ""
double  **hz;      //  ""

double  c1x;     // fdtd coefficents
double  **c2x;    //  ""
double  **c3x;    //  ""
double  *c4x;    //  ""
double  *c5x;    //  ""
double  c1y;     //  ""
double  **c2y;    //  ""
double  **c3y;    //  ""
double  *c4y;    //  ""
double  *c5y;    //  ""
double  *d1z;    //  ""
double  *d2z;    //  ""
double  *d3z;    //  ""
double  *d4z;    //  ""

double  sourceValue[NUMBEROFITERATIONCONSTANT];  // holds the pre-calculated values of the source, for the run of the simulation (note: these values could be calculated on the fly)

// standard C memory allocation for 2-D array
double  **AllocateMemory (int  imax, int  jmax, double  initialValue)
{
    int  i,j;
    double  **pointer;
    pointer = (double **)malloc(imax * sizeof(double *));
    for (i = 0; i < imax; i++) {
        pointer[i] = (double *)malloc(jmax * sizeof(double));
        for (j = 0; j < jmax; j++) {
            pointer[i][j] = initialValue;
        } /* jForLoop */
    } /* iForLoop */
    return(pointer) ;
}

// standard C memory allocation for 1-D array
double  *AllocateMemory1D (int  size, double  initialValue)
{
    int  j;
    double  *pointer;
    pointer = (double *)malloc(size * sizeof(double));
    for (j = 0; j < size; j++) {
        pointer[j] = initialValue;
    } /* jForLoop */
    return(pointer) ;
}

void  InitializeFdtd ()
{         
    double  mediaPermittivity[MEDIACONSTANT] = {1.0, 1.0};    // eps, index=0 is for vacuum, index=1 is for the metallic cylinder
    double  mediaConductivity[MEDIACONSTANT] = {0.0, 1.0e+7}; // sig,
    double  mediaPermeability[MEDIACONSTANT] = {1.0, 1.0};    // mur
    double  mediaResistivity[MEDIACONSTANT] = {0.0, 0.0};     // sim
    double  pi,speedOfLight,magneticPermeability0,electricalPermittivity0,impedance0,frequency,wavelength,angularFrequency;
    double  delta,dt,reflectionCoefficient0,gradingOrder,temporary,temporary1;
    int  i,j,xSizeMain,ySizeMain;
    int  abcSize ;
    double   mediaC1[MEDIACONSTANT], mediaC2[MEDIACONSTANT], mediaC3[MEDIACONSTANT], mediaC4[MEDIACONSTANT], mediaC5[MEDIACONSTANT], mediaD1[MEDIACONSTANT], mediaD2[MEDIACONSTANT], mediaD3[MEDIACONSTANT], mediaD4[MEDIACONSTANT];
    int  media;
    double  cylinderDiameter, cylinderRadius, temporaryi,temporaryj,distance2 ;
    int  cylinderXCenter,cylinderYCenter;
    double  x,x1,x2;
    double  electricalConductivityMaximum, boundaryWidth, gradientConductivity , boundaryFactor,gradientK;
    double  gradientC2[ABCSIZECONSTANT];
    double  gradientC3[ABCSIZECONSTANT];
    double  gradientC4[ABCSIZECONSTANT];
    double  gradientC5[ABCSIZECONSTANT];
    double  gradientD1[ABCSIZECONSTANT];
    double  gradientD2[ABCSIZECONSTANT];
    double  gradientD3[ABCSIZECONSTANT];
    double  gradientD4[ABCSIZECONSTANT];
    double  rtau, tau, delay;
    // char  ch;

    //***********************************************************************
    //     Fundamental constants
    //***********************************************************************

    pi  = (acos(-1.0));
    speedOfLight = 2.99792458e8;                          //speed of light in free space (meters/second)
    magneticPermeability0 = 4.0 * pi * 1.0e-7;            //permeability of free space
    electricalPermittivity0 = 1.0 / (speedOfLight * speedOfLight * magneticPermeability0);       //permittivity of free space
    impedance0 = sqrt(magneticPermeability0 / electricalPermittivity0);

    frequency = 5.0e+9;                                   //center frequency of source excitation (Hz)
    wavelength = speedOfLight / frequency  ;              //center wavelength of source excitation
    angularFrequency = 2.0 * pi * frequency;              //center frequency in radians

    //***********************************************************************
    //     Grid parameters
    //***********************************************************************

    xSizeMain = 100;                              // number of main grid cells in x-direction
    // ySizeMain = 100;                               // number of main grid cells in y-direction
    ySizeMain = 50;                               // number of main grid cells in y-direction
    abcSize = ABCSIZECONSTANT;                    // thickness of PML region
    xSize = xSizeMain + 2 * abcSize;              // number of total grid cells in x-direction
    ySize = ySizeMain + 2 * abcSize;              // number of total grid cells in y-direction

    //xSource = (xSizeMain / 2) + abcSize;             //location of z-directed hard source
    //ySource = (ySizeMain / 2) + abcSize;             //location of z-directed hard source
    xSource = 15 + abcSize;                       //location of z-directed hard source
    ySource = ySize / 2;                          //location of z-directed hard source

    delta = 3.0e-3;                                  //space increment of square lattice  (meters)
    dt = delta / (2.0 * speedOfLight);               //time step,  seconds, courant limit, Taflove1995 page 177

    maximumIteration = NUMBEROFITERATIONCONSTANT;                 //total number of time steps

    reflectionCoefficient0 = 1.0e-7;              // 1.0e-7 for PML, Nikolova part4 p.25
    gradingOrder = 3;                             // for PML, (m) was 3;  optimal values: 2 <= m <= 6,  Nikolova part4 p.29
       
    //***********************************************************************
    //     Material parameters
    //***********************************************************************
                   
    media = MEDIACONSTANT;        // number of different medias, ie 2: vacuum, metallicCylinder

    //***********************************************************************
    //     Wave excitation
    //***********************************************************************

#if 0
    for (i = 0; i < maximumIteration; i++) {
        temporary = (double  )i;
        sourceValue[i] = sin( angularFrequency * temporary * dt);   // simple sine wave
    } /* iForLoop */
#endif
#if 1
    rtau = 160.0e-12;
    tau = rtau / dt;
    delay = 3 * tau;
    for (i = 0; i < maximumIteration; i++) {
        sourceValue[i] = 0.0;
    } /* iForLoop */
    for (i = 0; i < (int  )(7.0 * tau); i++) {   // Gaussian pulse with a carrier frequency of 5 GHz
        temporary = (double  )i - delay;
        sourceValue[i] = sin( angularFrequency * (temporary) * dt) * exp(-( (temporary * temporary)/(tau * tau) ) );
    } /* iForLoop */
#endif

    //***********************************************************************
    //     Field arrays
    //***********************************************************************

    ex = AllocateMemory(xSize,    ySize + 1, 0.0 );        // 1 extra in y direction for pec  (the extra is used by bz only)
    ey = AllocateMemory(xSize + 1,ySize,     0.0 );        // 1 extra in x direction for pec  (the extra is used by bz only)
    dx = AllocateMemory(xSize,    ySize,     0.0 );
    dy = AllocateMemory(xSize,    ySize,     0.0 );
    bz = AllocateMemory(xSize,    ySize,     0.0 );
    hz = AllocateMemory(xSize,    ySize,     0.0 );
     

    //***********************************************************************
    //     Media coefficients
    //***********************************************************************

    // the coefficients are used in the "update electrical fields equations" as follows:
    // dx[i][j] += c1x * (hz[i][j] - hz[i][j-1])
    // dy[i][j] += c1y * (hz[i-1][j] - hz[i][j])
    // ex[i][j] = c2x[i][j] * ex[i][j] + c3x[i][j] * (c4x[i] * dx[i][j] + c5x[i] * dxOld[i][j])
    // ey[i][j] = c2y[i][j] * ey[i][j] + c3y[i][j] * (c4y[j] * dy[i][j] + c5y[j] * dyOld[i][j])
    // 
    // c1x = dt/(electricalPermittivity * delta)     (note: c1x is a constant)
    // c2x = (electricalPermittivity/dt - ConductivityY/2) / c3x
    // c3x = 1 / (electricalPermittivity/dt  + ConductivityY/2);
    // c4x = electricalPermittivity/dt + ConductivityX/2
    // c5x = -electricalPermittivity/dt + ConductivityX/2                     
    // The c4x and c5x terms are used in the UPML, in the main grid they reduce to:
    // c4x = electricalPermittivity/dt
    // c5x = -electricalPermittivity/dt
    // Also as c4x,c5x are only used along the edges they can don't need to be 2-d arrays.
    //
    // the coefficients are used in the "update magnetic fields equations" as follows:
    // bz[i][j] = d1z[i] * bz[i][j] + d2z[i] * ( ex[i][j+1] - ex[i][j] + ey[i][j] - ey[i+1][j] );
    // hz[i][j] = d3z[j] * hz[i][j] + d4z[j] * ( bz[i][j] - bzOld[i][j] );
    // d1z[i] = (1/dt - ConductivityX/(2 * electricalPermittivity)) / (1/dt + ConductivityX/(2 * electricalPermittivity))
    // d2z[i] = (1/delta) / (1/dt + ConductivityX/(2 * electricalPermittivity))
    // d3z[j] = (magneticPermeability/dt - ConductivityY * magneticPermeability/(2 * electricalPermittivity)) / (magneticPermeability/dt + ConductivityY * magneticPermeability/(2 * electricalPermittivity))
    // d4z[j] = (1/dt) / (magneticPermeability/dt + ConductivityY * magneticPermeability/(2 * electricalPermittivity))
    // In vacuum (conductivity=0) they reduce to  (also in this implementation, the cylinder doesn't modify the magnetic coefficients)
    // d1 = 1
    // d2 = dt/delta 
    // d3 = 1
    // d4 = 1/magneticPermeability0

    // carefully calculate the coefficients for the Vacuum and cylinder
    for (i = 0; i < media; i++) {
        temporary = (electricalPermittivity0 * mediaPermittivity[i]) / dt ;
        temporary1 = mediaConductivity[i] / 2.0;
        mediaC1[i] = dt / (electricalPermittivity0 * mediaPermittivity[i] * delta) ;
        mediaC2[i] = (temporary - temporary1) / (temporary + temporary1);
        mediaC3[i] = 1.0 / (temporary + temporary1);
        mediaC4[i] = temporary;
        mediaC5[i] = -temporary;

        mediaD1[i] = 1.0;
        mediaD2[i] = dt / delta;
        mediaD3[i] = 1.0;
        mediaD4[i] = 1.0 / magneticPermeability0;
    } /* iForLoop */


    //***********************************************************************
    //     Grid Coefficients
    //***********************************************************************

    //     Initialize entire grid to free space
    c1x =                          mediaC1[0] ;           // note: don't need to allocate for pec region, as it is not evaluated
    c2x = AllocateMemory  ( xSize,ySize, mediaC2[0]);     // also: Initialize the entire grid to vacuum.
    c3x = AllocateMemory  ( xSize,ySize, mediaC3[0]);     // "
    c4x = AllocateMemory1D( xSize, mediaC4[0]);           // "
    c5x = AllocateMemory1D( xSize, mediaC5[0]);           // "
    c1y =                          mediaC1[0] ;           // "
    c2y = AllocateMemory  ( xSize,ySize, mediaC2[0]);     // "
    c3y = AllocateMemory  ( xSize,ySize, mediaC3[0]);     // "
    c4y = AllocateMemory1D( ySize, mediaC4[0]);     // "
    c5y = AllocateMemory1D( ySize, mediaC5[0]);     // "

    d1z = AllocateMemory1D( xSize, mediaD1[0]);     // "
    d2z = AllocateMemory1D( xSize, mediaD2[0]);     // "
    d3z = AllocateMemory1D( ySize, mediaD3[0]);     // "
    d4z = AllocateMemory1D( ySize, mediaD4[0]);     // "


    //     Add metal cylinder
#if 1
    cylinderDiameter = 20;                                     // diameter of cylinder: 6 cm
    cylinderRadius = cylinderDiameter / 2.0;                   // radius of cylinder: 3 cm
    cylinderXCenter = (4 * xSizeMain) / 5 + abcSize;           // i-coordinate of cylinder's center
    cylinderYCenter = ySize / 2;                               // j-coordinate of cylinder's center
    for (i = 0; i < xSize; i++) {
        for (j = 0; j < ySize; j++) {
            temporaryi = (double  )(i - cylinderXCenter) ;
            temporaryj = (double  )(j - cylinderYCenter) ;
            distance2 = (temporaryi + 0.5) * (temporaryi + 0.5) + (temporaryj) * (temporaryj);
            if (distance2 <= (cylinderRadius * cylinderRadius)) {
                c2x[i][j]  = mediaC2[1];
                c3x[i][j]  = mediaC3[1];
            } /* if */
            // This looks tricky! Why can't caey/cbey use the same 'if' statement as caex/cbex above ??
            distance2 = (temporaryj + 0.5) * (temporaryj + 0.5) + (temporaryi) * (temporaryi);
            if (distance2 <= (cylinderRadius * cylinderRadius)) {
                c2y[i][j]  = mediaC2[1];
                c3y[i][j]  = mediaC3[1];
            } /* if */
        } /* jForLoop */
    } /* iForLoop */
#endif

    //***********************************************************************
    //     Fill the PML regions    ---  (Caution...Here there be Tygers!)
    //***********************************************************************

    // The most important part of the PML fdtd simulation is getting the
    // PML Coefficients correct. Which requires getting the correct PML gradient and
    // positioning the coefficients correctly on the x-y grid.

    // ALERT: It is possible to make a mistake here, and yet the simulation may appear to
    // be working properly. However a detailed analysis of reflections off the PML
    // will show they may be (much) larger than those for a correctly designed PML.

    boundaryWidth = (double  )abcSize * delta;    // width of PML region (in mm)

    // SigmaMaximum, using polynomial grading (Nikolova part 4, p.30), rmax=reflectionMax in percent
    electricalConductivityMaximum = -log(reflectionCoefficient0) * (gradingOrder + 1.0) / (2.0 * impedance0 * boundaryWidth);

    // boundaryFactor comes from the polynomial grading equation: sigma_x = sigmaxMaximum * (x/d)^m, where d=width of PML, m=gradingOrder, sigmaxMaximum = electricalConductivityMaximum    (Nikolova part4, p.28)
    //  IMPORTANT: The conductivity (sigma) must use the "average" value at each mesh point as follows:
    //  sigma_x = sigma_Maximum/delta * Integral_from_x0_to_x1 of (x/d)^m delta,  where x0=currentx-0.5, x1=currentx+0.5   (Nikolova part 4, p.32)
    //  integrating gives: sigma_x = (sigmaMaximum / (delta * d^m * m+1)) * ( x1^(m+1) - x0^(m+1) )     (Nikolova part 4, p.32)
    //  the first part is "boundaryFactor", so, sigma_x = boundaryFactor * ( x1^(m+1) - x0^(m+1) )   (Nikolova part 4, p.32)
    boundaryFactor = electricalConductivityMaximum / ( delta * (pow(boundaryWidth,gradingOrder)) * (gradingOrder + 1));
    gradientK = 1.0;

    // build the gradient
    //  caution: if the gradient is built improperly, the PML will not function correctly
    for (i = 0, x = 0.0; i < abcSize; i++, x++) {
        // 0=border between pml and vacuum
        // even: for ex and ey  parallel to border
        x1 = (x + 0.5) * delta;       // upper bounds for point i
        x2 = (x - 0.5) * delta;       // lower bounds for point i
        if (i == 0) {
            gradientConductivity = boundaryFactor * (pow(x1,(gradingOrder+1))  );   //   polynomial grading  (special case: on the edge, 1/2 = pml, 1/2 = vacuum)
        } /* if */
        else {
            gradientConductivity = boundaryFactor * (pow(x1,(gradingOrder+1)) - pow(x2,(gradingOrder+1)) );   //   polynomial grading
        } /* else */
        temporary = gradientK * electricalPermittivity0 / dt ;
        temporary1 = gradientConductivity / 2.0;

        gradientC2[i] = (temporary - temporary1) / (temporary + temporary1);
        gradientC3[i] = 1.0 / (temporary + temporary1);

        // odd: for hz and (ex,ey perpendicular to border)
        x1 = (x + 1.0) * delta;       // upper bounds for point i
        x2 = (x + 0.0) * delta;       // lower bounds for point i
        gradientConductivity = boundaryFactor * (pow(x1,(gradingOrder+1)) - pow(x2,(gradingOrder+1)) );   //   polynomial grading

        temporary = gradientK * electricalPermittivity0 / dt ;
        temporary1 = gradientConductivity / 2.0;
 
        gradientC4[i] = (temporary + temporary1);
        gradientC5[i] = (temporary1 - temporary);
         
        temporary = gradientK / dt ;
        temporary1 = gradientConductivity / (2.0 * electricalPermittivity0);

        gradientD1[i] = (temporary - temporary1) / (temporary + temporary1);
        gradientD2[i] = (1.0 / delta) / (temporary + temporary1);
                          
        temporary  *= magneticPermeability0;
        temporary1 *= magneticPermeability0;

        gradientD3[i] = (temporary - temporary1) / (temporary + temporary1);
        gradientD4[i] = (1.0 / dt) / (temporary + temporary1);

    } /* iForLoop */

    // ex, ey, hz --- front/back (y)
    for (j = 0; j < abcSize; j++) {                            // do coefficients for ex, ey and hz
        // do coefficients for ex for front and back regions

        for (i = 0; i < xSize; i++) {
            c2x[i][abcSize - j]         = gradientC2[j];     // front
            c3x[i][abcSize - j]         = gradientC3[j];
            c2x[i][ySize - abcSize + j] = gradientC2[j];     // back
            c3x[i][ySize - abcSize + j] = gradientC3[j];
        } /* iForLoop */     
    
        c4y[abcSize - j -1]         = gradientC4[j];     // front
        c5y[abcSize - j -1]         = gradientC5[j];
        c4y[ySize - abcSize + j] = gradientC4[j];     // back
        c5y[ySize - abcSize + j] = gradientC5[j];
    
        d3z[abcSize - j - 1]     = gradientD3[j];     // front, (note that the index is offset by 1 from c1x,c2x)
        d4z[abcSize - j - 1]     = gradientD4[j];     //   "         ( ditto )
        d3z[ySize - abcSize + j] = gradientD3[j];     // back
        d4z[ySize - abcSize + j] = gradientD4[j];     //   "
    } /* jForLoop */
    

    // ey,hz --- left/right (x)
    for (i = 0; i < abcSize; i++) {                            // do coefficients for ey and hz
        // do coefficients for ey for left and right regions
    
        for (j = 0; j < ySize; j++) {
            c2y[abcSize - i][j]         = gradientC2[i];     // left
            c3y[abcSize - i][j]         = gradientC3[i];
            c2y[xSize - abcSize + i][j] = gradientC2[i];     // right
            c3y[xSize - abcSize + i][j] = gradientC3[i];
        } /* jForLoop */     
    
        c4x[abcSize - i -1]         = gradientC4[i];     // left
        c5x[abcSize - i -1]         = gradientC5[i];
        c4x[xSize - abcSize + i] = gradientC4[i];     // right
        c5x[xSize - abcSize + i] = gradientC5[i];
    
        d1z[abcSize - i - 1]     = gradientD1[i];     // left, (note that the index is offset by 1 from c1y,c2y)
        d2z[abcSize - i - 1]     = gradientD2[i];     //   "         ( ditto )
        d1z[xSize - abcSize + i] = gradientD1[i];     // right
        d2z[xSize - abcSize + i] = gradientD2[i];     //   "
    } /* iForLoop */     


    // all done with Initialization!


    //***********************************************************************
    //     Print variables (diagnostic)
    //***********************************************************************

    printf("# pi:%16.10g\n",pi);
    printf("# speedOfLight:%16.10g\n",speedOfLight);
    printf("# magneticPermeability0:%16.10g\n",magneticPermeability0);
    printf("# electricalPermittivity0:%16.10g\n",electricalPermittivity0);
    printf("# impedance0:%16.10g\n",impedance0);
    printf("# frequency:%16.10g\n",frequency);
    printf("# wavelength:%16.10g\n",wavelength);
    printf("# angularFrequency:%16.10g\n",angularFrequency);
    printf("# delta:%16.10g\n",delta);
    printf("# dt:%16.10g\n",dt);
    printf("# reflectionCoefficient0:%16.10g\n",reflectionCoefficient0);
    printf("# gradingOrder:%16.10g\n",gradingOrder);
    printf("# xSizeMain:%d\n",xSizeMain);
    printf("# ySizeMain:%d\n",ySizeMain);
    printf("# abcSize:%d\n",abcSize);
    printf("# xSize:%d\n",xSize);
    printf("# ySize:%d\n",ySize);
    printf("# xSource:%d\n",xSource);
    printf("# ySource:%d\n",ySource);
    printf("# maximumIteration:%d\n",maximumIteration);

#if 0
    for (i = 0; i < maximumIteration; i++) {
        printf("%d: source: %16.10g\n",i,sourceValue[i]);
    } /* iForLoop */
    
    for (j = 0; j < xSize; j++) {     // varies with x: c2y c3y c4x c5x d1z d2z
        printf("x:%d  c2y:%g c3y:%g c4x:%g c5x:%g  d1z:%g d2z:%g\n",j, c2y[j], c3y[j], c4x[j], c5x[j],   d1z[j],  d2z[j]  );
    } /* jForLoop */
    for (j = 0; j < ySize; j++) {     // varies with y: c1x c2x c4y c5y d3z d4z
        printf("y:%d  c2x:%g c3x:%g c4y:%g c5y:%g  d3z:%g d4z:%g\n",j, c2x[j], c3x[j], c4y[j], c5y[j],   d3z[j],  d4z[j]  );
    } /* jForLoop */

    exit(0);
#endif

}




void  EvaluateFdtd (double  minimumValue, double  maximumValue)
{
    int  n,i,j;
    double  temporary;
    FILE *filePointer ;                   // for plotting
    int  plottingInterval,iValue;         // for plotting
    double  scaleValue;                   // for plotting
    char  filename[BIGLINESIZE] ;         // for plotting
    int  centerx = 50+ABCSIZECONSTANT ;   // for printing
    int  centery = 25+ABCSIZECONSTANT ;   //    ""

    //***********************************************************************
    //     BEGIN TIME-STEPPING LOOP
    //***********************************************************************

    plottingInterval = 0;
    for (n = 0; n  < maximumIteration; n++) {  // iteration loop

        fprintf(stderr,"n:%d\n",n);

        //***********************************************************************
        //     Update electric fields (EX and EY)
        //***********************************************************************

        for (i = 0; i < xSize; i++) {
            for (j = 1; j < ySize; j++) {        // j=0 = pec, so don't evaluate
                temporary = dx[i][j];
                dx[i][j] += c1x * ( hz[i][j] - hz[i][j-1] );
                ex[i][j] = c2x[i][j] * ex[i][j] + c3x[i][j] * (c4x[i] * dx[i][j] + c5x[i] * temporary );
            } /* jForLoop */
        } /* iForLoop */

        for (i = 1; i < xSize; i++) {            // i=0 = pec, so don't evaluate
            for (j = 0; j < ySize; j++) {
                temporary = dy[i][j];
                dy[i][j] += c1y * ( hz[i-1][j] - hz[i][j] );
                ey[i][j] = c2y[i][j] * ey[i][j] + c3y[i][j] * (c4y[j] * dy[i][j] + c5y[j] * temporary );
            } /* jForLoop */
        } /* iForLoop */


        //***********************************************************************
        //     Update magnetic fields (HZ)
        //***********************************************************************


        for (i = 0; i < xSize; i++) {
            for (j = 0; j < ySize; j++) {
                temporary = bz[i][j];
                bz[i][j] = d1z[i] * bz[i][j] + d2z[i] * ( ex[i][j+1] - ex[i][j] + ey[i][j] - ey[i+1][j] );
                hz[i][j] = d3z[j] * hz[i][j] + d4z[j] * ( bz[i][j] - temporary );
            } /* jForLoop */
        } /* iForLoop */

        hz[xSource][ySource] = sourceValue[n];



        //***********************************************************************
        //     Plot fields
        //***********************************************************************
#if 1
        if (plottingInterval == 0) {
            plottingInterval = 2;

            // plot a field (based somewhat on the ToyFdtd BlockOfBricks (bob) output format)
            // at each plottingInterval, print out a file of field data (field values normalized to a range between 0 and 255).

            sprintf(filename, "c_%06d.bob", n);
            filePointer = fopen(filename, "wb") ;
            if (filePointer == 0) {
                fprintf(stderr, "Difficulty opening c_%06d.bob", n);
                exit(1);
            } /* if */

            scaleValue = 256.0 / (maximumValue - minimumValue);
            for (j = 0; j < ySize; j++) {
                for (i = 0; i < xSize; i++) {
                    temporary = hz[i][j];
                    temporary = (temporary - minimumValue) * scaleValue;
                    iValue = (int  )(temporary) ;
                    if (iValue < 0) {
                        iValue = 0;
                    } /* if */
                    if (iValue > 255) {
                        iValue = 255;
                    } /* if */
                    putc( iValue, filePointer);
                } /* xForLoop */
            } /* yForLoop */
            fclose(filePointer);
        } /* if */
        plottingInterval--;
#endif

        //***********************************************************************
        //     Print fields
        //***********************************************************************

        printf("{ %d\n",n);
        for (i = 0; i < 8; i++) {    // printout some fields
            printf("eX_%d_%d: %16.10g\n", i,50, ex[ centerx-48 + i][centery]);
            printf("eY_%d_%d: %16.10g\n", i,50, ey[ centerx-48 + i][centery]);
            printf("hZ_%d_%d: %16.10g\n", i,50, hz[ centerx-48 + i][centery]);
        } /* iForLoop */
        printf("eX_%d_%d: %16.10g\n", 50,25, ex[ 50][25]);
        printf("eY_%d_%d: %16.10g\n", 50,25, ey[ 50][25]);
        printf("hZ_%d_%d: %16.10g\n", 50,25, hz[ 50][25]);
        printf("}\n");

    } /* iteration forLoop */
}


int  main ()
{
    InitializeFdtd();
    EvaluateFdtd(-0.1,0.1);
    return(0) ;
}

