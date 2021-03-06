C
C               Tennessee Eastman Process Control Test Problem
C
C                  Original codes written by
C
C                    James J. Downs and Ernest F. Vogel
C
C                  Process and Control Systems Engineering
C                        Tennessee Eastman Company
C                              P.O. Box 511
C                          Kingsport, Tennessee 37662
C
C--------------------------------------------------------------------
C
C  New version is a closed-loop plant-wide control scheme for 
C  the Tennessee Eastman Process Control Test Problem
C                        
C  The modifications are by:
C
C            Evan L. Russell, Leo H. Chiang and Richard D. Braatz
C
C                 Large Scale Systems Research Laboratory
C                      Department of Chemical Engineering
C                University of Illinois at Urbana-Champaign
C                       600 South Mathews Avenue, Box C-3
C                        Urbana, Illinois 61801
C                       http://brahms.scs.uiuc.edu
C
C The modified text is Copyright 1998-2002 by The Board of Trustees 
C of the University of Illinois.  All rights reserved.
C 
C Permission hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal with the Software without restriction, including without limitation
C the rights to use, copy, modify, merge, publish, distribute, sublicense,
C and/or sell copies of the Software, and to permit persons to whom the 
C Software is furnished to do so, subject to the following conditions:
C             1. Redistributions of source code must retain the above copyright
C               notice, this list of conditions and the following disclaimers.
C            2. Redistributions in binary form must reproduce the above 
C               copyright notice, this list of conditions and the following 
C               disclaimers in the documentation and/or other materials 
C               provided with the distribution.
C            3. Neither the names of Large Scale Research Systems Laboratory,
C               University of Illinois, nor the names of its contributors may
C               be used to endorse or promote products derived from this 
C               Software without specific prior written permission.
C
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
C OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
C THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
C OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
C ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
C DEALINGS IN THE SOFTWARE.
C----------------------------------------------------------------------
C
C  Users should cite the original code using the following references:
C
C    [1]	J.J. Downs and E.F. Vogel, "A plant-wide industrial process control 
C    problem." Presented at the AIChE 1990 Annual Meeting, Session on
C    Industrial Challenge Problems in Process Control, Paper #24a
C    Chicago, Illinois, November 14, 1990.
c
C    [2]	J.J. Downs and E.F. Vogel, "A plant-wide industrial process control 
C    problem," Computers and Chemical Engineering, 17:245-255 (1993).
C  
C  Users should cite the modified code using the following references:
C
C    [3]	E.L. Russell, L.H. Chiang, and R.D. Braatz. Data-driven Techniques 
C    for Fault Detection and Diagnosis in Chemical Processes, Springer-Verlag, 
C    London, 2000. 
C
C    [4]	L.H. Chiang, E.L. Russell, and R.D. Braatz. Fault Detection and 
C    Diagnosis in Industrial Systems, Springer-Verlag, London, 2001.  
C
C    [5]	L.H. Chiang, E.L. Russell, and R.D. Braatz. "Fault diagnosis in 
C    chemical processes using Fisher discriminant analysis, discriminant 
C    partial least squares, and principal component analysis," Chemometrics 
C    and Intelligent Laboratory Systems, 50:243-252, 2000. 
C
C    [6]	E.L. Russell, L.H. Chiang, and R.D. Braatz. "Fault detection in 
C    industrial processes using canonical variate analysis and dynamic 
C    principal component analysis," Chemometrics and Intelligent Laboratory 
C    Systems, 51:81-93, 2000. 
C
C
C  Main program for demonstrating application of the modified Tennessee Eastman
C  Process Control Test Problem
C
      PROGRAM TE
      IMPLICIT NONE
C       Show number of features in CSV data file
      LOGICAL WRITE_CSV_NUMFEAT
      PARAMETER(WRITE_CSV_NUMFEAT= .FALSE.)
C      PARAMETER(WRITE_CSV_NUMFEAT= .TRUE.)
C
C Write only the 41 measure variables 'XMEAS' of table 4 and table 5 in [2], or
C include also the 12 controled variables 'XMV' of table table 3 in [2]
      LOGICAL WRITE_CSV_ONLY_MEASURED
      PARAMETER(WRITE_CSV_ONLY_MEASURED= .FALSE.)
C      PARAMETER(WRITE_CSV_ONLY_MEASURED= .TRUE.)
C
      LOGICAL WRITE_ORIGINAL_OUTPUT
C      PARAMETER(WRITE_ORIGINAL_OUTPUT= .FALSE.)
      PARAMETER(WRITE_ORIGINAL_OUTPUT= .TRUE.)

C     In configuration file, 0=Generate randomly
      DOUBLE PRECISION RANDOMSEED
C
C   Maximum number of events in configuration file
      integer MAXEVENTS
      parameter (MAXEVENTS=1000)
      integer EVENTS(MAXEVENTS,3)   ! Step fault on=1/off=0
C  
C  Instructions for running the program
C  ====================================
C
C  1) Go to line 220, change NPTS to the number of data points to simulate. For each
C     minute of operation, 60 points are generated.
      INTEGER NPTS_
C      PARAMETER (NPTS_ = 172800)       ! ORIGINAL
C      PARAMETER (NPTS_ = 385600)
C
C  2) Go to line 226, change SSPTS to the number of data points to simulate in steady
C     state operation before implementing the disturbance.
      INTEGER SSPTS_
C      PARAMETER(SSPTS_ = 3600 * 8)       ! =28800   ORIGINAL
C      PARAMETER(SSPTS_ = 172800)
C
C  3) Go to line 367, implement any of the 21 programmed disturbances. For example, to
C     implement disturbance 2, type IDV(2)=1 .
      INTEGER IDV_(20)   ! Multiple fault vector
C                  1 2 3 4 5 6 7 8 9 1011121314151617181920
C      DATA IDV_  /0,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0/     ! ORIGINAL
C      DATA IDV_  /0,1,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0,0,0,0/
C
C  4) The program will generate 15 output files and all data are recorded every
C     180 seconds, see Table 1 for details.  The default path is the home directory.
C     To change the file name and path, modify lines 346-360 accordingly.  
C     To overwrite the files that already existed, change STATUS='new' to 
C     STATUS='old' from lines 346-360.
C
C
      CHARACTER *80 outfilename
      parameter (outfilename='../out/all.csv')
      CHARACTER *80 CONFIGFNAME
C            
C
C                 Table 1: Content of the output files
C
C      File Name                          Content
C      ---------                          -------
C    TE_data_inc.dat                       Time (in seconds) 
C    TE_data_mv1.dat         Measurements for manipulated variables 1 to 4
C    TE_data_mv2.dat         Measurements for manipulated variables 5 to 8
C    TE_data_mv3.dat         Measurements for manipulated variables 9 to 12
C    TE_data_me01.dat         Measurements for measurement variables 1 to 4
C    TE_data_me02.dat         Measurements for measurement variables 5 to 8
C    TE_data_me03.dat         Measurements for measurement variables 9 to 12
C    TE_data_me04.dat         Measurements for measurement variables 13 to 16
C    TE_data_me05.dat         Measurements for measurement variables 17 to 20
C    TE_data_me06.dat         Measurements for measurement variables 21 to 24
C    TE_data_me07.dat         Measurements for measurement variables 25 to 28
C    TE_data_me08.dat         Measurements for measurement variables 29 to 32
C    TE_data_me09.dat         Measurements for measurement variables 33 to 36
C    TE_data_me10.dat         Measurements for measurement variables 37 to 40
C    TE_data_me11.dat         Measurements for measurement variable 41 
C
C  5) To ensure the randomness of the measurement noises, the random number
C     G in the sub program (teprob.f, line 1187) has to be changed each time before
C     running 'temain_mod.f'.
C
C  6) Save the changes in 'temain_mod.f' and 'teprob.f' and compile the program in
C      unix by typing
C      f77 temain_mod.f teprob.f
C
C  7) Run the program by typing 
C      a.out
C
C
C=============================================================================
C
C
C  MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   DISTURBANCE VECTOR COMMON BLOCK
C
      INTEGER IDV
      COMMON/DVEC/ IDV(20)
C
C   CONTROLLER COMMON BLOCK
C
C      DOUBLE PRECISION SETPT, GAIN, TAUI, ERROLD, DELTAT
C      COMMON/CTRL/ SETPT, GAIN, TAUI, ERROLD, DELTAT
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      INTEGER FLAG
      COMMON/FLAG6/ FLAG
C
      DOUBLE PRECISION GAIN1, ERROLD1
      COMMON/CTRL1/ GAIN1, ERROLD1
      DOUBLE PRECISION GAIN2, ERROLD2
      COMMON/CTRL2/ GAIN2, ERROLD2
      DOUBLE PRECISION GAIN3, ERROLD3
      COMMON/CTRL3/ GAIN3, ERROLD3
      DOUBLE PRECISION  GAIN4, ERROLD4
      COMMON/CTRL4/ GAIN4, ERROLD4
      DOUBLE PRECISION GAIN5, TAUI5, ERROLD5
      COMMON/CTRL5/ GAIN5, TAUI5, ERROLD5
      DOUBLE PRECISION GAIN6, ERROLD6
      COMMON/CTRL6/ GAIN6, ERROLD6
      DOUBLE PRECISION GAIN7, ERROLD7
      COMMON/CTRL7/  GAIN7, ERROLD7
      DOUBLE PRECISION GAIN8, ERROLD8
      COMMON/CTRL8/ GAIN8, ERROLD8
      DOUBLE PRECISION GAIN9, ERROLD9
      COMMON/CTRL9/ GAIN9, ERROLD9
      DOUBLE PRECISION GAIN10, TAUI10, ERROLD10
      COMMON/CTRL10/ GAIN10, TAUI10, ERROLD10
      DOUBLE PRECISION GAIN11, TAUI11, ERROLD11
      COMMON/CTRL11/ GAIN11, TAUI11, ERROLD11
      DOUBLE PRECISION GAIN13, TAUI13, ERROLD13
      COMMON/CTRL13/ GAIN13, TAUI13, ERROLD13
      DOUBLE PRECISION GAIN14, TAUI14, ERROLD14
      COMMON/CTRL14/ GAIN14, TAUI14, ERROLD14
      DOUBLE PRECISION GAIN15, TAUI15, ERROLD15
      COMMON/CTRL15/ GAIN15, TAUI15, ERROLD15
      DOUBLE PRECISION GAIN16, TAUI16, ERROLD16
      COMMON/CTRL16/ GAIN16, TAUI16, ERROLD16
      DOUBLE PRECISION GAIN17, TAUI17, ERROLD17
      COMMON/CTRL17/ GAIN17, TAUI17, ERROLD17
      DOUBLE PRECISION GAIN18, TAUI18, ERROLD18
      COMMON/CTRL18/ GAIN18, TAUI18, ERROLD18
      DOUBLE PRECISION GAIN19, TAUI19, ERROLD19
      COMMON/CTRL19/ GAIN19, TAUI19, ERROLD19
      DOUBLE PRECISION GAIN20, TAUI20, ERROLD20
      COMMON/CTRL20/ GAIN20, TAUI20, ERROLD20
      DOUBLE PRECISION GAIN22, TAUI22, ERROLD22
      COMMON/CTRL22/ GAIN22, TAUI22, ERROLD22
C
C  Local Variables
C
      INTEGER I, J, F, NN, NPTS, SSPTS, TEST, TEST1, TEST3, TEST4
C
      DOUBLE PRECISION TIME, YY(50), YP(50)
      INTEGER NUMTS, TIMESTAMP(MAXEVENTS), NEXTTS, TS
      INTEGER fault, state, numfeat
C
C
      integer nsmp    ! Number of sampled points
      integer argc
      character*80 argv
      integer IARGC

      nsmp = 0

      numfeat = 53
      if( WRITE_CSV_ONLY_MEASURED ) numfeat = 41

      DATA CONFIGFNAME / '../cfg/config.csv' /
      argc = IARGC() ! Number of command line arguments
      if( argc .eq. 0 ) then
          write(*,*) 'Loading default config: ', CONFIGFNAME
      else
          call getarg(1,argv)
          CONFIGFNAME = argv
          write(*,*) 'Loading config:', CONFIGFNAME
      endif
C
C  Read parameters from configuration file
      CALL READCONFIG(CONFIGFNAME,EVENTS,MAXEVENTS,
     +                TIMESTAMP,NUMTS, NPTS_,SSPTS_,IDV_,RANDOMSEED)
C
C
C
C     Generate CSV readable output
      OPEN(UNIT=9,FILE=outfilename,STATUS='replace')
      write(*,*) 'Saving output in ', outfilename
      IF (WRITE_CSV_NUMFEAT) THEN
          WRITE(9,"(1X,I10)") numfeat
      ENDIF

      IF(WRITE_ORIGINAL_OUTPUT) THEN
        write(*,*) 'Saving original TE output in folder ''../out/'''
        OPEN(UNIT=10,FILE='../out/TE_data_inc.dat',STATUS='replace')
        OPEN(UNIT=11,FILE='../out/TE_data_mv1.dat',STATUS='replace')
        OPEN(UNIT=12,FILE='../out/TE_data_mv2.dat',STATUS='replace')
        OPEN(UNIT=13,FILE='../out/TE_data_mv3.dat',STATUS='replace')
        OPEN(UNIT=51,FILE='../out/TE_data_me01.dat',STATUS='replace')
        OPEN(UNIT=52,FILE='../out/TE_data_me02.dat',STATUS='replace')
        OPEN(UNIT=53,FILE='../out/TE_data_me03.dat',STATUS='replace')
        OPEN(UNIT=54,FILE='../out/TE_data_me04.dat',STATUS='replace')
        OPEN(UNIT=55,FILE='../out/TE_data_me05.dat',STATUS='replace')
        OPEN(UNIT=56,FILE='../out/TE_data_me06.dat',STATUS='replace')
        OPEN(UNIT=57,FILE='../out/TE_data_me07.dat',STATUS='replace')
        OPEN(UNIT=58,FILE='../out/TE_data_me08.dat',STATUS='replace')
        OPEN(UNIT=59,FILE='../out/TE_data_me09.dat',STATUS='replace')
        OPEN(UNIT=60,FILE='../out/TE_data_me10.dat',STATUS='replace')
        OPEN(UNIT=61,FILE='../out/TE_data_me11.dat',STATUS='replace')
      ENDIF
C
C
C  Set the number of differential equations (states).  The process has 50
C  states.  If the user wishes to integrate additional states, NN must be
C  increased by the number of additional differential equations.
C
      NN = 50
C
C  Set the number of points to simulate
C
      NPTS = NPTS_

C
C  Set the number of pints to simulate in steady state operation
C

      SSPTS = SSPTS_


C
C  Integrator Step Size:  1 Second Converted to Hours
C
      DELTAT = 1. / 3600.
C
C  Initialize Process
C  (Sets TIME to zero)
C
      CALL TEINIT(NN,TIME,YY,YP,RANDOMSEED)
C
C  Set Controller Parameters
C  Make a Stripper Level Set Point Change of +15%
C
CC      SETPT = XMEAS(15) + 15.0
CC      GAIN = 2.0
CC      TAUI = 5.0
CC      ERROLD = 0.0
      SETPT(1)=3664.0        
      GAIN1=1.0
      ERROLD1=0.0
      SETPT(2)=4509.3
      GAIN2=1.0
      ERROLD2=0.0
      SETPT(3)=.25052
      GAIN3=1.
      ERROLD3=0.0
      SETPT(4)=9.3477
      GAIN4=1.
      ERROLD4=0.0
      SETPT(5)=26.902
      GAIN5=-0.083          
      TAUI5=1./3600.   
      ERROLD5=0.0
      SETPT(6)=0.33712  
      GAIN6=1.22                     
      ERROLD6=0.0
      SETPT(7)=50.0
      GAIN7=-2.06      
      ERROLD7=0.0
      SETPT(8)=50.0
      GAIN8=-1.62      
      ERROLD8=0.0
      SETPT(9)=230.31
      GAIN9=0.41          
      ERROLD9=0.0      
      SETPT(10)=94.599
      GAIN10= -0.156     * 10.
      TAUI10=1452./3600. 
      ERROLD10=0.0
      SETPT(11)=22.949    
      GAIN11=1.09        
      TAUI11=2600./3600.
      ERROLD11=0.0
      SETPT(13)=32.188
      GAIN13=18.              
      TAUI13=3168./3600.   
      ERROLD13=0.0
      SETPT(14)=6.8820
      GAIN14=8.3        
      TAUI14=3168.0/3600.
      ERROLD14=0.0
      SETPT(15)=18.776                     
      GAIN15=2.37              
      TAUI15=5069./3600.    
      ERROLD15=0.0
      SETPT(16)=65.731
      GAIN16=1.69        / 10.
      TAUI16=236./3600.
      ERROLD16=0.0
      SETPT(17)=75.000
      GAIN17=11.1      / 10.
      TAUI17=3168./3600.  
      ERROLD17=0.0        
      SETPT(18)=120.40
      GAIN18=2.83      * 10.
      TAUI18=982./3600.
      ERROLD18=0.0
      SETPT(19)=13.823
      GAIN19=-83.2        / 5. /3.  
      TAUI19=6336./3600. 
      ERROLD19=0.0
      SETPT(20)=0.83570  
      GAIN20=-16.3       / 5.         
      TAUI20=12408./3600.  
      ERROLD20=0.0
      SETPT(12)=2633.7
      GAIN22=-1.0        * 5.         
      TAUI22=1000./3600.  
      ERROLD22=0.0
C
C    Example Disturbance:
C    Change Reactor Cooling
C
C      if (.false.) then
      XMV(1) = 63.053 + 0.
      XMV(2) = 53.980 + 0.
      XMV(3) = 24.644 + 0.    
      XMV(4) = 61.302 + 0.
      XMV(5) = 22.210 + 0.
      XMV(6) = 40.064 + 0.
      XMV(7) = 38.100 + 0.
      XMV(8) = 46.534 + 0.
      XMV(9) = 47.446 + 0.
      XMV(10)= 41.106 + 0.
      XMV(11)= 18.114 + 0.
C      endif
C
C      SETPT(6)=SETPT(6) + 0.2
C
C  Set all Disturbance Flags to OFF
C
      DO 100 I = 1, 20
          IDV(I) = 0
 100  CONTINUE
C

C
C
C  Simulation Loop
C
      write(*,*) 'Starting simulation...'
      do i=1,numts
         write(*,'(1x,A,I4,A,I8,A,I3,A,I3)') 'Timestamp ', i,
     +   ' =', TIMESTAMP(i),
     +   ' fault=',events(i,2),' state=',events(i,3)
      end do
      TS = -1
      NEXTTS = -1
      if( NUMTS .ge. 1 ) then
          TS = 1
          NEXTTS = TIMESTAMP(TS)
      endif

      DO 1000 I = 1, NPTS
C          write(0,"(1x,A16,I8,A4,I8,A28,A9)",advance='no') ! Only Fortran 95
          write(0,"(1x,A16,I8,A4,I8,A28,A9,$)")
     +  'Simulation step ', i, ' of ', NPTS,
     + '\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b',
     + '\b\b\b\b\b\b\b\b\b\b\b'

C Check if there is an event happens at time i
          do while( NEXTTS .eq. I )
C             write(*,*) 'nextts=',nextts,'i=',i
              if( NEXTTS .ne. EVENTS(TS,1) ) then
                  write(0,*)  'Invalid time stamp'
                  write(0,*)  NEXTTS, ' <>  ', EVENTS(TS,1)
                  stop
              else
                 fault = EVENTS(TS,2)
                 state = EVENTS(TS,3)
                 IDV(fault) = state
                 TS = TS + 1
                 if( TS .gt. NUMTS ) then
                     NEXTTS = -1
                 else
                     NEXTTS = TIMESTAMP(TS)
                 endif
                 write(*,'(1X,A,I8,A,I7,A,I5,A,I5,A,I8)')
     +               'Event at time ',i,' fault=',fault,
     +               ' state=',state,' ts#=',ts-1,' nextts=',nextts
              endif
           enddo

C          IF (I.GE.SSPTS) THEN
C                 DO F = 1, 20
C                      IDV(F) = IDV_(F)
C                 END DO
C          ENDIF



      TEST=MOD(I,3)
      IF (TEST.EQ.0) THEN
            CALL CONTRL1
            CALL CONTRL2
            CALL CONTRL3
            CALL CONTRL4
            CALL CONTRL5
            CALL CONTRL6
            CALL CONTRL7
            CALL CONTRL8
            CALL CONTRL9
            CALL CONTRL10
            CALL CONTRL11
            CALL CONTRL16
            CALL CONTRL17
            CALL CONTRL18
        ENDIF
        TEST1=MOD(I,360)
        IF (TEST1.EQ.0) THEN
            CALL CONTRL13
            CALL CONTRL14
            CALL CONTRL15
            CALL CONTRL19
        ENDIF
        TEST1=MOD(I,900)
        IF (TEST1.EQ.0) CALL CONTRL20
        TEST3=MOD(I,5000)       
        IF (TEST3.EQ.0) THEN
C            PRINT *, 'Simulation time (in seconds) = ', I
C
C          write(0,"(1x,A31,I8,A28,A13,$)")
C     +  'Simulation time (in seconds) = ', I,
C     + '\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b',
C     + '\b\b\b\b\b\b\b\b\b\b\b\b\b'
        ENDIF
C
      TEST4=MOD(I,180)  ! write each 180 steps
      IF (TEST4.EQ.0) THEN
            nsmp = nsmp + 1   ! Increment number of sampled points
            write(*,"(1X,A,I8,A,I8,A,A1,20(I1))") "Sampling point #",
     +            nsmp, " at t=", I, " sec  Class=","C",(IDV(J),J=1,20)
C            WRITE(*,"(1X,A1,20(I1),A3)"), "C",(IDV(J),J=1,20), "   " ! DEBUG
C            READ(*,*)

            CALL OUTPUTNEW(WRITE_CSV_ONLY_MEASURED)
            IF(WRITE_ORIGINAL_OUTPUT) THEN
                  CALL OUTPUT
            ENDIF
            WRITE(10,111) I
 111        FORMAT(1X,I6)
      ENDIF
C
      CALL INTGTR(NN,TIME,DELTAT,YY,YP)
C
      CALL CONSHAND
C
 1000 CONTINUE
      PRINT *, ''
      PRINT *, 'Simulation is done. '

C
      CLOSE(UNIT=9)

      IF(WRITE_ORIGINAL_OUTPUT) THEN
        CLOSE(UNIT=10)
        CLOSE(UNIT=11)
        CLOSE(UNIT=12)
        CLOSE(UNIT=13)
        CLOSE(UNIT=51)
        CLOSE(UNIT=52)
        CLOSE(UNIT=53)
        CLOSE(UNIT=54)
        CLOSE(UNIT=55)
        CLOSE(UNIT=56)
        CLOSE(UNIT=57)
        CLOSE(UNIT=58)
        CLOSE(UNIT=59)
        CLOSE(UNIT=60)
        CLOSE(UNIT=61)
      ENDIF
      END
C
C=============================================================================
C
CC      SUBROUTINE CONTRL
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
CC      DOUBLE PRECISION XMEAS, XMV
CC      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
CC      DOUBLE PRECISION SETPT, GAIN, TAUI, ERROLD, DELTAT
CC      COMMON/CTRL/ SETPT, GAIN, TAUI, ERROLD, DELTAT
C
CC      DOUBLE PRECISION ERR, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
CC      ERR = SETPT - XMEAS(15)
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
CC      DXMV = GAIN * ( ( ERR - ERROLD ) + ERR * DELTAT * 60. / TAUI )
C
CC      XMV(8) = XMV(8) - DXMV
C
CC      ERROLD = ERR
C
CC      RETURN
CC      END
C
C=============================================================================
C
      SUBROUTINE CONTRL1
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN1, ERROLD1
      COMMON/CTRL1/ GAIN1, ERROLD1
C
      DOUBLE PRECISION ERR1, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR1 = (SETPT(1) - XMEAS(2)) * 100. / 5811.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN1 * ( ( ERR1 - ERROLD1 ) )
C
      XMV(1) = XMV(1) + DXMV
C
      ERROLD1 = ERR1
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL2
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN2, ERROLD2
      COMMON/CTRL2/ GAIN2, ERROLD2
C
      DOUBLE PRECISION ERR2, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR2 = (SETPT(2) - XMEAS(3)) * 100. / 8354. 
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN2 * ( ( ERR2 - ERROLD2 ) )
C
      XMV(2) = XMV(2) + DXMV
C
      ERROLD2 = ERR2
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL3
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN3, ERROLD3
      COMMON/CTRL3/ GAIN3, ERROLD3
C
      DOUBLE PRECISION ERR3, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR3 = (SETPT(3) - XMEAS(1)) * 100. / 1.017
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN3 * ( ( ERR3 - ERROLD3 ) )
C
      XMV(3) = XMV(3) + DXMV
C
      ERROLD3 = ERR3
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL4
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN4, ERROLD4
      COMMON/CTRL4/ GAIN4, ERROLD4
C
      DOUBLE PRECISION ERR4, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR4 = (SETPT(4) - XMEAS(4)) * 100. / 15.25
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN4 * ( ( ERR4 - ERROLD4 ) )
C
      XMV(4) = XMV(4) + DXMV
C
      ERROLD4 = ERR4
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL5
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN5, TAUI5, ERROLD5
      COMMON/CTRL5/ GAIN5, TAUI5, ERROLD5
C
      DOUBLE PRECISION ERR5, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR5 = (SETPT(5) - XMEAS(5))  * 100. / 53.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
C       PRINT *, 'GAIN5= ', GAIN5
C      PRINT *, 'TAUI5= ', TAUI5
C      PRINT *, 'ERR5= ', ERR5
C      PRINT *, 'ERROLD5= ', ERROLD5     
C
      DXMV = GAIN5 * ((ERR5 - ERROLD5)+ERR5*DELTAT*3./TAUI5)
C
      XMV(5) = XMV(5) + DXMV
C
      ERROLD5 = ERR5
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL6
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
      INTEGER FLAG
       COMMON/FLAG6/ FLAG
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN6, ERROLD6
      COMMON/CTRL6/ GAIN6, ERROLD6
C
      DOUBLE PRECISION ERR6, DXMV
C
C  Example PI Controller:
C     Stripper Level Controller
      IF (XMEAS(13).GE.2950.0) THEN
            XMV(6)=100.0
            FLAG=1
      ELSEIF (FLAG.EQ.1.AND.XMEAS(13).GE.2633.7) THEN
            XMV(6)=100.0
      ELSEIF (FLAG.EQ.1.AND.XMEAS(13).LE.2633.7) THEN
            XMV(6)=40.060
C            write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  6 CHANGE:',
C     +                SETPT(6),'-->',0.33712
            SETPT(6)=0.33712
            ERROLD6=0.0
             FLAG=0
      ELSEIF (XMEAS(13).LE.2300.) THEN
            XMV(6)=0.0
            FLAG=2
      ELSEIF (FLAG.EQ.2.AND.XMEAS(13).LE.2633.7) THEN
            XMV(6)=0.0
      ELSEIF (FLAG.EQ.2.AND.XMEAS(13).GE.2633.7) THEN
            XMV(6)=40.060
C            write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  6 CHANGE:',
C     +                SETPT(6),'-->',0.33712
            SETPT(6)=0.33712
            ERROLD6=0.0
            FLAG=0
      ELSE      
            FLAG=0
C
C    Calculate Error
C
       ERR6 = (SETPT(6) - XMEAS(10)) * 100. /1.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
C      PRINT *, 'XMV(6)= ', XMV(6)
      DXMV = GAIN6 * ( ( ERR6 - ERROLD6 ) )
C
C       PRINT *, 'GAIN6= ', GAIN6
C      PRINT *, 'SETPT(6)= ', SETPT(6)      
C      PRINT *, 'XMEAS(10)= ', XMEAS(10)     
      XMV(6) = XMV(6) + DXMV
C
C       PRINT *, 'ERROLD6= ', ERROLD6     
C      PRINT *, 'ERR6= ', ERR6
C      PRINT *, 'XMV(6)== ', XMV(6)
      ERROLD6 = ERR6
      ENDIF
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL7
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN7, ERROLD7
      COMMON/CTRL7/ GAIN7, ERROLD7
C
      DOUBLE PRECISION ERR7, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR7 = (SETPT(7) - XMEAS(12)) * 100. / 70.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN7 * ( ( ERR7 - ERROLD7 ) )
C
      XMV(7) = XMV(7) + DXMV
C
      ERROLD7 = ERR7
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL8
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN8, ERROLD8
      COMMON/CTRL8/ GAIN8, ERROLD8
C
      DOUBLE PRECISION ERR8, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR8 = (SETPT(8) - XMEAS(15)) * 100. / 70.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV =  GAIN8 * ( ( ERR8 - ERROLD8 ) )
C
      XMV(8) = XMV(8) + DXMV
C
      ERROLD8 = ERR8
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL9
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN9, ERROLD9
      COMMON/CTRL9/ GAIN9, ERROLD9
C
      DOUBLE PRECISION ERR9, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR9 = (SETPT(9) - XMEAS(19)) * 100. / 460. 
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN9 * ( ( ERR9 - ERROLD9 ) )
C
      XMV(9) = XMV(9) + DXMV
C
      ERROLD9 = ERR9
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL10
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN10, TAUI10, ERROLD10
      COMMON/CTRL10/ GAIN10, TAUI10, ERROLD10
C
      DOUBLE PRECISION ERR10, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR10 = (SETPT(10) - XMEAS(21)) * 100. / 150.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN10*((ERR10 - ERROLD10)+ERR10*DELTAT*3./TAUI10)
C
      XMV(10) = XMV(10) + DXMV
C
      ERROLD10 = ERR10
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL11
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN11, TAUI11, ERROLD11
      COMMON/CTRL11/ GAIN11, TAUI11, ERROLD11
C
      DOUBLE PRECISION ERR11, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR11 = (SETPT(11) - XMEAS(17)) * 100. / 46.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN11*((ERR11 - ERROLD11)+ERR11*DELTAT*3./TAUI11)
C
      XMV(11) = XMV(11) + DXMV
C
      ERROLD11 = ERR11
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL13
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN13, TAUI13, ERROLD13
      COMMON/CTRL13/ GAIN13, TAUI13, ERROLD13
C
      DOUBLE PRECISION ERR13, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR13 = (SETPT(13) - XMEAS(23)) * 100. / 100.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN13 * ((ERR13 - ERROLD13)+ERR13*DELTAT*360./TAUI13)
C
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  3 CHANGE:',
C     +                SETPT(3),'-->',SETPT(3) + DXMV * 1.017 / 100.
      SETPT(3) = SETPT(3) + DXMV * 1.017 / 100.
C
      ERROLD13 = ERR13
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL14
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN14, TAUI14, ERROLD14
      COMMON/CTRL14/ GAIN14, TAUI14, ERROLD14
C
      DOUBLE PRECISION ERR14, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR14 = (SETPT(14) - XMEAS(26)) * 100. /100.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN14*((ERR14 - ERROLD14)+ERR14*DELTAT*360./TAUI14)
C
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  1 CHANGE:',
C     +                SETPT(1),'-->',SETPT(1) + DXMV * 5811. / 100.
      SETPT(1) = SETPT(1) + DXMV * 5811. / 100.
C
      ERROLD14 = ERR14
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL15
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN15, TAUI15, ERROLD15
      COMMON/CTRL15/ GAIN15, TAUI15, ERROLD15
C
      DOUBLE PRECISION ERR15, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR15 = (SETPT(15) - XMEAS(27)) * 100. / 100.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN15 * ((ERR15 - ERROLD15)+ERR15*DELTAT*360./TAUI15)
C
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  2 CHANGE:',
C     +                SETPT(2),'-->',SETPT(2) + DXMV * 8354. / 100.
      SETPT(2) = SETPT(2) + DXMV * 8354. / 100.
C
      ERROLD15 = ERR15
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL16
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN16, TAUI16, ERROLD16
      COMMON/CTRL16/ GAIN16, TAUI16, ERROLD16
C
      DOUBLE PRECISION ERR16, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR16 = (SETPT(16) - XMEAS(18)) * 100. / 130.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN16 * ((ERR16 - ERROLD16)+ERR16*DELTAT*3./TAUI16)
C
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  9 CHANGE:',
C     +                SETPT(9),'-->',SETPT(9) + DXMV * 460. / 100.
      SETPT(9) = SETPT(9) + DXMV * 460. / 100.
C
      ERROLD16 = ERR16
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL17
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN17, TAUI17, ERROLD17
      COMMON/CTRL17/ GAIN17, TAUI17, ERROLD17
C
      DOUBLE PRECISION ERR17, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR17 = (SETPT(17) - XMEAS(8)) * 100. / 50.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV =GAIN17*((ERR17 - ERROLD17)+ERR17*DELTAT*3./TAUI17)
C
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  4 CHANGE:',
C     +                SETPT(4),'-->',SETPT(4) + DXMV * 15.25 / 100.
      SETPT(4) = SETPT(4) + DXMV * 15.25 / 100.
C
      ERROLD17 = ERR17
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL18
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN18, TAUI18, ERROLD18
      COMMON/CTRL18/ GAIN18, TAUI18, ERROLD18
C
      DOUBLE PRECISION ERR18, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR18 = (SETPT(18) - XMEAS(9)) * 100. / 150. 
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN18 * ((ERR18 - ERROLD18)+ERR18*DELTAT*3./TAUI18)
C
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT 10 CHANGE:',
C     +                SETPT(10),'-->',SETPT(10) + DXMV * 150. / 100.
      SETPT(10) = SETPT(10) + DXMV * 150. / 100.
C
      ERROLD18 = ERR18
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL19
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN19, TAUI19, ERROLD19
      COMMON/CTRL19/ GAIN19, TAUI19, ERROLD19
C
      DOUBLE PRECISION ERR19, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR19 = (SETPT(19) - XMEAS(30)) * 100. / 26.
C      PRINT *, 'ERROLD19= ', ERROLD19
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN19*((ERR19 - ERROLD19)+ERR19*DELTAT*360./TAUI19)
C
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  6 CHANGE:',
C     +                SETPT(6),'-->', SETPT(6) + DXMV * 1. / 100.
      SETPT(6) = SETPT(6) + DXMV * 1. / 100.
C      PRINT *, 'SETPT(6)= ', SETPT(6)
C
      ERROLD19 = ERR19
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL20
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN20, TAUI20, ERROLD20
      COMMON/CTRL20/  GAIN20, TAUI20, ERROLD20
C    
      DOUBLE PRECISION ERR20, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR20 = (SETPT(20) - XMEAS(38)) * 100. / 1.6
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN20*((ERR20 - ERROLD20)+ERR20*DELTAT*900./TAUI20)
C
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT 16 CHANGE:',
C     +                SETPT(16),'-->',SETPT(16) + DXMV  * 130. / 100.
      SETPT(16) = SETPT(16) + DXMV  * 130. / 100.
C
      ERROLD20 = ERR20
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL22
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN22, TAUI22, ERROLD22
      COMMON/CTRL22/  GAIN22, TAUI22, ERROLD22
C    
      DOUBLE PRECISION ERR22, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR22 = SETPT(12) - XMEAS(13)
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN22*((ERR22 - ERROLD22)+ERR22*DELTAT*3./TAUI22)
C
      XMV(6) = XMV(6) + DXMV
C
      ERROLD22 = ERR22
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE OUTPUTNEW(WRITE_CSV_ONLY_MEASURED)
      IMPLICIT NONE
      LOGICAL WRITE_CSV_ONLY_MEASURED
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
      INTEGER IDV
      COMMON/DVEC/ IDV(20)

      INTEGER I
      CHARACTER SEP, VIRG, PTVIRG, TAB
      PARAMETER (VIRG=',', PTVIRG=';')
      TAB=char(9)
      SEP = TAB
C      SEP = PTVIRG

C
      if( WRITE_CSV_ONLY_MEASURED ) then
          WRITE(9,"(1X,41(1P,E12.4,A1),A1,20(I1))")
     .        (XMEAS(I),SEP,I=1,41),"C",(IDV(I),I=1,20)
      else
C http://www.personal.psu.edu/jhm/f90/lectures/23.html  (shift leading digit with the 'P' descriptor)
          WRITE(9,"(1X,12(1P,E12.4,A1),41(1P,E12.4,A1),A1,20(I1))")
     .        (XMEAS(I),SEP,I=1,41), (XMV(I),SEP,I=1,12),
     .        "C",(IDV(I),I=1,20)
      endif

      RETURN
      END
C
C=============================================================================
C
      
      SUBROUTINE READCONFIG(FNAME,EVENTS,MAXEVENTS,
     +    TIMESTAMP,NUMTS,NPTS,SSPTS,IDV,RANDOMSEED)
      IMPLICIT NONE
      CHARACTER *(*) FNAME
      DOUBLE PRECISION RANDOMSEED
      INTEGER MAXEVENTS,NPTS,SSPTS,IDV(20)
      INTEGER EVENTS(MAXEVENTS,3)
      INTEGER NUMTS, TIMESTAMP(MAXEVENTS)
C
      integer i, u, eventcnt, fault, state, ts, lastts
      parameter (u=99)
      character*80 linebuf
      character f

      open( unit=u, file=FNAME, status='old' )
      write(*,*) 'Reading configuration from file ''',FNAME,''''
      read( u, * ) NPTS
      write(*,*) 'NPTS=', NPTS
      read( u, * ) RANDOMSEED
      write(*,*) 'RANDOMSEED=', RANDOMSEED

C    EOF only works with MS-Fortran
C       do while (.not. eof(u))
C         if (.not. eof(u)) then

      eventcnt = 0
      lastts = -1
      do while (.true.)
          read( u, '(A79)', end=99 ) linebuf
C         write( 0, * )  'Read: >>>', linebuf, '<<<'
          i = 1
          do while( linebuf(i:i) .eq. ' ')
              i = i+1
          end do
          if( linebuf(i:i) .eq. '#' ) then

          else
             read( linebuf, *, end=99) ts, fault, state
             write( 0, * )  'timestamp=', ts, ' fault=', fault,
     +        ' state=', state, ' lasttimestamp=', lastts
              if( ts .lt. 0 .or. ts .gt. NPTS ) then
                  write( 0, * )  'Invalid time stamp. Exit...',
     +                ts
                  stop
              endif
              if( ts .lt. lastts ) then
                  write(0,*)  'Invalid time stamp. Cannot be older.'
                  write(0,*)  ts, ' =last < new= ', lastts
                  stop
              else
                  lastts = ts
              endif
              if( fault .le. 0 .or. fault .gt. 20 ) then
                  write( 0, * )  'Invalid fault number. Exit...', fault
                  stop
              endif
              if( state .ne. 0 .and. state .ne. 1 ) then
                  write( 0, * )  'Invalid event state. Exit...', state
                  stop
              endif
              eventcnt = eventcnt + 1
              events(eventcnt,1) = ts
              events(eventcnt,2) = fault
              events(eventcnt,3) = state
              TIMESTAMP(eventcnt) = ts
          endif
          if (eventcnt .ge. MAXEVENTS ) then
             write( 0, * )  'Too many events. Exit...'
             stop
          else
          endif
* Inserir na tabela
C          EVENTS( codmat, sem, mes ) = quant
      end do

99    close( unit=u )
      write(*,*) 'Read', eventcnt, ' events from ', FNAME,'.'
      numts = eventcnt
C
C     stop
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE OUTPUT
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
      WRITE(11,100) XMV(1), XMV(2), XMV(3), XMV(4)
      WRITE(12,100) XMV(5), XMV(6), XMV(7), XMV(8)
      WRITE(13,100) XMV(9), XMV(10), XMV(11), XMV(12)
      WRITE(51,100) XMEAS(1), XMEAS(2), XMEAS(3), XMEAS(4)
      WRITE(52,100) XMEAS(5), XMEAS(6), XMEAS(7), XMEAS(8)
      WRITE(53,100) XMEAS(9), XMEAS(10), XMEAS(11), XMEAS(12)
      WRITE(54,100) XMEAS(13), XMEAS(14), XMEAS(15), XMEAS(16)
      WRITE(55,100) XMEAS(17), XMEAS(18), XMEAS(19), XMEAS(20)
      WRITE(56,100) XMEAS(21), XMEAS(22), XMEAS(23), XMEAS(24)
      WRITE(57,100) XMEAS(25), XMEAS(26), XMEAS(27), XMEAS(28)
      WRITE(58,100) XMEAS(29), XMEAS(30), XMEAS(31), XMEAS(32)
      WRITE(59,100) XMEAS(33), XMEAS(34), XMEAS(35), XMEAS(36)
      WRITE(60,100) XMEAS(37), XMEAS(38), XMEAS(39), XMEAS(40)
      WRITE(61,300) XMEAS(41)
 100  FORMAT(1X,E13.5,2X,E13.5,2X,E13.5,2X,E13.5)
 200  FORMAT(1X,E13.5,2X,E13.5,2X,E13.5)
 300  FORMAT(1X,E13.5)
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE INTGTR(NN,TIME,DELTAT,YY,YP)
C
C  Euler Integration Algorithm
C
C
      INTEGER I, NN
C
      DOUBLE PRECISION TIME, DELTAT, YY(NN), YP(NN)
C
      CALL TEFUNC(NN,TIME,YY,YP)
C
      TIME = TIME + DELTAT
C
      DO 100 I = 1, NN
C
          YY(I) = YY(I) + YP(I) * DELTAT 
C
 100  CONTINUE
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONSHAND
C
C  Euler Integration Algorithm
C
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
      INTEGER I
C      
      DO 100 I=1, 11
          IF (XMV(I).LE.0.0) XMV(I)=0.
              IF (XMV(I).GE.100.0) XMV(I)=100.
 100  CONTINUE
C
      RETURN
      END
