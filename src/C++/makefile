######################################################################################
DIRSRC = ./src/
######################################################################################
COMP = g++
CFLAGS = -O3 -std=c++11 -pthread
CC  = $(COMP) $(CFLAGS) -c
CO  = $(COMP) $(CFLAGS) -o
######################################################################################
EXE = PhysBeam.exe
all: $(EXE)
######################################################################################
PhysBeam.exe: PhysBeam.o Propagation.o DataHandler.o
	$(CO) PhysBeam.exe PhysBeam.o Propagation.o DataHandler.o
######################################################################################
PhysBeam.o: $(DIRSRC)PhysBeam.cpp Propagation.o DataHandler.o
	$(CC) $(DIRSRC)PhysBeam.cpp -I$(DIRSRC)
Propagation.o: $(DIRSRC)Propagation.h $(DIRSRC)Propagation.cpp 
	$(CC) $(DIRSRC)Propagation.cpp -I$(DIRSRC)
DataHandler.o: $(DIRSRC)DataHandler.h $(DIRSRC)DataHandler.cpp
	$(CC) $(DIRSRC)DataHandler.cpp -I$(DIRSRC)
######################################################################################
clean:
	rm -f *~
	rm -f *.o
	rm -f *.exe
	rm -f -r ccache
	rm -f .fuse*
	rm -f *.out
	rm -f $(DIRSRC)*.h.gch
######################################################################################
