#######################################################################
# Global makefile. Contains rules for creating different .exe files.  #
# These versions are named as such:				      #
#	dcpu: double-precision CPU				      #
#	fcpu: single-precision CPU				      #
#	dgpu: double-precision GPU				      #
#	fgpu: single-precision GPU				      #
# 								      #
# For .exe specific build rules, please see the individual makefiles. # 
# 								      #
# NOTE: in order to compile and run the GPU versions, the nvcc	      #
# compiler is required.						      #
#								      #
# Author: Arend Moerman (A.Moerman-1@tudelft.nl)		      #
#######################################################################
dcpu:
	make -f makefile.dcpu 
#######################################################################
fcpu:
	make -f makefile.fcpu 
#######################################################################
dgpu:
	make -f makefile.dgpu
#######################################################################	
fgpu:
	make -f makefile.fgpu
#######################################################################	
all:
	make -f makefile.dcpu 
	make -f makefile.fcpu 
	make -f makefile.dgpu 
	make -f makefile.fgpu
#######################################################################	
clean:
	rm -f *~
	rm -f *.o
	rm -f *.exe
	rm -f -r ccache
	rm -f .fuse*
	rm -f *.out
	rm -f *.h.gch
	rm -f *.ptx
#######################################################################
