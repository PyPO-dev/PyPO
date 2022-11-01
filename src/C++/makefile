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
# Author of src: Arend Moerman (A.Moerman-1@tudelft.nl)		      #
refl:
	make -f makefile.beam
	make -f makefile.refl
#######################################################################
cpu:
	make -f makefile.beam
	make -f makefile.refl
	make -f makefile.cpu
#######################################################################
gpu:
	make -f makefile.beam
	make -f makefile.refl
	make -f makefile.gpu
#######################################################################
all:
	make -f makefile.beam
	make -f makefile.refl
	make -f makefile.cpu
	make -f makefile.gpu
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
	rm -f *.so
#######################################################################
