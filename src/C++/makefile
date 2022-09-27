# Global makefile. Contains rules for creating different .exe files

dcpu:
	make -f makefile.dcpu 
	
fcpu:
	make -f makefile.fcpu 
	
dgpu:
	make -f makefile.dgpu
	
fgpu:
	make -f makefile.fgpu
	
all:
	make -f makefile.dcpu 
	make -f makefile.fcpu 
	make -f makefile.dgpu 
	make -f makefile.fgpu
	
clean:
	rm -f *~
	rm -f *.o
	rm -f *.exe
	rm -f -r ccache
	rm -f .fuse*
	rm -f *.out
	rm -f *.h.gch
	rm -f *.ptx
