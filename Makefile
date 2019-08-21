.POSIX:
.SUFFIXES: .cpp .hpp .h .cu .o .d .asm

include config.mk

example:
	make -C example

install:
	cp -r blaze_cuda $(PREFIX)/include

uninstall:
	rm -rf $(PREFIX)/include/blaze_cuda

.PHONY: install uninstall example
