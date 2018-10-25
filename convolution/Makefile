EXEC=convolution
OBJDIR=./

CC=gcc
CPP=g++
NVCC=nvcc

CFLAGS=-O3
LDFLAGS_CC=
LDFLAGS_NVCC=-lopencv_imgproc -lopencv_highgui -lopencv_core -lcublas
OBJ=convolution.o

EXECOBJ = $(addprefix $(OBJDIR), $(OBJ))

all: $(EXECOBJ)

$(OBJDIR)%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS_CC)

$(OBJDIR)%.o: %.cpp
	$(CPP) $(CFLAGS) -c $< -o $@ $(LDFLAGS_CC)

$(OBJDIR)%.o: %.cu
	$(NVCC) --ptx "$(CFLAGS)" $< $(LDFLAGS_NVCC)
	$(NVCC) "$(CFLAGS)" $< -o $@ $(LDFLAGS_NVCC)

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXECOBJ)

