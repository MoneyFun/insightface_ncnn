CXX = g++
COMMON = `pkg-config --cflags opencv`
LIB = ../lib/ncnn/libncnn.a
LIB += `pkg-config --libs opencv`
INCLUDE = -I../include/ncnn
COMMON += -fopenmp -pthread -std=c++11
DEPS = $(wildcard *.h)
OBJ = base.o mtcnn.o arcface.o
all : main
main : main.cpp $(OBJ)
	$(CXX) $(COMMON) $(INCLUDE) $^ -o $@ $(LIB) -L../lib/ncnn/ippicv_lnx/lib/intel64
%.o : %.cpp $(DEPS)
	$(CXX) $(COMMON) $(INCLUDE) -c $< -o $@
.PHONY : clean
clean :
	rm -rf $(OBJ) main

