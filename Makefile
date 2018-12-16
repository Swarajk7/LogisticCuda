CC		:= g++
C_FLAGS := -std=c++11 -Wall -Wextra
NVCC        = nvcc

NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\" -std=c++11
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

BIN		:= bin
SRC		:= src
INCLUDE	:= include
LIB		:= lib

LIBRARIES	:=

ifeq ($(OS),Windows_NT)
EXECUTABLE	:= main.exe
else
EXECUTABLE	:= main
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
OBJ	        = $(BIN)/data_reader_cpp.o $(BIN)/utils.o $(BIN)/main.o $(BIN)/logistic.o $(BIN)/gpu_classification_model_cu.o

all: $(BIN)/$(EXECUTABLE)

clean:
	$(RM) $(BIN)/*

run: all
	./$(BIN)/$(EXECUTABLE)

#$(BIN)/$(EXECUTABLE): $(SRC)/*
#	$(CC) $(C_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)

$(BIN)/data_reader_cpp.o: $(SRC)/data_reader.cpp
	$(NVCC) -c -o $@ $(SRC)/data_reader.cpp $(NVCC_FLAGS)

$(BIN)/utils.o: $(SRC)/utils.cpp
	$(NVCC) -c -o $@ $(SRC)/utils.cpp $(NVCC_FLAGS)

$(BIN)/main.o: $(SRC)/main.cpp 
	$(NVCC) -c -o $@ $(SRC)/main.cpp $(NVCC_FLAGS)

$(BIN)/logistic.o: $(SRC)/logistic.cpp
	$(NVCC) -c -o $@ $(SRC)/logistic.cpp $(NVCC_FLAGS)

$(BIN)/gpu_classification_model_cu.o: $(SRC)/gpu_classification_model.cu $(SRC)/logistic_regression_kernels.cu $(SRC)/support.h $(SRC)/gpu_data_handling.h
	$(NVCC) -c -o $@ $(SRC)/gpu_classification_model.cu $(NVCC_FLAGS)

$(BIN)/$(EXECUTABLE): $(OBJ)
	$(NVCC) $(OBJ) -o $@ $(LD_FLAGS) $(NVCC_FLAGS)	
#$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)
