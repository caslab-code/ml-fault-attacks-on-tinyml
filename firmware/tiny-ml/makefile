# Hey Emacs, this is a -*- makefile -*-
#----------------------------------------------------------------------------
#
# Makefile for the TinyML ChipWhisperer target
# (builds the four TinyML models and their RSR-protected variants)
#
#----------------------------------------------------------------------------
# On command line:
#
# make all = Make software.
#
# make clean = Clean out built project files.
#
# make coff = Convert ELF to AVR COFF.
#
# make extcoff = Convert ELF to AVR Extended COFF.
#
# make program = Download the hex file to the device, using avrdude.
#                Please customize the avrdude settings below first!
#
# make debug = Start either simulavr or avarice as specified for debugging,
#              with avr-gdb or avr-insight as the front end for debugging.
#
# make filename.s = Just compile filename.c into the assembler code only.
#
# make filename.i = Create a preprocessed source file for use in submitting
#                   bug reports to the GCC project.
#
# To rebuild project do "make clean" then "make all".
#----------------------------------------------------------------------------

# Target file name (without extension).
# This is the name of the compiled .hex file.
TARGET = tiny-ml

# List C source files here.
# Header files (.h) are automatically pulled in.
SRC += tiny-ml.c

SS_VER = SS_VER_2_1

#Add simpleserial project to build
include ../simpleserial/Makefile.simpleserial

FIRMWAREPATH = ../.

###############################################
# paths
FAST_GRNN_PATH              := $(FIRMWAREPATH)/fastgrnn
LOGISTIC_REGRESSION_PATH    := $(FIRMWAREPATH)/logistic-regression
CMSIS_NN_PATH               := $(FIRMWAREPATH)/cmsis_nn/Source
WAKEWORD_PATH               := $(FIRMWAREPATH)/wakeWord-cmsisnn
IMAGE_CNN_PATH              := $(FIRMWAREPATH)/image-cnn
CMSIS_NN_INCLUDE_PATH       := $(FIRMWAREPATH)/cmsis_nn/Include 

# 1) collect all include‐dirs
FAST_GRNN_INC_DIRS           := $(shell find $(FAST_GRNN_PATH)          -type d)
LOGISTIC_REGRESSION_INC_DIRS := $(shell find $(LOGISTIC_REGRESSION_PATH) -type d)
CMSIS_NN_INC_DIRS            := $(shell find $(CMSIS_NN_PATH)            -type d)
CMSIS_NN_INC_DIRS            := \ $(shell find $(CMSIS_NN_PATH)          -type d) \
                                $(CMSIS_NN_INCLUDE_PATH)          
WAKEWORD_INC_DIRS            := $(shell find $(WAKEWORD_PATH)            -type d) 
IMAGE_CNN_INC_DIRS           := $(shell find $(IMAGE_CNN_PATH)           -type d) 

# export into CFLAGS so #includes work
CFLAGS += \
  $(patsubst %,-I%,$(FAST_GRNN_INC_DIRS)) \
  $(patsubst %,-I%,$(LOGISTIC_REGRESSION_INC_DIRS)) \
  $(patsubst %,-I%,$(CMSIS_NN_INC_DIRS)) \
  $(patsubst %,-I%,$(WAKEWORD_INC_DIRS)) \
  $(patsubst %,-I%,$(IMAGE_CNN_INC_DIRS)) 

# 2) find all .c sources (skip Jupyter checkpoints)
ALL_FAST_GRNN_SRCS := \
  $(shell find $(FAST_GRNN_PATH)           -type f -name '*.c' ! -path '*/.ipynb_checkpoints/*')
ALL_LOGISTIC_SRCS := \
  $(shell find $(LOGISTIC_REGRESSION_PATH) -type f -name '*.c' ! -path '*/.ipynb_checkpoints/*')
ALL_CMSIS_NN_SRCS := \
  $(shell find $(CMSIS_NN_PATH)            -type f -name '*.c' ! -path '*/.ipynb_checkpoints/*')
ALL_WAKEWORD_SRCS := \
  $(shell find $(WAKEWORD_PATH)            -type f -name '*.c' ! -path '*/.ipynb_checkpoints/*')      
ALL_IMAGE_CNN_SRCS := \
  $(shell find $(IMAGE_CNN_PATH)           -type f -name '*.c' ! -path '*/.ipynb_checkpoints/*')  

FAST_GRNN_SRCS    := $(ALL_FAST_GRNN_SRCS)
LOGISTIC_SRCS     := $(ALL_LOGISTIC_SRCS)
CMSIS_NN_SRCS     := $(ALL_CMSIS_NN_SRCS)
WAKEWORD_SRCS     := $(ALL_WAKEWORD_SRCS)   
IMAGE_CNN_SRCS    := $(ALL_IMAGE_CNN_SRCS)  

$(info FastGRNN sources        : $(words $(FAST_GRNN_SRCS)))
$(info Logistic Regression src : $(words $(LOGISTIC_SRCS)))
$(info CMSIS-NN sources        : $(words $(CMSIS_NN_SRCS)))
$(info Wake-word sources       : $(words $(WAKEWORD_SRCS)))  
$(info Image CNN sources       : $(words $(IMAGE_CNN_SRCS)))  

# 3) map sources → objects
FAST_GRNN_OBJS    := $(FAST_GRNN_SRCS:.c=.o)
LOGISTIC_OBJS     := $(LOGISTIC_SRCS:.c=.o)
CMSIS_NN_OBJS     := $(CMSIS_NN_SRCS:.c=.o)
WAKEWORD_OBJS     := $(WAKEWORD_SRCS:.c=.o)  
IMAGE_CNN_OBJS    := $(IMAGE_CNN_SRCS:.c=.o)  

# 4) per-module pattern rules
$(FAST_GRNN_PATH)/%.o: $(FAST_GRNN_PATH)/%.c
	@echo Building FastGRNN module: $@
	$(CC) $(ALL_CFLAGS) -c $< -o $@

$(LOGISTIC_REGRESSION_PATH)/%.o: $(LOGISTIC_REGRESSION_PATH)/%.c
	@echo Building Logistic Regression module: $@
	$(CC) $(ALL_CFLAGS) -c $< -o $@

$(CMSIS_NN_PATH)/%.o: $(CMSIS_NN_PATH)/%.c
	@echo Building CMSIS-NN module: $@
	$(CC) $(ALL_CFLAGS) -c $< -o $@

$(WAKEWORD_PATH)/%.o: $(WAKEWORD_PATH)/%.c     
	@echo Building Wake-word module: $@
	$(CC) $(ALL_CFLAGS) -c $< -o $@

$(IMAGE_CNN_PATH)/%.o: $(IMAGE_CNN_PATH)/%.c    
	@echo Building Wake-word module: $@
	$(CC) $(ALL_CFLAGS) -c $< -o $@

# pull in math library for both
LDFLAGS += -lm

# Increase reserved stack size to 32 KB (0x8000)
LDFLAGS       += -Wl,--defsym=__stack_size__=0x7000

# 5) final ELF depends on everything
$(TARGET)-$(PLATFORM).elf: \
    $(FAST_GRNN_OBJS) \
    $(LOGISTIC_OBJS) \
    $(CMSIS_NN_OBJS) \
    $(WAKEWORD_OBJS) \
    $(IMAGE_CNN_OBJS)
###############################################



include $(FIRMWAREPATH)/Makefile.inc


# Make “all” the default goal
.DEFAULT_GOAL := build
