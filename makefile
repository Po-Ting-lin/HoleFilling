HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
        HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc
CXXFLAGS+=-std=c++11
CXXFLAGS+=`pkg-config --cflags opencv`
LIBS=`pkg-config --libs opencv`

SOURCES=$(wildcard HoleFillHip/*.cpp HoleFillHip/*.h)
DEST=holeFilling.out

all: ${DEST}

${DEST}: $(SOURCES)
	${HIPCC} $(SOURCES) --disable-warnings -o $@ ${LIBS} ${CXXFLAGS}

clean:
	rm -f *.out *.o
