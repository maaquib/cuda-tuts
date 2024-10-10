APPS=hello vector_add matrix_mul

all: ${APPS}

%: %.cu
	nvcc -o $@ $<

format:
	clang-format -i -style=Google *.cu

clean:
	rm -rf ${APPS}
