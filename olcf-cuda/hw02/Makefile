APPS=stencil_1d matrix_mul_shared

all: ${APPS}

%: %.cu
	nvcc -o $@ $<

format:
	clang-format -i -style=Google *.cu

clean:
	rm -rf ${APPS}
