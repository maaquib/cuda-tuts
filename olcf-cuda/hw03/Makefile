APPS=vector_add

all: ${APPS}

%: %.cu
	nvcc -o $@ $<

profile: vector_add
	nsys profile --stats=true vector_add

format:
	clang-format -i -style=Google *.cu

clean:
	rm -rf ${APPS} report*.nsys-rep report*.sqlite
