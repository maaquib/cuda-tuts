APPS=matrix_sums

all: ${APPS}

%: %.cu
	nvcc -o $@ $<

profile: matrix_sums
	ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum matrix_sums

format:
	clang-format -i -style=Google *.cu

clean:
	rm -rf ${APPS}
