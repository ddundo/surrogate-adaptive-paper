RESOLUTIONS := 64  # 64 for global, 128 for rest
SUBINTERVALS := 1
ADAPTATION_METHODS := uniform
TARGET_COMPLEXITIES := 0
EXPORT := True
EXPORT_FREQUENCY := 2
FLAMEGRAPH := False

output_dir := outputs

run:
	@$(foreach resolution, $(RESOLUTIONS), \
		$(foreach method, $(ADAPTATION_METHODS), \
		$(foreach complexity, $(TARGET_COMPLEXITIES), \
		$(foreach subinterval, $(SUBINTERVALS), \
		firedrake-clean && \
		$(eval experiment_dir := $(output_dir)/$(resolution)_$(subinterval)_$(method)_$(complexity)) \
		$(if $(or $(filter True,$(EXPORT)),$(filter True,$(FLAMEGRAPH))), mkdir -p $(experiment_dir);) \
		echo "Running experiment $(experiment_dir)" && \
		python run.py --initial_resolution $(resolution) --num_subintervals $(subinterval) --adaptation_method $(method) --target_complexity $(complexity) \
		$(if $(filter True,$(EXPORT)),--export_dir $(experiment_dir)) \
		--export_frequency $(EXPORT_FREQUENCY) \
		$(if $(filter True,$(FLAMEGRAPH)),-log_view :$(experiment_dir)/flamegraph.txt:ascii_flamegraph); \
		))))

flamegraph:
	@for dir in $(output_dir)/*/; do \
		for txtfile in $$dir/flamegraph.txt; do \
			svgfile=$$(basename $$txtfile .txt).svg; \
			echo "Generating flamegraph $$dir$$svgfile from $$txtfile"; \
			./flamegraph.pl --countname us $$txtfile > $$dir/$$svgfile; \
		done; \
	done

clean:
	rm -rf $(output_dir)
