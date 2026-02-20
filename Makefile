PYTHON ?= python3

export PYTHONPATH := src

.PHONY: build train infer-py infer-cpp-lgbm infer-cpp-lr export demo run-all viz test build-cpp docker-build docker-demo log-backfill clean

build: build-cpp

build-cpp:
	$(PYTHON) -m inference_bench.inference.cpp_lgbm --build
	$(PYTHON) -m inference_bench.inference.cpp_lr --build

train:
	$(PYTHON) -m inference_bench.pipeline.run_trainer

infer-py:
	$(PYTHON) -m inference_bench.pipeline.run_infer_py --threads 1

infer-cpp-lgbm:
	$(PYTHON) -m inference_bench.pipeline.run_infer_cpp --threads 1

infer-cpp-lr:
	$(PYTHON) -m inference_bench.pipeline.run_infer_cpp_lr --threads 1

export:
	$(PYTHON) -m inference_bench.viz.export_static

demo: run-all

run-all:
	$(PYTHON) -m inference_bench.pipeline.run_all

viz:
	streamlit run src/inference_bench/viz/app.py

test:
	$(PYTHON) -m inference_bench.ops.run_tests --pytest-args -q

log-backfill:
	$(PYTHON) -c "from inference_bench.ops.run_logger import backfill_known_history; backfill_known_history()"

docker-build:
	docker compose build

docker-demo:
	docker compose run --rm demo

clean:
	rm -rf build artifacts/lib reports/site
