.PHONY: build check

build:
	go build -o bin/hib ./cmd/hib

check: build
	go vet ./...
	go test -race ./...
	@./bin/hib >/dev/null 2>&1; code=$$?; \
	if [ $$code -ne 2 ]; then echo "hib with no command exited $$code, want 2"; exit 1; fi
	uv venv -q --allow-existing .venv
	uv pip install -q --python .venv -e python
	.venv/bin/python -c "import hibernus; assert hibernus.connect().start() == 0"
	@out=$$(.venv/bin/python examples/toy/train.py --steps 3 --step-seconds 0); \
	if [ "$$out" != "$$(printf 'step 1\nstep 2\nstep 3')" ]; then echo "toy loop printed:"; echo "$$out"; exit 1; fi
	@echo "check passed"
