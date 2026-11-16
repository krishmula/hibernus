.PHONY: build venv check e2e

build:
	go build -o bin/hib ./cmd/hib

venv:
	uv venv -q --allow-existing .venv
	uv pip install -q --python .venv -e python

check: build venv
	go vet ./...
	go test -race ./...
	@./bin/hib >/dev/null 2>&1; code=$$?; \
	if [ $$code -ne 2 ]; then echo "hib with no command exited $$code, want 2"; exit 1; fi
	.venv/bin/python -c "import hibernus; assert hibernus.connect().start() == 0"
	@out=$$(.venv/bin/python examples/toy/train.py --steps 3 --step-seconds 0); \
	if [ "$$out" != "$$(printf 'step 1\nstep 2\nstep 3')" ]; then echo "toy loop printed:"; echo "$$out"; exit 1; fi
	@echo "check passed"

e2e: build venv
	./scripts/e2e.sh
