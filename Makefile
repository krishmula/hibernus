.PHONY: build check

build:
	go build -o bin/hib ./cmd/hib

check: build
	go vet ./...
	@./bin/hib >/dev/null 2>&1; code=$$?; \
	if [ $$code -ne 2 ]; then echo "hib with no command exited $$code, want 2"; exit 1; fi
	uv venv -q --allow-existing .venv
	uv pip install -q --python .venv -e python
	.venv/bin/python -c "import hibernus; assert hibernus.connect().start() == 0"
	@echo "phase 0 check passed"
