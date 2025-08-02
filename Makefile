# Makefile

.venv:
	uv venv .venv
	uv sync --upgrade

install: .venv
	mkdir -p .trash
