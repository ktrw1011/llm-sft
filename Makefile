SHELL := /bin/bash
.PHONY: flash-attn

flash-attn:
	echo "install flash-attn2"
	source .venv/bin/activate && pip install flash-attn --no-build-isolation