SHELL := /usr/bin/env bash

.PHONY: install run headless bootstrap-gazebo check

install:
	poetry install
	poetry run pip install -r requirements-ui.txt

run:
	poetry run python -m cais_spade_llm

headless:
	poetry run python -m cais_spade_llm.ui_main --headless

bootstrap-gazebo:
	bash scripts/bootstrap_gazebo_workspace.sh

check:
	poetry check
	poetry run python -m compileall -q cais_spade_llm ros2
	poetry run python -m cais_spade_llm.ui_main --help
