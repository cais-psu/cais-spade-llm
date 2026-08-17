SHELL := /usr/bin/env bash

.PHONY: install run run-physical-ur5e headless setup-perception-host bootstrap-gazebo check
.PHONY: cleanup-report cleanup cleanup-global-ros

install:
	poetry install
	poetry run pip install -r requirements-ui.txt

run:
	poetry run python -m cais_spade_llm

run-physical-ur5e:
	source /opt/ros/humble/setup.bash && \
		source "$${HOME}/ros2_ws/install/setup.bash" && \
		ROS_DOMAIN_ID=42 poetry run python -m cais_spade_llm

headless:
	poetry run python -m cais_spade_llm.ui_main --headless

setup-perception-host:
	sudo bash scripts/setup_perception_host.sh

bootstrap-gazebo:
	bash scripts/bootstrap_gazebo_workspace.sh

check:
	poetry check
	poetry run python -m compileall -q cais_spade_llm ros2
	poetry run python -m cais_spade_llm.ui_main --help

cleanup-report:
	poetry run python -m cais_spade_llm.utils.runtime_cleanup --scope all

cleanup:
	poetry run python -m cais_spade_llm.utils.runtime_cleanup --scope cais --apply

cleanup-global-ros:
	poetry run python -m cais_spade_llm.utils.runtime_cleanup --scope global-ros --apply
