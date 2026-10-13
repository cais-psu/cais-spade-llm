#!/usr/bin/env python3
"""Start recovery-scene controllers while reconciling uncertain service replies."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any


def ensure_controllers_active(
    names: list[str],
    read_states: Callable[[], dict[str, str]],
    change_state: Callable[[str, list[str]], Any],
    *,
    attempts: int = 3,
) -> None:
    """Load, configure and activate named controllers, checking observed state.

    A timed-out mutation may already have succeeded. Reading state before the
    next mutation prevents duplicate loads or configuring an active controller.
    No controller outside ``names`` is stopped, unloaded or reconfigured.
    """
    if not names or len(set(names)) != len(names) or attempts < 1:
        raise ValueError('Expected distinct controller names and a positive attempt limit.')

    def states() -> dict[str, str]:
        result = read_states()
        for name in names:
            state = result.get(name)
            if state is not None and state not in {'unconfigured', 'inactive', 'active'}:
                raise RuntimeError(f'{name}: unsupported observed controller state {state!r}')
        return result

    def mutate(operation: str, selected: list[str]) -> None:
        try:
            change_state(operation, selected)
        except TimeoutError:
            # The manager may have applied the request despite a lost reply.
            pass

    for name in names:
        observed = states()
        for _ in range(attempts):
            if name in observed:
                break
            mutate('load_controller', [name])
            observed = states()
        if name not in observed:
            raise RuntimeError(f'{name}: loading was not confirmed after {attempts} attempts')
        for _ in range(attempts):
            if observed[name] in {'inactive', 'active'}:
                break
            mutate('configure_controller', [name])
            observed = states()
            if name not in observed:
                raise RuntimeError(f'{name}: disappeared while configuring')
        if observed[name] not in {'inactive', 'active'}:
            raise RuntimeError(f'{name}: configuration was not confirmed after {attempts} attempts')

    observed = states()
    for _ in range(attempts):
        pending = [name for name in names if observed.get(name) != 'active']
        if not pending:
            return
        if any(observed.get(name) != 'inactive' for name in pending):
            raise RuntimeError(f'Controller states changed before activation: {observed!r}')
        mutate('switch_controller', pending)
        observed = states()
    pending = [name for name in names if observed.get(name) != 'active']
    if pending:
        raise RuntimeError(f'Activation was not confirmed for {pending!r}')


def main(args: list[str] | None = None) -> int:
    """Run the bounded startup procedure for one controller manager."""
    import rclpy
    from controller_manager_msgs.srv import (
        ConfigureController, ListControllers, LoadController, SwitchController,
    )
    from rclpy.duration import Duration
    from rclpy.utilities import remove_ros_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controllers', nargs='+')
    parser.add_argument('--controller-manager', required=True)
    parser.add_argument('--controller-manager-timeout', type=float, default=60.0)
    parser.add_argument('--service-call-timeout', type=float, default=60.0)
    parser.add_argument('--switch-timeout', type=float, default=60.0)
    parser.add_argument('--activate-as-group', action='store_true')
    cli = parser.parse_args(remove_ros_args(args=args or sys.argv)[1:])
    if min(cli.controller_manager_timeout, cli.service_call_timeout, cli.switch_timeout) <= 0:
        parser.error('Timeouts must be positive.')
    rclpy.init(args=args)
    node = rclpy.create_node('recovery_controller_spawner')
    clients: dict[str, Any] = {}

    def call(operation: str, service_type: Any, request: Any) -> Any:
        service = f'{cli.controller_manager}/{operation}'
        client = clients.get(operation)
        if client is None:
            client = node.create_client(service_type, service)
            clients[operation] = client
        if not client.wait_for_service(timeout_sec=cli.controller_manager_timeout):
            raise RuntimeError(f'Service unavailable: {service}')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=cli.service_call_timeout)
        if not future.done():
            client.remove_pending_request(future)
            node.get_logger().warning(f'{service}: reply timed out; checking controller states')
            raise TimeoutError(service)
        result = future.result()
        if result is None:
            raise RuntimeError(f'{service}: no response')
        return result

    def read_states() -> dict[str, str]:
        for attempt in range(3):
            try:
                response = call('list_controllers', ListControllers, ListControllers.Request())
                return {controller.name: controller.state for controller in response.controller}
            except TimeoutError:
                if attempt == 2:
                    raise
        raise RuntimeError('Controller observation did not complete.')

    def change_state(operation: str, selected: list[str]) -> None:
        if operation == 'switch_controller':
            request = SwitchController.Request()
            request.activate_controllers = selected
            request.deactivate_controllers = []
            request.strictness = SwitchController.Request.STRICT
            request.activate_asap = False
            request.timeout = Duration(seconds=cli.switch_timeout).to_msg()
            service_type = SwitchController
        else:
            service_type = LoadController if operation == 'load_controller' else ConfigureController
            request = service_type.Request()
            request.name = selected[0]
        response = call(operation, service_type, request)
        if not response.ok:
            node.get_logger().warning(f'{operation} returned false for {selected!r}; checking states')

    try:
        ensure_controllers_active(cli.controllers, read_states, change_state)
        node.get_logger().info(f'Observed active controllers: {cli.controllers!r}')
        return 0
    except (RuntimeError, TimeoutError, ValueError) as error:
        node.get_logger().error(str(error))
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
