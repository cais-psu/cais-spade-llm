#!/usr/bin/env python3
from __future__ import annotations

"""Reconcile Nav2 lifecycle replies while retaining its existing manager and bonds."""

import threading
import time
from collections.abc import Callable


def reconcile_transition(transition: int, read_state: Callable[[], int],
                         change_state: Callable[[], bool]) -> bool:
    """Forward one lifecycle transition and require its observed terminal state.

    A lost reply may follow a completed transition. Never resend that mutation;
    the observed lifecycle state decides whether the request succeeded.
    """
    transitions = {1: (1, 2), 2: (2, 1), 3: (2, 3), 4: (3, 2),
                   5: (1, 4), 6: (2, 4), 7: (3, 4)}
    if transition not in transitions:
        return False
    source, target = transitions[transition]
    observed = read_state()
    if observed == target:
        return True
    if observed != source:
        return False
    try:
        change_state()
    except TimeoutError:
        pass
    return read_state() == target


def main() -> None:
    """Expose bounded lifecycle services without dispatching robot commands."""
    import rclpy
    from lifecycle_msgs.srv import ChangeState, GetState
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

    rclpy.init()
    node = rclpy.create_node('recovery_nav2_lifecycle')
    node.declare_parameter('node_names', [''])
    names = node.get_parameter('node_names').value
    if not names or any(not isinstance(name, str) or not name or '/' in name for name in names) or len(set(names)) != len(names):
        raise ValueError('Expected distinct local lifecycle node names')
    group = ReentrantCallbackGroup()
    clients, locks, services = {}, {}, []
    for name in names:
        clients[name] = (
            node.create_client(GetState, name + '/get_state', callback_group=group),
            node.create_client(ChangeState, name + '/change_state_unreconciled', callback_group=group),
        )
        locks[name] = threading.Lock()

    def call(client, request, timeout: float):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not client.service_is_ready():
            if time.monotonic() >= deadline:
                raise TimeoutError('Lifecycle service discovery timed out')
            time.sleep(.02)
        if not rclpy.ok():
            raise InterruptedError('Lifecycle proxy stopped')
        future = client.call_async(request)
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(.02)
        if not future.done():
            client.remove_pending_request(future)
            future.cancel()
            if not rclpy.ok():
                raise InterruptedError('Lifecycle proxy stopped')
            raise TimeoutError('Lifecycle service reply timed out')
        result = future.result()
        if result is None:
            raise RuntimeError('Lifecycle service returned no response')
        return result

    def callback(name):
        def handle(request, response):
            # One in-flight transition per node, including a repeated manager
            # request after an uncertain reply. Restart also begins by reading.
            with locks[name]:
                read_client, change_client = clients[name]

                def read_state():
                    deadline = time.monotonic() + 15.
                    while True:
                        try:
                            state = call(read_client, GetState.Request(), 5.).current_state.id
                        except TimeoutError:
                            if time.monotonic() >= deadline:
                                raise
                            continue
                        if state < 10 or time.monotonic() >= deadline:
                            return state
                        time.sleep(.05)

                try:
                    response.success = reconcile_transition(request.transition.id, read_state,
                        lambda: call(change_client, request, 10.).success)
                except (TimeoutError, RuntimeError, InterruptedError) as exc:
                    node.get_logger().error(f'{name} lifecycle state is unconfirmed: {exc}')
                    response.success = False
                if not response.success:
                    node.get_logger().error(f'{name} transition {request.transition.id} was not confirmed')
                return response
        return handle

    for name in names:
        services.append(node.create_service(ChangeState, name + '/change_state', callback(name), callback_group=group))
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()
        executor.shutdown(timeout_sec=2.)
        node.destroy_node()


if __name__ == '__main__':
    main()
