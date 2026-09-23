"""KMR-owned callable primitives for nominal tasks and validated recovery macros."""

from __future__ import annotations

from copy import deepcopy

from cais_spade_llm.recovery_framework.gazebo_worker import GazeboExecutionError
from cais_spade_llm.recovery_framework.kmr_tasks import capability_decompositions
from cais_spade_llm.resources.resource_profile import ResourceProfile, register_resource_profile


class KMRPrimitives:
    """Route primitive calls to the same owned ROS worker as KMR tasks."""

    def __init__(self, agent) -> None:
        self.agent = agent

    async def _call(self, primitive: str, parameters: dict) -> dict:
        agent = self.agent
        request = deepcopy(getattr(agent, '_kmr_execution_request', None))
        if request is None:
            return {'success': False, 'message': 'No observed KMR task context for recovery'}
        runtime = getattr(agent, 'environment_runtime', None) or getattr(agent, 'delivery_runtime', None)
        if runtime is not None and runtime.stopped:
            return {'success': False, 'message': 'KMR execution is stopped'}
        request.update(mode='primitive', primitive=primitive, primitive_parameters=parameters)
        try:
            result = await agent.worker.run(request)
        except GazeboExecutionError as exc:
            agent.record_primitive_evidence(exc.result)
            return {'success': False, 'message': str(exc), 'observations': deepcopy(exc.result)}
        agent.record_primitive_evidence(result)
        output = result.get('result')
        return {**(output if isinstance(output, dict) else {'value': output}),
                'success': True, 'observations': deepcopy(result)}

    async def compute_pick_targets(self, initial: list[float] | None = None) -> dict:
        """Compute downward pickup targets from a fresh observed Storage part.

        ---
        description: Compute downward pickup targets from a fresh observed Storage part.
        preconditions:
          held_part:
            equals: null
        effects: {}
        ---
        """
        return await self._call('compute_pick_targets', {'initial': initial})

    async def compute_place_targets(self, transform: list[float]) -> dict:
        """Compute placement and arm clearance targets from observed custody.

        ---
        description: Compute placement and arm clearance targets from observed custody.
        preconditions:
          held_part:
            exists: true
        effects: {}
        ---
        """
        return await self._call('compute_place_targets', {'transform': transform})

    async def move_cartesian(self, target: list[float]) -> dict:
        """Move along a complete collision-checked path preserving TCP orientation.

        ---
        description: Move along a complete collision-checked path preserving TCP orientation.
        preconditions: {}
        effects:
          current_pose:
            set_from_param: target
        ---
        """
        return await self._call('move_cartesian', {'target': target})

    async def move_to_pose(self, target: list[float], seed: list[float] | None = None) -> dict:
        """Enter a Cartesian segment using a collision-checked posture.

        ---
        description: Enter a Cartesian segment using a collision-checked posture.
        preconditions: {}
        effects:
          current_pose:
            set_from_param: target
        ---
        """
        return await self._call('move_to_pose', {'target': target, 'seed': seed})

    async def move_to_configuration(self, joints: list[float], hold_arm_base: bool = False) -> dict:
        """Move to bounded joint positions with collision checks.

        ---
        description: Move to bounded joint positions with collision checks.
        preconditions: {}
        effects: {}
        ---
        """
        return await self._call('move_to_configuration', {'joints': joints, 'hold_arm_base': hold_arm_base})

    async def move_home(self) -> dict:
        """Return the empty arm to the configured downward posture at Storage.

        ---
        description: Return the empty KMR arm to its downward-facing Storage home.
        preconditions:
          held_part:
            equals: null
          resource_location:
            equals: Storage
        effects:
          current_pose_ref:
            set: home
        ---
        """
        return await self._call('move_home', {})

    async def rotate_arm_base(self, joint_a1: float) -> dict:
        """Turn joint_a1 while holding the other joints after checking the complete sweep.

        ---
        description: Turn joint_a1 while holding the other joints after checking the complete
          sweep.
        preconditions: {}
        effects: {}
        ---
        """
        return await self._call('rotate_arm_base', {'joint_a1': joint_a1})

    async def open_gripper(self) -> dict:
        """Open the gripper at a supported release pose or while empty.

        ---
        description: Open the gripper at a supported release pose or while empty.
        preconditions: {}
        effects:
          gripper_state:
            set: open
        ---
        """
        return await self._call('open_gripper', {})

    async def close_gripper(self) -> dict:
        """Close the gripper at the observed pickup pose.

        ---
        description: Close the gripper at the observed pickup pose.
        preconditions: {}
        effects:
          gripper_state:
            set: closed
        ---
        """
        return await self._call('close_gripper', {})

    async def observe_grasp(self, initial: list[float]) -> dict:
        """Observe the part to TCP transform without establishing custody.

        ---
        description: Observe the part to TCP transform without establishing custody.
        preconditions:
          gripper_state:
            equals: closed
        effects: {}
        ---
        """
        return await self._call('observe_grasp', {'initial': initial})

    async def attach_part(self) -> dict:
        """Acknowledge attachment of the bound part to the closed gripper.

        ---
        description: Acknowledge attachment of the bound part to the closed gripper.
        preconditions:
          gripper_state:
            equals: closed
        effects: {}
        ---
        """
        return await self._call('attach_part', {})

    async def detach_part(self) -> dict:
        """Acknowledge release of the bound part at a supported destination.

        ---
        description: Acknowledge release of the bound part at a supported destination.
        preconditions:
          gripper_state:
            equals: open
        effects:
          held_part:
            set: null
        ---
        """
        return await self._call('detach_part', {})

    async def part_collision(self, attached: bool) -> dict:
        """Update the collision scene from observed part attachment.

        ---
        description: Update the collision scene from observed part attachment.
        preconditions: {}
        effects: {}
        ---
        """
        return await self._call('part_collision', {'attached': attached})

    async def custody(self, transform: list[float]) -> dict:
        """Check the held part against its acknowledged attachment transform.

        ---
        description: Check the held part against its acknowledged attachment transform.
        preconditions: {}
        effects: {}
        ---
        """
        return await self._call('custody', {'transform': transform})

    async def confirm_carrying(self) -> dict:
        """Confirm the arm remains at its acknowledged carrying posture.

        ---
        description: Confirm the arm remains at its acknowledged carrying posture.
        preconditions: {}
        effects: {}
        ---
        """
        return await self._call('confirm_carrying', {})

    async def observe_carrying(self, transform: list[float]) -> dict:
        """Observe custody and preserve the current arm posture for transport.

        ---
        description: Observe custody and preserve the current arm posture for transport.
        preconditions: {}
        effects: {}
        ---
        """
        return await self._call('observe_carrying', {'transform': transform})

    async def observe_custody(self, previous: dict) -> dict:
        """Validate previous custody against current Gazebo evidence.

        ---
        description: Validate previous custody against current Gazebo evidence.
        preconditions: {}
        effects: {}
        ---
        """
        return await self._call('observe_custody', {'previous': previous})

    async def validate_transport(self) -> dict:
        """Collision-check complete robot geometry and payload along the configured route.

        ---
        description: Collision-check complete robot geometry and payload along the configured
          route.
        preconditions: {}
        effects: {}
        ---
        """
        return await self._call('validate_transport', {})

    async def dock(self, target_resource: str, transform: list[float]) -> dict:
        """Execute the guarded docking controller after validated transport clearance.

        ---
        description: Execute the guarded docking controller after validated transport clearance.
        preconditions: {}
        effects:
          resource_location:
            set_from_param: target_resource
        ---
        """
        return await self._call('dock', {'target_resource': target_resource, 'transform': transform})

    async def observe_dock(self, transform: list[float]) -> dict:
        """Observe the dock and custody before acknowledging arrival.

        ---
        description: Observe the dock and custody before acknowledging arrival.
        preconditions: {}
        effects: {}
        ---
        """
        return await self._call('observe_dock', {'transform': transform})

    async def observe_release(self, destination: list[float]) -> dict:
        """Observe support, orientation, and arm clearance after release.

        ---
        description: Observe support, orientation, and arm clearance after release.
        preconditions: {}
        effects:
          held_part:
            set: null
          current_state:
            set: idle
        ---
        """
        return await self._call('observe_release', {'destination': destination})


KMR_RECOVERY_PRIMITIVES = (
    'compute_pick_targets',
    'compute_place_targets',
    'move_cartesian',
    'move_to_pose',
    'move_to_configuration',
    'move_home',
    'rotate_arm_base',
    'open_gripper',
    'close_gripper',
    'observe_grasp',
    'attach_part',
    'detach_part',
    'part_collision',
    'custody',
    'confirm_carrying',
    'observe_carrying',
    'observe_custody',
    'validate_transport',
    'dock',
    'observe_dock',
    'observe_release',
)

KMR_RESOURCE_PROFILE = ResourceProfile(
    resource_type='kmr',
    snapshot_fields=('current_state', 'resource_location', 'held_part', 'gripper_state', 'current_pose'),
    primitive_owner_resolver=lambda agent: agent.kmr_primitives,
    capability_decomposition_provider=capability_decompositions,
    carried_entity_field='held_part',
    carried_entity_location_builder=lambda resource_jid, _snapshot: resource_jid,
    expected_end_state_projection_map={
        'held_part': 'held_part', 'resource_state': 'current_state',
        'resource_location': 'resource_location',
    },
    primitive_kind_map={
        name: ('observation' if name.startswith(('observe_', 'compute_', 'validate_', 'confirm_')) else 'motion')
        for name in KMR_RECOVERY_PRIMITIVES
    },
    observation_output_schema_map={
        'compute_pick_targets': {name: {'type': 'array', 'items': {'type': 'number'}}
                                 for name in ('target', 'approach', 'lift', 'seed')},
        'compute_place_targets': {name: {'type': 'array', 'items': {'type': 'number'}}
                                  for name in ('target', 'approach', 'retreat', 'destination', 'before_turn')},
        'move_cartesian': {'tcp_pose': {'type': 'array', 'items': {'type': 'number'}},
                           'avoid_collisions': 'boolean', 'fraction': 'number'},
    },
)
register_resource_profile(KMR_RESOURCE_PROFILE)
