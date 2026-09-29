"""KMR-owned callable primitives for nominal tasks and validated recovery macros."""

from __future__ import annotations

from copy import deepcopy

from cais_spade_llm.recovery_framework.gazebo_worker import GazeboExecutionError
from cais_spade_llm.recovery_framework.kmr_tasks import capability_decompositions
from cais_spade_llm.resources.resource_profile import ResourceProfile, register_resource_profile
from cais_spade_llm.resources.gazebo_programs import (
    current_program_revision, primitive_available, resource_program_revision,
)
from cais_spade_llm.recovery_framework import ROOT


class KMRPrimitives:
    """Route primitive calls to the same owned ROS worker as KMR tasks."""

    def __init__(self, agent) -> None:
        self.agent = agent

    async def _call(self, primitive: str, parameters: dict) -> dict:
        agent = self.agent
        runtime = getattr(agent, 'environment_runtime', None) or getattr(agent, 'delivery_runtime', None)
        if runtime is not None:
            previous = getattr(agent, 'workflow_custody', None)
            if previous is None:
                for transition in reversed(runtime.context.transitions):
                    acknowledgement = transition['acknowledgement']
                    if acknowledgement.get('resource_id') == 'KMR':
                        previous = transition.get('observations') or acknowledgement.get('observations')
                        break
            valuation = runtime.context.snapshot()
            for field in ('held_part', 'resource_location'):
                if field in getattr(agent, '_primitive_state', {}):
                    valuation['KMR'][field] = deepcopy(agent._primitive_state[field])
            request = {
                'mode': 'primitive', 'inputs': runtime.context.inputs,
                'valuation': valuation,
                'geometry': deepcopy(getattr(runtime.context, 'geometry', {})),
                'probe': getattr(runtime, 'kmr_probe', None) or runtime.prepared.get('probe'),
                'custody': deepcopy(previous),
            }
        else:
            request = deepcopy(getattr(agent, '_kmr_execution_request', None))
            if request is None:
                return {'success': False, 'message': 'No active KMR runtime context for recovery'}
        scene = request.get('inputs', {}).get('scene', {})
        if 'resource_programs' in scene:
            if not primitive_available(scene, 'KMR', primitive, recovery=True):
                return {'success': False, 'message': f'KMR primitive is unavailable for recovery: {primitive}'}
            scene_file = getattr(runtime, 'scene_file', '')
            if scene_file and current_program_revision(ROOT / scene_file) != resource_program_revision(scene):
                return {'success': False, 'message': 'Stale Gazebo program revision'}
        if runtime is not None and runtime.stopped:
            return {'success': False, 'message': 'KMR execution is stopped'}
        request.update(mode='primitive', primitive=primitive, primitive_parameters=parameters)
        agent._kmr_execution_request = deepcopy(request)
        try:
            result = await agent.worker.run(request)
        except GazeboExecutionError as exc:
            agent.record_primitive_evidence(exc.result)
            return {'success': False, 'message': str(exc), 'observations': deepcopy(exc.result)}
        agent.record_primitive_evidence(result)
        output = result.get('result')
        return {**(output if isinstance(output, dict) else {'value': output}),
                'success': True, 'observations': deepcopy(result)}

    async def detect_parts(self) -> dict:
        """Read fresh Gazebo poses for configured Storage parts."""
        return await self._call('detect_parts', {})

    async def compute_pick_targets(self, initial: list[float] | None = None) -> dict:
        """Compute KMR pickup poses from the observed Storage part."""
        return await self._call('compute_pick_targets', {'initial': initial})

    async def compute_place_targets(self, transform: list[float]) -> dict:
        """Compute placement targets from the observed grasp transform."""
        return await self._call('compute_place_targets', {'transform': transform})

    async def move_to_named_pose(self, pose_name: str) -> dict:
        """Move the arm to a configured home or transport posture."""
        return await self._call('move_to_named_pose', {'pose_name': pose_name})

    async def move_cartesian(self, target: list[float],
                             waypoints: list[list[float]] | None = None,
                             seed: list[float] | None = None) -> dict:
        """Follow a checked Cartesian TCP path from measured feedback."""
        return await self._call('move_cartesian',
                                {'target': target, 'waypoints': waypoints, 'seed': seed})

    async def move_relative(self, offset: list[float]) -> dict:
        """Move the TCP by a world-frame XYZ offset."""
        return await self._call('move_relative', {'offset': offset})

    async def move_joints(self, joints: list[float]) -> dict:
        """Move all seven KMR arm joints to absolute radian targets."""
        return await self._call('move_joints', {'joints': joints})

    async def rotate_joint(self, joint_name: str, delta_deg: float) -> dict:
        """Rotate one selected KMR arm joint by a relative angle."""
        return await self._call('rotate_joint', {'joint_name': joint_name,
                                                 'delta_deg': delta_deg})

    async def open_gripper(self) -> dict:
        """Open the KMR gripper and confirm the observed width."""
        return await self._call('open_gripper', {})

    async def close_gripper(self) -> dict:
        """Close the KMR gripper and confirm the observed width."""
        return await self._call('close_gripper', {})

    async def grasp_part(self, part_name: str | None = None,
                         initial: list[float] | None = None) -> dict:
        """Close, attach, and observe the targeted part."""
        return await self._call('grasp_part', {'part_name': part_name, 'initial': initial})

    async def release_part(self, transform: list[float] | None = None) -> dict:
        """Open, detach, and update the part collision geometry."""
        return await self._call('release_part', {'transform': transform})

    async def attach_part(self, part_name: str | None = None) -> dict:
        """Attach the observed part to the closed KMR gripper."""
        return await self._call('attach_part', {'part_name': part_name})

    async def detach_part(self) -> dict:
        """Detach the observed part from the open KMR gripper."""
        return await self._call('detach_part', {})

    async def move_base(self, target_pose: list[float],
                        waypoints: list[list[float]] | None = None,
                        transform: list[float] | None = None) -> dict:
        """Plan and follow a checked base route to an exact planar pose."""
        return await self._call('move_base', {'target_pose': target_pose,
                                              'waypoints': waypoints,
                                              'transform': transform})


KMR_RECOVERY_PRIMITIVES = (
    'detect_parts', 'compute_pick_targets', 'compute_place_targets',
    'move_to_named_pose', 'move_cartesian', 'move_relative',
    'move_joints', 'rotate_joint', 'open_gripper', 'close_gripper',
    'grasp_part', 'release_part', 'attach_part', 'detach_part', 'move_base',
)


KMR_RESOURCE_PROFILE = ResourceProfile(
    resource_type='kmr',
    snapshot_fields=('current_state', 'resource_location', 'held_part', 'gripper_state', 'current_pose', 'base_pose'),
    primitive_owner_resolver=lambda agent: agent.kmr_primitives,
    capability_decomposition_provider=capability_decompositions,
    carried_entity_field='held_part',
    carried_entity_location_builder=lambda resource_jid, _snapshot: resource_jid,
    expected_end_state_projection_map={
        'held_part': 'held_part', 'resource_state': 'current_state',
        'resource_location': 'resource_location',
    },
    primitive_kind_map={
        name: ('observation' if name.startswith(('detect_', 'compute_')) else 'motion')
        for name in KMR_RECOVERY_PRIMITIVES
    },
    observation_output_schema_map={
        'compute_pick_targets': {name: {'type': 'array', 'items': {'type': 'number'}}
                                 for name in ('target', 'approach', 'lift', 'seed')},
        'compute_place_targets': {name: {'type': 'array', 'items': {'type': 'number'}}
                                  for name in ('target', 'approach', 'retreat', 'destination', 'before_turn')},
        'move_base': {'base_pose': {'type': 'array', 'items': {'type': 'number'}}},
        'move_cartesian': {'tcp_pose': {'type': 'array', 'items': {'type': 'number'}},
                           'avoid_collisions': 'boolean', 'fraction': 'number'},
    },
)
register_resource_profile(KMR_RESOURCE_PROFILE)
