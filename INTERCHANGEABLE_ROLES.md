# Interchangeable assembly roles, v1

`PartSpec` retains a physical inventory entry and a corresponding assembly-role
definition for schema compatibility. They are distinct namespaces: physical IDs
index `env.parts`; role IDs index targets, support, groups and joint endpoints.
`TaskConfig.interchangeable_groups` explicitly declares equal-class, equal-size
physical pieces that can fill each other's roles. Groups must be disjoint.
An empty tuple retains fixed-piece behavior.

`pick(id)` and `place_staging(id)` refer to a physical object. Staging remains
that object's original location. `place_target(role)` places the currently held
compatible object at that role's target. `attach(joint)` fastens the physical
objects occupying the joint's roles. Home and wait retain their behavior,
including moving home with a held object.

The graph and phase-1 allocation apply to roles. Physical inventory is shared:
either agent may collect a compatible piece for one of its outstanding roles,
but cannot complete another agent's allocated role. Scripted phase-2 scopes and
preferences also apply to assembly roles. A role cannot be simultaneously
reserved by two placement actions. No frontier heuristic or anti-loop rule was
introduced.

Voxel class channels remain unchanged. Target ghosts disappear when their role
is filled, regardless of which physical object fills it. This does not expose
hidden human profiles or solve all physical-ID/perception ambiguities.

Logs identify `assembly_semantics=interchangeable-roles-v1`. A successful
placement's event `arg` names the role; `physical_part` names the object and
`target_role` records occupancy. Footers include `role_occupants`. Python replay
and the browser viewer retain physical identities and role occupancy separately;
old events without these fields remain readable. Task/experiment hashes change.

`journal_furniture_config` retains fixed-piece behavior by default. Pass
`interchangeable=True` to enable equivalent furniture instances. In
loghrc-training, select `--assembly-semantics interchangeable` explicitly, so
historical configurations do not silently opt in.
New role-based experiments must use fresh runs and models; equal action-space
size does not make old checkpoints semantically equivalent.

Validation covers reversed assignment of every interchangeable group in both
furniture tasks and both phases, role-based supports/joints/precedence, allocation,
reservation, staging/home recovery, reset, replay and viewer parity.
