"""Opt-in admission of committed hand trajectories, with continuous checking.

The controller knows only current hand positions and actions already admitted.
New actions from either agent must avoid the other's committed trajectory and
its stationary endpoint. Future human decisions and task scope are not queried.
This changes low-level control, not the task graph or robot policy action mask.
"""
import math

from loghrc.core import physics
from loghrc.core.provenance import fingerprint
from loghrc.core.proximity import encounters


def path_length(path):
    return sum(math.dist(a,b) for a,b in zip(path,path[1:]))


def paths_safe(path, duration, other_hand, other_action):
    """Exact piecewise-linear check, including both stationary tails."""
    horizon=max(duration,other_action.remaining if other_action is not None else 0)
    for tick in range(1,horizon+1):
        own={'path':path,'total':duration,'remaining':duration-tick} if tick<=duration else None
        other=None
        if other_action is not None and tick<=other_action.remaining:
            other={'path':other_action.path,'total':other_action.total,'remaining':other_action.remaining-tick}
        frame={'t':tick,'actions':[own,other], 'hands':[path[-1],
            other_action.path[-1] if other_action is not None and tick>other_action.remaining else other_hand]}
        if encounters(frame):return False
    return True


class CoordinatedMotion:
    """Mixin for the LogHrcEnv executor; independent of training and furniture."""
    def _build_experiment_manifest(self):
        result=super()._build_experiment_manifest()
        result['motion_coordination']={'version':'committed-path-v1','threshold':1.5,
            'scope':'both agents, committed motion plus stationary endpoints; no future choices',
            'candidates':'original, XY/YX, four perimeter lanes at travel height',
            'duration':'ceil(base duration * max(1, detour length / original length))',
            'failure':'reject request before effects; elapsed invalid/blocked time retained',
            'mask':'unchanged graph/physics mask; route admission is executor-only'}
        result['experiment_hash']=fingerprint({k:v for k,v in result.items()
            if k not in ('experiment_hash','runtime','source')})
        return result

    def reset(self, **kwargs):
        self.motion_rejections={0:0,1:0}
        self.motion_detours={0:0,1:0}
        self.motion_extra_ticks={0:0,1:0}
        self._admitted=None
        return super().reset(**kwargs)

    def _destination(self,agent,verb,arg):
        if verb=='wait':return self.agents[agent].hand,'wait'
        if verb=='move_home':
            return self._f(self.config.home_robot if agent==0 else self.config.home_human),'move_home'
        if verb=='pick':
            p=self.config.part(arg);state=self.parts[arg]
            return (None,'pick') if state.pose is None else (self._top_face(state.pose,p.size),'pick')
        if verb in ('place_target','place_staging'):
            p=self.config.part(arg)
            return self._top_face(self.target_of(arg) if verb=='place_target' else self.start_of(arg),p.size),'place'
        if verb=='attach':
            j=self.config.joint(arg)
            if any(self.role_part_id(p) is None for p in j.parts):return None,'attach'
            return physics.attach_destination(self,j),'attach'
        return None,None

    def _plan(self,agent,destination,base_duration):
        start=self.agents[agent].hand
        original=self._route(start,destination)
        z=float(self.config.grid_size[2]-1)
        a=(start[0],start[1],z);b=(destination[0],destination[1],z)
        candidates=[original]
        # Waiting means staying still, never a silent movement substituted for noop.
        if start!=destination:
            mids=[[(a[0],b[1],z)],[(b[0],a[1],z)]]
            mids += [[(x,a[1],z),(x,b[1],z)] for x in (0.,float(self.config.grid_size[0]-1))]
            mids += [[(a[0],y,z),(b[0],y,z)] for y in (0.,float(self.config.grid_size[1]-1))]
            candidates += [[start,a,*mid,b,destination] for mid in mids]
        original_length=path_length(original)
        plans=[];seen=set()
        for path in candidates:
            path=[p for i,p in enumerate(path) if i==0 or p!=path[i-1]]
            key=tuple(path)
            if key in seen:continue
            seen.add(key)
            length=path_length(path)
            duration=max(base_duration,math.ceil(base_duration*length/original_length)) if original_length>1e-9 else base_duration
            plans.append((duration,length,path))
        other=self.agents[1-agent]
        for duration,_,path in sorted(plans,key=lambda p:(p[0],p[1])):
            if paths_safe(path,duration,other.hand,other.action):return path,duration,original
        return None

    def _start_action(self,agent,verb,arg):
        # Preserve base validity checks/reasons before considering route admission.
        from loghrc.constraints.masking import physics_mask, procedural_mask
        try:index=self.action_table.index((verb,arg))
        except ValueError:return super()._start_action(agent,verb,arg)
        if not (physics_mask(self,agent)[index] and procedural_mask(self,agent)[index]):
            return super()._start_action(agent,verb,arg)
        destination,duration_key=self._destination(agent,verb,arg)
        if destination is None:return super()._start_action(agent,verb,arg)
        duration=self._duration(agent,duration_key)
        plan=self._plan(agent,destination,duration)
        if plan is None:
            self.motion_rejections[agent]+=1
            self._log_event({'type':'motion_rejected','agent':agent,'verb':verb,'arg':arg})
            return False,'no safe committed-motion route'
        self._admitted=(agent,plan,duration)
        try:return super()._start_action(agent,verb,arg)
        finally:self._admitted=None

    def _begin(self,agent,verb,arg,dur,to):
        admission=self._admitted
        if admission is None:
            # Inherited pause/hesitation and invalid-action handling are stationary.
            return super()._begin(agent,verb,arg,dur,to)
        _,(path,duration,original),base_duration=admission
        # Base _begin logs the action after applying zone bookkeeping. Supply the
        # admitted route locally; _route is otherwise unchanged and deterministic.
        self._planned_route=path
        try:super()._begin(agent,verb,arg,duration,to)
        finally:self._planned_route=None
        if path!=original:
            self.motion_detours[agent]+=1
            self.motion_extra_ticks[agent]+=duration-base_duration
            self._log_event({'type':'motion_detour','agent':agent,'extra_ticks':duration-base_duration})

    def _route(self,frm,to):
        planned=getattr(self,'_planned_route',None)
        return planned if planned is not None else super()._route(frm,to)

    def metrics_summary(self,success):
        result=super().metrics_summary(success)
        for agent,name in ((0,'robot'),(1,'human')):
            result.update({f'{name}_motion_rejections':self.motion_rejections[agent],
                f'{name}_motion_detours':self.motion_detours[agent],
                f'{name}_motion_extra_ticks':self.motion_extra_ticks[agent]})
        return result
