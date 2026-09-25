"""Observable task commitments, emitted by a scripted human when starting a group."""
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from loghrc.core.config import HUMAN, ROBOT
from loghrc.core.provenance import fingerprint
from loghrc.humans.scripted import HumanProfile


@dataclass(frozen=True)
class PreferenceProfile(HumanProfile):
    accepts_help: bool = True
    may_take_over_robot: bool = False


class PublicCommitments:
    """Mixin: publicly announced commitments, never a hidden-scope robot mask.

    Scope is read only by the human announcing its current group and by offline
    evaluation counters. Robot execution/masking reads the same published state
    supplied to the policy. No reward term depends on a profile or scope.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.observation_space=gym.spaces.Dict({**self.observation_space.spaces,
            'task_signals':gym.spaces.Box(0.,1.,(len(self.config.groups),3),np.float32)})

    def _build_experiment_manifest(self):
        result=super()._build_experiment_manifest()
        result['interaction_semantics']='journal-v4-observable-help-preferences'
        result['task_signals']={'version':'current-group-commitment-v1','columns':['announced','help_accepted','exclusive_active'],
            'emission':'first successful human action start in each group','release':'group complete',
            'scope':'current groups only; profile and future groups not exposed',
            'enforcement':'public exclusive commitments affect execution and mask equally',
            'reward':'unchanged; rejection consumes ordinary invalid-action time'}
        result['experiment_hash']=fingerprint({k:v for k,v in result.items() if k not in ('experiment_hash','runtime','source')})
        return result

    def reset(self,**kwargs):
        self.task_signals=np.zeros((len(self.config.groups),3),np.float32)
        self.preference_counts=dict(accepted_help=0,unwanted_help_attempts=0,
                                   pre_signal_interventions=0,complementary_operations=0)
        self._robot_operation_category=None
        return super().reset(**kwargs)

    def _group_index(self,verb,arg):
        for i,g in enumerate(self.config.groups):
            if arg in (g.joint_ids if verb=='attach' else g.part_ids):return i
        return None

    def _publicly_allowed(self,verb,arg):
        if verb in ('place_target','attach'):
            group=self._group_index(verb,arg)
            return group is None or not self.task_signals[group,2]
        if verb=='pick':
            useful=[role for role in self.compatible_roles(arg) if not self.role_filled(role)
                    and (self.graph is None or self.graph.role_ready(self,self.config.part(role),arg))]
            # Ordinary graph-invalid picks are not preference violations.
            return not useful or any(not self.task_signals[self._group_index('place_target',role),2] for role in useful)
        return True  # home, noop, and staging remain recovery actions

    def execution_mask(self):
        result=super().execution_mask()
        for i,physical in enumerate(self._binding):
            if result[i] and physical is not None:
                result[i]=self._publicly_allowed(*self.action_table[physical])
        return result

    def _start_action(self,agent,verb,arg):
        if agent==ROBOT and not self._publicly_allowed(verb,arg):
            self.preference_counts['unwanted_help_attempts']+=1
            self._log_event({'type':'commitment_rejection','agent':agent,'verb':verb,'arg':arg})
            return False,'blocked by publicly announced commitment'
        group=None
        if agent==HUMAN:
            if verb=='pick':
                # Identify the scripted human's current intention, not the
                # nominal identity of an interchangeable physical piece.
                for g in self.human_policy._my_groups(self):
                    if self.human_policy._next_group_step(self,g)==(verb,arg):
                        group=next(i for i,x in enumerate(self.config.groups) if x.name==g.name)
                        break
            elif verb in ('place_target','attach'):
                group=self._group_index(verb,arg)
        category=None
        if agent==ROBOT and verb in ('place_target','attach'):
            index=self._group_index(verb,arg)
            signal=self.task_signals[index]
            # Evaluation only: unannounced scope is never fed to the policy,
            # validity, reward or motion controller.
            scope=self.human_profile.group_scope
            in_scope=scope is None or self.config.groups[index].name in scope
            category=('accepted_help' if signal[0] and signal[1] else
                      'pre_signal_interventions' if in_scope else 'complementary_operations')
        ok,message=super()._start_action(agent,verb,arg)
        if ok and agent==ROBOT:self._robot_operation_category=category
        if ok and group is not None and not self.task_signals[group,0]:
            accept=self.human_profile.accepts_help
            self.task_signals[group]=[1.,float(accept),float(not accept)]
            self._log_event({'type':'task_commitment','group':self.config.groups[group].name,
                            'accepts_help':accept,'exclusive_active':not accept})
        return ok,message

    def _finish(self,agent,act):
        super()._finish(agent,act)
        if agent==ROBOT and act.verb in ('place_target','attach'):
            event=next(e for e in reversed(self.events) if e['type']=='action_end' and e['agent']==ROBOT)
            if event.get('ok') and self._robot_operation_category is not None:
                self.preference_counts[self._robot_operation_category]+=1
                self._log_event({'type':'robot_operation_classified','category':self._robot_operation_category,
                                 'verb':act.verb,'arg':act.arg})
            self._robot_operation_category=None
        for i,g in enumerate(self.config.groups):
            if self.task_signals[i,2] and all(self.role_filled(p) for p in g.part_ids) and all(self.joints_done[j] for j in g.joint_ids):
                self.task_signals[i,2]=0.
                self._log_event({'type':'task_commitment_released','group':g.name,'reason':'complete'})

    def _observation(self):
        return {**super()._observation(),'task_signals':self.task_signals.copy()}

    def metrics_summary(self,success):
        return {**super().metrics_summary(success),**self.preference_counts}
