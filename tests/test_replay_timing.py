import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from loghrc.core.env import LogHrcEnv
from loghrc.core.config import ROBOT
from loghrc.core.replay import Replay
from loghrc.humans.scripted import HumanProfile, ScriptedPolicy
from loghrc.humans.profiles.furniture import journal_profiles
from loghrc.tasks.furniture.table import full_table_config, journal_furniture_config


@pytest.mark.parametrize('profile,seed', [('journal_prefer_a',1000004),('journal_proactive',7),('journal_wait_initiative',3)])
def test_python_and_viewer_match_live_hands_on_every_tick(profile,seed):
    config=journal_furniture_config(full_table_config())
    env=LogHrcEnv(config=config,phase=2,human_profile=profile,profile_pool=journal_profiles(config,2))
    live={}
    original=env._check_safety
    def capture():
        original()
        live[env.t]={a:tuple(env.agents[a].hand) for a in (0,1)}
    env._check_safety=capture
    env.reset(seed=seed)
    live[0]={a:tuple(env.agents[a].hand) for a in (0,1)}
    def physical_state():
        return {str(pid): {'pose': list(st.pose.as_tuple()) if st.pose is not None else None,
                          'heldBy': st.held_by, 'targetRole': st.target_role}
                for pid, st in env.parts.items()}
    live_parts = {0: physical_state()}
    original_tick = env._tick
    def capture_tick():
        original_tick()
        live_parts[env.t] = physical_state()
    env._tick = capture_tick
    robot=ScriptedPolicy(HumanProfile('reference',proactivity=1,pause_prob=0,error_prob=0,
        group_order=('leg3','leg4','leg1','leg2')),ROBOT)
    robot.reset(config,np.random.default_rng(seed))
    rng=np.random.default_rng(seed+10000000)
    retreat=False
    try:
        while True:
            verb,arg,reason=('move_home',None,'act') if retreat else robot.decide(env,rng)
            action=env.action_table.index((verb,arg)) if reason=='act' else 0
            _,_,term,trunc,info=env.step(action)
            retreat=verb in ('place_target','attach') and info['valid']
            if term or trunc:break
        record=env.episode_record()
        replay=Replay(record)
        assert replay.verify()
        for frame in replay.states():
            assert {str(pid): {'pose': list(st.pose) if st.pose is not None else None,
                               'heldBy': st.held_by, 'targetRole': st.target_role}
                    for pid, st in frame.parts.items()} == live_parts[frame.tick]
            for agent in (0,1):
                np.testing.assert_allclose(frame.hands[agent],live[frame.tick][agent],atol=.011,rtol=0)
        # Execute the actual viewer's pure reconstruction functions with Node.
        import loghrc
        html=(Path(loghrc.__file__).parent/'viz/viewer/viewer.html').read_text()
        start=html.index('function posAlongPath(')
        end=html.index('return out;',html.index('function buildFrames('))+len('return out;')
        source=html[start:html.index('}',end)+1]
        node=shutil.which('node')
        if node is None:pytest.skip('Node is required for viewer parity')
        script='const ROBOT=0,HUMAN=1;\n'+source+"\nconst ep=JSON.parse(require('fs').readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(buildFrames(ep).map(f=>({tick:f.tick,hands:f.hands,parts:f.parts}))));"
        result=subprocess.run([node,'-e',script],input=json.dumps(record),text=True,capture_output=True,check=True)
        for frame in json.loads(result.stdout):
            assert frame['parts'] == live_parts[frame['tick']]
            for agent in (0,1):
                np.testing.assert_allclose(frame['hands'][str(agent)],live[frame['tick']][agent],atol=.011,rtol=0)
    finally:env.close()
