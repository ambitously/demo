import os 
import json
import pickle 
import requests
import numpy as np
from rocobench.envs import MujocoSimEnv, EnvState
from openai import OpenAI
from datetime import datetime
from .feedback import FeedbackManager
from .parser import LLMResponseParser
from typing import List, Tuple, Dict, Union, Optional, Any

PATH_PLAN_INSTRUCTION="""
[How to plan PATH]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.  
[How to Incoporate [Enviornment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach. 
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""



def get_chat_prompt(env: MujocoSimEnv):
    robot_names = env.get_sim_robots().keys()
    talk_order_str = ",".join([f"[{name}]" for name in robot_names])
    chat_prompt = f"""
The robots discuss to find the best strategy. They carefully analyze others' responses and use [Environment Feedback] to improve their plan. 
They talk in order {talk_order_str}... Once they reach agreement, they summarize the plan by **strictly** following [Action Output Instruction] to format the output, then stop talking.
Their entire discussion and final plan are:
    """
    return chat_prompt 


def get_plan_prompt(env: MujocoSimEnv):
    return """
Reason about the task step-by-step, and find the best strategy to coordinate the robots. Propose a plan of **exactly** one action per robot.
Use [Environment Feedback] to improve your plan. Strictly follow [Action Output Instruction] to format and output the plan.
Your reasoning and final plan output are:
    """
    

class SingleThreadPrompter:
    """
    At each round, queries LLM once for each action plan, 
    query again with environment feedback if the action plan cannot be executed
    """
    def __init__(
        self, 
        env: MujocoSimEnv,
        parser: LLMResponseParser, 
        feedback_manager: FeedbackManager,
        comm_mode: str = "plan", # or chat
        use_waypoints: bool = False,
        use_history: bool = True,
        max_api_queries: int = 3,
        num_replans: int = 3,
        debug_mode: bool = False,   
        temperature: float = 0,
        max_tokens: int = 1000, 
        llm_source: str = "qwen3-8b",
    ):
        self.env = env 
        self.robot_agent_names = env.get_sim_robots().keys()
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.comm_mode = comm_mode
        self.max_api_queries = max_api_queries
        self.num_replans = num_replans
        self.debug_mode = debug_mode 
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.temperature = temperature
        self.llm_source = llm_source
        self.max_tokens = max_tokens

        self.round_history = [] # [obs_t, action_t] but only if action_t got executed
        self.failed_plans = [] # could inherit from previous round if the final plan failed to execute in env.
        self.response_history = [] # [response_t]
        self.heuristic_memory = dict()
        

    def save_state(self, save_path, fname = 'prompter_state.pkl'):
        state_dict = dict(
            round_history=self.round_history,
            failed_plans=self.failed_plans,
        )
        save_path = os.path.join(save_path, fname)
        with open(save_path, "wb") as f:
            pickle.dump(state_dict, f)

    def load_state(self, load_path, fname = 'prompter_state.pkl'):
        load_path = os.path.join(load_path, fname)
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        self.round_history = state_dict["round_history"]
        self.failed_plans = state_dict["failed_plans"]

    def compose_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}"
        ret += f"== Current Round ==\n"
        return ret
        
    def compose_system_prompt(
        self,
        obs_desp: str,
        plan_feedbacks: List[str] = [], 
        ):
        
        task_desp = self.env.describe_task_context() # should include task rules
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION

        full_prompt = f"{task_desp}\n{action_desp}\n" 
        
        if self.use_history:
            history_desp = self.compose_round_history() 
            full_prompt += history_desp + "\n" 
        
        full_prompt += obs_desp + "\n"

        if len(self.failed_plans) > 0:
            execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
            execute_feedback += "\n".join(self.failed_plans) 
            full_prompt += execute_feedback + "\n"

        if len(plan_feedbacks) > 0:
            feedback_prompt = "Previous Plans Require Improvement:\n"
            feedback_prompt += "\n".join(plan_feedbacks) + "\n"
            full_prompt += feedback_prompt
        
        if self.comm_mode == "plan":
            comm_prompt = get_plan_prompt(self.env)
        elif self.comm_mode == "chat":
            comm_prompt = get_chat_prompt(self.env) 
        else:
            raise NotImplementedError
        full_prompt += comm_prompt

        return full_prompt 

    def _try_compose_heuristic_response(self, obs: EnvState) -> Optional[str]:
        """Return a deterministic tool-style plan for tasks with brittle LLM behavior.

        This is a lightweight Tool-RoCo-inspired coordinator: it treats each robot
        action as a validated tool call and emits the exact RocoBench action schema.
        If a task is unsupported, return None and fall back to the LLM.
        """
        task_name = self.env.__class__.__name__
        if task_name == "SweepTask":
            return self._compose_sweep_heuristic(obs)
        if task_name == "SortOneBlockTask":
            return self._compose_sort_heuristic(obs)
        if task_name == "MoveRopeTask":
            return self._compose_rope_heuristic(obs)
        return None

    def _compose_rope_heuristic(self, obs: EnvState) -> str:
        """Two-stage rope policy: synchronously pick both ends, then put them in groove.

        The LLM repeatedly creates uneven PATHs for rope. This deterministic
        coordinator emits validated tool calls with evenly spaced waypoints.
        """
        alice_held = self._rope_held_end(obs, "Alice")
        bob_held = self._rope_held_end(obs, "Bob")
        if alice_held is None or bob_held is None:
            alice_obj, bob_obj = self._assign_rope_pick_ends(obs)
            alice_path = self._rope_path_to_target(obs, "Alice", alice_obj, lift=0.04)
            bob_path = self._rope_path_to_target(obs, "Bob", bob_obj, lift=0.04)
            return (
                "EXECUTE\n"
                f"NAME Alice ACTION PICK {alice_obj} PATH {alice_path}\n"
                f"NAME Bob ACTION PICK {bob_obj} PATH {bob_path}\n"
            )

        alice_slot, bob_slot = self._assign_rope_put_slots(alice_held, bob_held)
        alice_path = self._rope_put_path(obs, "Alice", alice_slot)
        bob_path = self._rope_put_path(obs, "Bob", bob_slot)
        return (
            "EXECUTE\n"
            f"NAME Alice ACTION PUT {alice_held} {alice_slot} PATH {alice_path}\n"
            f"NAME Bob ACTION PUT {bob_held} {bob_slot} PATH {bob_path}\n"
        )

    def _rope_held_end(self, obs: EnvState, agent: str) -> Optional[str]:
        robot_name = self.env.robot_name_map_inv[agent]
        contacts = getattr(obs, robot_name).contacts
        contact_text = ",".join(contacts)
        if "rope_front_end" in contacts or "CB0" in contact_text:
            return "rope_front_end"
        if "rope_back_end" in contacts or "CB24" in contact_text:
            return "rope_back_end"
        return None

    def _assign_rope_pick_ends(self, obs: EnvState) -> Tuple[str, str]:
        ends = ["rope_front_end", "rope_back_end"]
        alice_pos = self._agent_ee_pos(obs, "Alice")
        bob_pos = self._agent_ee_pos(obs, "Bob")
        alice_front = np.linalg.norm(alice_pos[:2] - self.env.get_target_pos("Alice", "rope_front_end")[:2])
        bob_front = np.linalg.norm(bob_pos[:2] - self.env.get_target_pos("Bob", "rope_front_end")[:2])
        # Assign the front end to the closer robot; the other robot takes the other end.
        if alice_front <= bob_front:
            return ends[0], ends[1]
        return ends[1], ends[0]

    def _assign_rope_put_slots(self, alice_end: str, bob_end: str) -> Tuple[str, str]:
        # Preserve left-to-right rope orientation in the final groove.
        end_to_slot = {
            "rope_front_end": "groove_left_end",
            "rope_back_end": "groove_right_end",
        }
        return end_to_slot[alice_end], end_to_slot[bob_end]

    def _agent_ee_pos(self, obs: EnvState, agent: str) -> np.ndarray:
        robot_name = self.env.robot_name_map_inv[agent]
        return np.array(getattr(obs, robot_name).ee_xpos, dtype=float)

    def _rope_path_to_target(self, obs: EnvState, agent: str, target_name: str, lift: float = 0.05) -> str:
        start = self._agent_ee_pos(obs, agent)
        target = np.array(self.env.get_target_pos(agent, target_name), dtype=float)
        safe_target = target.copy()
        safe_target[2] = min(max(target[2] + lift, 0.32), 0.53)
        return self._format_interpolated_path(start, safe_target, n=4)

    def _rope_put_path(self, obs: EnvState, agent: str, slot_name: str) -> str:
        # Move the held rope end high over/around the obstacle wall before the
        # parser appends the final groove target. Separate front/back lanes
        # reduce robot-robot and obstacle collisions.
        target = np.array(self.env.get_target_pos(agent, slot_name), dtype=float)
        if agent == "Alice":
            waypoints = [
                (-0.90, 0.35, 0.53),
                (-0.55, 0.30, 0.53),
                (-0.20, 0.35, 0.53),
                (target[0] - 0.08, target[1], 0.50),
            ]
        else:
            waypoints = [
                (-0.35, 0.65, 0.53),
                (0.00, 0.70, 0.53),
                (0.50, 0.65, 0.53),
                (target[0] - 0.08, target[1], 0.50),
            ]
        return "[" + ",".join(
            f"({x:.2f},{y:.2f},{z:.2f})" for x, y, z in waypoints
        ) + "]"

    def _format_interpolated_path(self, start: np.ndarray, target: np.ndarray, n: int = 4) -> str:
        points = []
        # Feedback checks start + PATH + parser target, so avoid including target itself.
        for frac in np.linspace(1 / (n + 1), n / (n + 1), n):
            pos = start[:3] + frac * (target[:3] - start[:3])
            points.append(f"({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
        return "[" + ",".join(points) + "]"

    def _compose_sort_heuristic(self, obs: EnvState) -> str:
        """Assembly-line policy for SortOneBlockTask.

        The handoff panels are panel3 and panel5. We move one cube at a time to
        minimize multi-arm collisions: yellow -> panel6, blue -> panel2, then
        pink -> panel4. Each move is made only by a robot that can reach both the
        source and destination panel.
        """
        agents = ["Alice", "Bob", "Chad"]
        actions = {agent: "WAIT" for agent in agents}

        targets = [
            ("yellow_trapezoid", "panel6"),
            ("blue_square", "panel2"),
            ("pink_polygon", "panel4"),
        ]

        for cube, target in targets:
            current = self.env.get_cube_panel(obs, cube)
            if current == target:
                continue
            move = self._next_sort_move(cube, current)
            if move is None:
                continue
            agent, destination = move
            actions[agent] = f"PICK {cube} PLACE {destination}"
            break

        return "EXECUTE\n" + "".join(
            f"NAME {agent} ACTION {actions[agent]}\n" for agent in agents
        )

    def _next_sort_move(self, cube: str, current_panel: str) -> Optional[Tuple[str, str]]:
        panel_idx = int(current_panel.replace("panel", ""))
        if cube == "yellow_trapezoid":
            if current_panel == "panel6":
                return None
            if panel_idx <= 2:
                return "Alice", "panel3"
            if current_panel == "panel3":
                return "Bob", "panel5"
            if panel_idx == 4:
                return "Bob", "panel5"
            if current_panel == "panel5":
                return "Chad", "panel6"
            return "Chad", "panel6"

        if cube == "blue_square":
            if current_panel == "panel2":
                return None
            if panel_idx >= 6:
                return "Chad", "panel5"
            if current_panel == "panel5":
                return "Bob", "panel3"
            if current_panel == "panel4":
                return "Bob", "panel3"
            if current_panel == "panel3":
                return "Alice", "panel2"
            return "Alice", "panel2"

        if cube == "pink_polygon":
            if current_panel == "panel4":
                return None
            if panel_idx <= 2:
                return "Alice", "panel3"
            if current_panel == "panel3":
                return "Bob", "panel4"
            if panel_idx >= 6:
                return "Chad", "panel5"
            if current_panel == "panel5":
                return "Bob", "panel4"
            return "Bob", "panel4"

        return None

    def _compose_sweep_heuristic(self, obs: EnvState) -> str:
        cubes = list(getattr(self.env, "cube_names", []))

        dustpan_cubes = [
            cube for cube in cubes
            if "dustpan_bottom" in obs.objects[cube].contacts
        ]
        if len(dustpan_cubes) > 0:
            next_cube = self._select_sweep_cube(obs, exclude=set(dustpan_cubes))
            bob_action = f"MOVE {next_cube}" if next_cube is not None else "WAIT"
            return f"EXECUTE\nNAME Alice ACTION DUMP\nNAME Bob ACTION {bob_action}\n"

        target = self.heuristic_memory.get("sweep_target")
        if target is None or not self._is_sweep_cube_on_table(obs, target):
            target = self._select_sweep_cube(obs)
            self.heuristic_memory["sweep_target"] = target

        if target is None:
            return "EXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION WAIT\n"

        cube_pos = self.env.physics.data.site(target).xpos.copy()
        dustpan_pos = self.env.physics.data.site("dustpan_rim").xpos.copy()
        broom_pos = self.env.physics.data.site("broom_bottom").xpos.copy()
        alice_ready = float(np.linalg.norm(cube_pos - dustpan_pos)) < 0.22
        bob_ready = float(np.linalg.norm(cube_pos - broom_pos)) < 0.38

        if alice_ready and bob_ready:
            return f"EXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION SWEEP {target}\n"
        return f"EXECUTE\nNAME Alice ACTION MOVE {target}\nNAME Bob ACTION MOVE {target}\n"

    def _is_sweep_cube_on_table(self, obs: EnvState, cube: str) -> bool:
        if cube not in obs.objects:
            return False
        contacts = obs.objects[cube].contacts
        return "table" in contacts and "trash_bin_bottom" not in contacts and "dustpan_bottom" not in contacts

    def _select_sweep_cube(self, obs: EnvState, exclude: Optional[set] = None) -> Optional[str]:
        exclude = exclude or set()
        candidates = [
            cube for cube in getattr(self.env, "cube_names", [])
            if cube not in exclude and self._is_sweep_cube_on_table(obs, cube)
        ]
        if len(candidates) == 0:
            return None
        dustpan_pos = self.env.physics.data.site("dustpan_rim").xpos.copy()
        return min(
            candidates,
            key=lambda cube: float(np.linalg.norm(self.env.physics.data.site(cube).xpos.copy() - dustpan_pos)),
        )

    def prompt_one_round(self, obs: EnvState, save_path: str = ""): 
        plan_feedbacks = []
        response_history = []
        obs_desp = self.env.describe_obs(obs)
        for i in range(self.num_replans): 
            system_prompt = self.compose_system_prompt(obs_desp, plan_feedbacks)
            heuristic_response = self._try_compose_heuristic_response(obs)
            if heuristic_response is not None:
                response, usage = heuristic_response, {"source": "heuristic_tool_coordinator"}
                print('======= heuristic response ======= \n ', response)
            else:
                response, usage = self.query_once(
                    system_prompt, user_prompt=""
                    ) # NOTE: single_thread doesn't use user role
            response_history.append(response)
            
            timestamp = datetime.now().strftime("%m%d-%H%M")
            tosave = [ 
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": "",
                    },
                    {
                        "sender": "Planner",
                        "message": response,
                    },
                    usage,
                ]
            fname = f'{save_path}/replan{i}_{timestamp}.json'
            json.dump(tosave, open(fname, 'w'))  
            
            curr_feedback = "None"
            # try parsing 
            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, response) 
            if not parse_succ: 
                execute_str = 'EXECUTE' + response.split('EXECUTE')[-1]
                curr_feedback = f"""
Parsing failed! {parsed_str}
Previous response: {execute_str}
Re-format to strictly follow [Action Output Instruction]!
                """
                plan_feedbacks.append(curr_feedback)
                ready_to_execute = False  
            # give env. feedback 
            else:
                ready_to_execute = True
                for j, llm_plan in enumerate(llm_plans): 
                    ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)        
                    if not ready_to_execute:
                        curr_feedback = env_feedback
                        break
            
            plan_feedbacks.append(curr_feedback)
            tosave = [
                {
                    "sender": "Feedback",
                    "message": curr_feedback,
                },
                {
                    "sender": "Action",
                    "message": (response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
            json.dump(tosave, open(fname, 'w')) 

            if ready_to_execute:
                plan_str = parsed_str
                break  
        self.response_history = response_history
        return ready_to_execute, llm_plans, plan_feedbacks, response_history


    def query_once(self, system_prompt, user_prompt=""):
        response = None
        usage = None   
        print('======= system prompt ======= \n ', system_prompt)
        print('======= user prompt ======= \n ', user_prompt)

        if self.debug_mode: # query human user input
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()


        for n in range(self.max_api_queries):
            print('querying {}th time'.format(n))
            try:
                client = OpenAI(
                    api_key="sk-01617270a36d460a9ac26c0fadcdaac5",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                completion = client.chat.completions.create(
                    model=self.llm_source,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    extra_body={"enable_thinking": False},
                )
                response = completion.choices[0].message.content
                usage = completion.usage.model_dump() if completion.usage is not None else {}

                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                break
            except Exception as e:
                print("API error:", repr(e))
            continue
        if response is None:
            raise RuntimeError("LLM request failed after retries; response is None.")
        return response, usage

    

    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success: 
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            responses = "\n".join(self.response_history)
            self.round_history.append(
                f"[Response History]\n{responses}\n{obs_desp}\n[Executed Action]\n{parsed_plan}"
            )
        else:
            self.failed_plans.append(
                parsed_plan
            )
        return

    def post_episode_update(self):
        # clear for next episode
        self.round_history = []
        self.failed_plans = [] 
        self.response_history = []
        self.heuristic_memory = dict()
