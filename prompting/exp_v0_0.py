import operator, logging, json, random, uuid, time
import joblib
import numpy as np
from typing import Any, Iterable, List, Dict, Optional, TypedDict
from typing import TypedDict, Annotated

from traitlets import Int
from mcbs.config import settings
from mcbs.llm_api.hm_langchain_generator import LlmApiGenerator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage
from prompting.params_v0_1 import SetParams
from prompting.exp_logging import Logging
    
class State(TypedDict):
    # Input
    thread_id: str # 스레드 ID
    input_prompt_ver: str # 입력 프롬프트 버전
    case_index_list: List[str] # 테스트 케이스 '00', '01', '02', ... 중 랜덤 N개 리스트
    cami_prompt: str # 챗봇 시스템 인스트럭션
    exp_iteration: str # 실험 iteration '00', '01', '02', '03', ...
    prompt_save_path: str # 개선된 프롬프트 저장 경로
    # Output
    response_record: Dict[str, Dict[str, Any]] # 대화 response 전체 기록
    chat_results: Dict[str, Dict[str, Any]] # 대화 결과
    evals_results: Dict[str, Dict[str, Any]] # 평가 결과(전처리 후)
    agg_evals_results: Dict[str, Dict[str, Any]] # 평가 결과(집계 후)
    evals_feedback: str # 평가 결과(str 정리본)
    problem_feedback: str # Problem Organizer 피드백
    new_prompt: str # Prompt Improver 피드백
    # Final Output
    exp_results: Dict[str, Any] # 실험 전체 데이터 (exp 축적)

class GraphExecutor:
    def __init__(self, preconfig: classmethod):
        f"""
        param_set: dict
        preconfig: classmethod (PreConfig)
            - initial_prompt_ver: str
            - initial_message_ver: str
            - cami_direction_ver: str
            - wdir: str
            - counseling_guide_set: List[str]
            - tester_persona_set: List[str]
            - eval_score_table: pd.DataFrame
            - initial_message: str
            - cami_direction: str
            - limited_turn: int
        """
        self.preconfig = preconfig
        self.params_dict = None
        self.logging = None
        self.llm_api_generators = None
        self.scoring_item_numbers = [f"scoring{k}" for k in self.preconfig.eval_score_table.item_number.tolist()]

    def get_param_set(self, case_index, cami_prompt):
        set_params = SetParams(
            cami_prompt=cami_prompt, 
            counseling_guide=self.preconfig.counseling_guide_set[case_index],
            tester_persona=self.preconfig.tester_persona_set[case_index],
            eval_score_table=self.preconfig.eval_score_table,
            cami_direction=self.preconfig.cami_direction
            )
        param_set = set_params.params
        return param_set

    def prepare_exp(self, state: State) -> State:
        self.start_time = time.time()
        self.llm_api_generators = {
            'counselor': {},
            'tester': {},
            'judge1': {},
            'judge2': {},
            'problem_organizer': {},
            'prompt_improver': {}
        }
        for role in ['counselor', 'tester', 'judge1', 'judge2', 'problem_organizer', 'prompt_improver']:
            if role in ['problem_organizer', 'prompt_improver']:
                self.llm_api_generators[role] = LlmApiGenerator()
            else:
                for case_index in state['case_index_list']:
                    self.llm_api_generators[role][case_index] = LlmApiGenerator()
        

        input_prompt_ver = state['input_prompt_ver']
        exp_iteration = int(state['exp_iteration']) +1
        exp_iteration = f"{exp_iteration:02d}"

        prompt_file_path = state.get('prompt_save_path', None)
        if prompt_file_path is None:
            prompt_file_path = f"{self.preconfig.wdir}/cami_prompt/cami_prompt_{input_prompt_ver}.md"
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            cami_prompt = f.read()
        prompt_save_path = f"{self.preconfig.wdir}/cami_prompt/cami_prompt_{input_prompt_ver}_{exp_iteration}.md"

        params_dict = {}        
        for case_index in state['case_index_list']:
            param_set = self.get_param_set(
                    case_index=case_index,
                    cami_prompt=cami_prompt
                )
            params_dict[case_index] = param_set
        self.params_dict = params_dict

        self.logging = Logging(
            save_dir=f"{self.preconfig.wdir}/exp_record/exp_log_{input_prompt_ver}_{exp_iteration}.csv",
            exp_iteration=exp_iteration
        )

        # 로그: 실험 준비 완료
        try:
            if self.logging:
                log_row = self.logging.log_row(
                    event_name='exp_preparing',
                    data={
                        'status': 'ready',
                        'input_prompt_ver': input_prompt_ver,
                    },
                    additional_info={
                        'case_index_list': state.get('case_index_list', []),
                        'save_dir': self.logging.save_dir
                    },
                    exp_iteration=exp_iteration
                )
                self.logging.save_logging(log_row)
        except Exception as e:
            logging.warning(f"exp_preparing logging failed: {e}")

        return {
            'exp_iteration': exp_iteration,
            'cami_prompt': cami_prompt,
            'prompt_save_path': prompt_save_path
        }

    def _generate_chat_message(
                                self,
                                param_set,
                                current_role: str,
                                case_index: str,
                                last_message: dict,
                                turn: int
                                ) -> str:
        
        counter_role = 'counselor' if current_role == 'tester' else 'tester'
        current_query = last_message[counter_role]
        chat_history = self.llm_api_generators[current_role][case_index].message_history

        message = self.llm_api_generators[current_role][case_index].generate_message(
            query=current_query,
            params=param_set[current_role],
            tagname=current_role,
            message_history=chat_history
        )

        if current_role == 'counselor':
            message = message.replace('```json', '').replace('```', '')
            role_message_json = json.loads(message)
            front_message = role_message_json['front_message']
        elif current_role == 'tester':
            front_message = message

        self.llm_api_generators[current_role][case_index].update_message_history(
            query=current_query,
            response_text=message
        )
        role_last_message = front_message

        # 로그: 채팅 생성
        try:
            if self.logging:
                log_row = self.logging.log_row(
                    event_name='chat_generated...',
                    info=f'Case{case_index}...[{turn}]{current_role}',
                )
                self.logging.save_logging(log_row)
        except Exception as e:
            logging.warning(f"chat_generated... logging failed: {e}")

        return front_message, role_last_message
        
    def _chat_process_single(self, state:State, case_index: str) -> dict:
 
        param_set = self.params_dict[case_index]
        last_message = {
            'counselor' : '',
            'tester' : self.preconfig.initial_message
        }
        chat_window = '### Chat Record\n\n'
        for turn in range(1, self.preconfig.limited_turn+1):
            chat_window += f"**Turn[{turn}]**\n\n"
            for current_role in ['counselor', 'tester']:
                front_message, last_message[current_role] = self._generate_chat_message(
                    param_set=param_set,
                    current_role=current_role,
                    case_index=case_index,
                    last_message=last_message,
                    turn=turn
                )
                chat_window += f"""
> {current_role}

{front_message}

"""
            # 챗봇 이탈 시 대화 루프 종료...
            if '[챗봇 이탈]' in front_message:
                break

        # 로그: 단일 케이스 채팅 종료
        try:
            if self.logging:
                log_row = self.logging.log_row(
                    event_name='chat_case_completed',
                    data={
                        'case_index': case_index,
                        'churn_turn': turn
                    },
                    info=f'Case{case_index}...\n\n{chat_window}',
                )
                self.logging.save_logging(log_row)
        except Exception as e:
            logging.warning(f"chat_case_completed logging failed: {e}")

        return {
            "case_index": case_index,
            "chat_window": chat_window,
            "chat_history": self.llm_api_generators['counselor'][case_index].message_history,
            "churn_turn": turn
        }

    def chat_process(self, state: State) -> State:
        # 로그: 채팅 프로세스 시작
        try:
            if self.logging:
                log_row = self.logging.log_row(
                    event_name='chat_process_started',
                    data={'num_cases': len(state['case_index_list'])},
                    additional_info={'case_index_list': state['case_index_list']},
                )
                self.logging.save_logging(log_row)
        except Exception as e:
            logging.warning(f"chat_process_started logging failed: {e}")

        runner = RunnableLambda(lambda case_index: self._chat_process_single(state=state, case_index=case_index)).map()
        chat_results_list = runner.invoke(state['case_index_list'])
        chat_results = {}
        for chat_result in chat_results_list:
            key = chat_result['case_index']
            chat_results[key] = {k: v for k, v in chat_result.items() if k != 'case_index'}

        # 로그: 채팅 프로세스 종료
        try:
            if self.logging:
                churns = [res.get('churn_turn', None) for res in chat_results]
                log_row = self.logging.log_row(
                    event_name='chat_process_completed',
                    data={
                        'num_cases': len(chat_results),
                        'churn_turns': churns
                    },
                )
                self.logging.save_logging(log_row)
        except Exception as e:
            logging.warning(f"chat_process_completed logging failed: {e}")
        return {
            "chat_results": chat_results
        }

    def _eval_process_single(self, state: State, judge_id: str, chat_window: str, case_index: str) -> State:
        query = f"""
> **까미 상담 대화 기록** 

{chat_window}
"""
        any_key = list(self.params_dict.keys())[0]
        eval_param = self.params_dict[any_key][judge_id]
        response = self.llm_api_generators[judge_id][case_index].generate_message(
            query=query,
            params=eval_param,
            tagname='evaluation'
        )
        # 로그: 단일 평가 종료
        try:
            if self.logging:
                model_name = eval_param['model']
                log_row = self.logging.log_row(
                    event_name='eval_single_completed',
                    info=f'Case{case_index}...[{judge_id}]{model_name}',
                    data={'judge_id': judge_id, 'response_len': len(response) if isinstance(response, str) else 0},
                )
                self.logging.save_logging(log_row)
        except Exception as e:
            logging.warning(f"eval_single_completed logging failed: {e}")
        
        response = response.replace('```json', '').replace('```', '')
        return json.loads(response)

    def eval_process(self, state: State) -> State:        
        chat_results = state['chat_results']
        evals_results = {}
        for case_index in chat_results.keys():
            chat_window = chat_results[case_index]['chat_window']
            runner = RunnableLambda(lambda t: self._eval_process_single(
                state=state, 
                judge_id=t, 
                chat_window=chat_window, 
                case_index=case_index)
                ).map()
            evals_result = runner.invoke(['judge1', 'judge2'])
            evals_result_cleaned = self._get_eval_case(evals_result)
            evals_results[case_index] = evals_result_cleaned

        # 로그: 평가 프로세스 종료
        try:
            if self.logging:
                log_row = self.logging.log_row(
                    event_name='eval_process_completed',
                    data={'num_results': len(evals_results)},
                )
                self.logging.save_logging(log_row)
        except Exception as e:
            logging.warning(f"eval_process_completed logging failed: {e}")
        return {
            "evals_results": evals_results
        }

    def _get_eval_case(self, tmp):
        def _get_score(tmp, score_order):
            score_list = [tmp[c][score_order]['score'] for c in range(len(tmp))]
            return int(np.mean(score_list))

        def _get_score_reason(tmp, score_order):
            score_reason_list = [tmp[c][score_order]['reason'] for c in range(len(tmp))]
            return "\n\n".join(score_reason_list)

        evals_case = {}
        for scoring_col in self.scoring_item_numbers:
            evals_case[scoring_col] = _get_score(tmp, scoring_col)
            evals_case[f'{scoring_col}_reason'] = _get_score_reason(tmp, scoring_col)
        return evals_case

    def agg_eval_results(self, state: State) -> State:
        evals_results = state['evals_results']
        agg_evals_results = {
            scoring_col: {
                'score': [],
                'reason': []
            } for scoring_col in self.scoring_item_numbers
        }

        for case_index in evals_results.keys():
            tmp = evals_results[case_index]
            for scoring_col in self.scoring_item_numbers:
                agg_evals_results[scoring_col]['score'].append(tmp[scoring_col])
                agg_evals_results[scoring_col]['reason'].append(tmp[f'{scoring_col}_reason'])
        
        for scoring_col in self.scoring_item_numbers:
            agg_evals_results[scoring_col]['score_mean'] = float(np.mean(agg_evals_results[scoring_col]['score']))
            agg_evals_results[scoring_col]['score_std'] = float(np.std(agg_evals_results[scoring_col]['score']))
            agg_evals_results[scoring_col]['reasons'] = "\n\n".join(agg_evals_results[scoring_col]['reason'])

        eval_score_table = self.preconfig.eval_score_table
        weight_map = {
            scoring_col: eval_score_table.loc[eval_score_table['item_number'] == int(scoring_col.replace('scoring', '')), 'weight'].values[0]
            for scoring_col in self.scoring_item_numbers
        }
        agg_evals_results['scoring_total'] = {
            'score': []
        }
        agg_evals_results['scoring_total']['score'] = [
            float(agg_evals_results[scoring_col]['score_mean'] * weight_map[scoring_col])
            for scoring_col in self.scoring_item_numbers
        ]
        agg_evals_results['scoring_total']['score_sum'] = sum(agg_evals_results['scoring_total']['score'])

        agg_evals_results['churn'] = {
            'churn_turn': [state['chat_results'][case_index]['churn_turn'] for case_index in state['chat_results'].keys()]
        }
        agg_evals_results['churn']['churn_turn_mean'] = float(np.mean(agg_evals_results['churn']['churn_turn']))
        agg_evals_results['churn']['churn_turn_std'] = float(np.std(agg_evals_results['churn']['churn_turn']))

        return {
            'agg_evals_results': agg_evals_results
        }

    def _convert_agg_evals(self, state: State) -> str:
        agg_evals_results = state['agg_evals_results']
        prompt = state['cami_prompt']
        drop_index = prompt.find("### Dog Counseling Guide")
        input_prompt = prompt[:drop_index]

        feedback_md = f"""
# 상담 챗봇 까미 성능 평가 보고서

### 프롬프트
/// 프롬프트 시작.

{input_prompt}

/// 프롬프트 끝.

---

### 까미 챗봇 지향점

{self.preconfig.cami_direction}

### 까미 챗봇 평가 기준

{self.preconfig.eval_score_table.to_markdown()}

---

### 평가 결과

#### 1. 요약
- 평균 채팅 이탈턴수: {agg_evals_results['churn']['churn_turn_mean']:.1f} (표준편차: {agg_evals_results['churn']['churn_turn_std']:.1f}) [최대:{self.preconfig.limited_turn}]
- 총점: {agg_evals_results['scoring_total']['score_sum']:.1f}

#### 2. 상세
"""
        for scoring_col in self.scoring_item_numbers:
            feedback_md += f"""
- 점수: {agg_evals_results[scoring_col]['score_mean']:.1f} (std: {agg_evals_results[scoring_col]['score_std']:.1f}) [총 10점]
- 이유:
```
{agg_evals_results[scoring_col]['reasons']}
```
"""
        return feedback_md

    def problem_organizing_process(self, state: State) -> State:
        any_key = list(self.params_dict.keys())[0]
        param = self.params_dict[any_key]['problem_organizer']
        feedback_md= self._convert_agg_evals(state)
        message = self.llm_api_generators['problem_organizer'].generate_message(
            query=feedback_md,
            params=param,
            tagname='problem_organizer'
        )

        # 로그: 문제 정리 종료
        try:
            if self.logging:
                log_row = self.logging.log_row(
                    event_name='problem_organizing_completed',
                    info=f"Input:\n\n{feedback_md}")
                self.logging.save_logging(log_row)
        except Exception as e:
            logging.warning(f"problem_organizing_completed logging failed: {e}")
        return {
            'evals_feedback': feedback_md,
            'problem_feedback': message
        }

    def prompt_improver_process(self, state: State) -> State:
        any_key = list(self.params_dict.keys())[0]
        param = self.params_dict[any_key]['prompt_improver']

        prompt = state['cami_prompt']
        drop_index = prompt.find("### Dog Counseling Guide")
        input_prompt = prompt[:drop_index]

        current_query = f"""
> **현재 프롬프트**

{input_prompt}

---

### 까미 챗봇 지향점

{self.preconfig.cami_direction}

### 까미 챗봇 평가 기준

{self.preconfig.eval_score_table.to_markdown()}

---

> **개선 요청**

{state['problem_feedback']}
"""
        message = self.llm_api_generators['prompt_improver'].generate_message(
            query=current_query,
            params=param,
            tagname='prompt_improver'
        )
        # 로그: 프롬프트 개선 완료
        try:
            if self.logging:
                log_row = self.logging.log_row(
                    event_name='prompt_improver_completed',
                    info=f"Input:\n\n{current_query}"
                    )
                self.logging.save_logging(log_row)
        except Exception as e:
            logging.warning(f"prompt_improver_completed logging failed: {e}")
        return {
            'new_prompt': message
        }

    def finish_exp_iteration(self, state: State) -> State:
        total_response_record = {}
        for role in self.llm_api_generators.keys():
            if role in ['counselor', 'tester', 'judge1', 'judge2']:
                total_response_record[role] = {
                    case_index: value.all_responses
                    for case_index, value in self.llm_api_generators[role].items()
                }
            else:
                total_response_record[role] = self.llm_api_generators[role].all_responses
        
        current_exp_content = {
            'thread_id': state['thread_id'],
            'input_prompt_ver': state['input_prompt_ver'],
            'case_index_list': state['case_index_list'],
            'cami_prompt': state['cami_prompt'],
            'prompt_save_path': state['prompt_save_path'],
            'response_record': total_response_record,
            'chat_results': state['chat_results'],
            'evals_results': state['evals_results'],
            'agg_evals_results': state['agg_evals_results'],
            'evals_feedback': state['evals_feedback'],
            'problem_feedback': state['problem_feedback'],
            'exp_score': state['agg_evals_results']['scoring_total']['score_sum'],
            'exp_time': (time.time() - self.start_time),
            'new_prompt': state['new_prompt']
        }

        exp_contents = state.get('exp_results', {})
        exp_contents[state['exp_iteration']] = current_exp_content

        with open(state['prompt_save_path'], 'w', encoding='utf-8') as f:
            f.write(current_exp_content['new_prompt'])

        # 로그: 실험 이터레이션 종료 및 다음 케이스 샘플링
        try:
            if self.logging:
                log_row = self.logging.log_row(
                    event_name='finish_exp_iteration',
                    data={'exp_iteration': state['exp_iteration']},
                    prompt_save_path=state['prompt_save_path'],
                    exp_iteration=self.logging.exp_iteration
                )
                self.logging.save_logging(log_row)
        except Exception as e:
            logging.warning(f"finish_exp_iteration logging failed: {e}")
        # 이터레이션 반복 전에 스냅샷 저장 (현 시점 state + 최신 exp_results 병합본)
        try:
            snapshot_state = dict(state)
            snapshot_state['exp_results'] = exp_contents
            snapshot_path = f"{self.preconfig.wdir}/exp_record/exp_result_{self.preconfig.initial_prompt_ver}_snopshot.pkl"
            joblib.dump(snapshot_state, snapshot_path)
        except Exception as e:
            logging.warning(f"snapshot saving failed: {e}")
        return {
            'exp_results': exp_contents,
            'case_index_list': random.sample([f"{i:02d}" for i in range(len(self.preconfig.tester_persona_set))], 
                                k=self.preconfig.case_sampling_k), # 다음 실험 케이스 랜덤 선택
        }
    
class RunAgentsProcess:
    def __init__(self, preconfig: classmethod):
        self.preconfig = preconfig
        self.graph = self._build_graph()

    def _build_graph(self):
        executor = GraphExecutor(self.preconfig)

        builder = StateGraph(State)
        builder.add_node("exp_preparing", executor.prepare_exp)
        builder.add_node("chat", executor.chat_process)
        builder.add_node("eval", executor.eval_process)
        builder.add_node("agg_eval", executor.agg_eval_results)
        builder.add_node("problem_organizing", executor.problem_organizing_process)
        builder.add_node("prompt_improver", executor.prompt_improver_process)
        builder.add_node("finish_exp_iteration", executor.finish_exp_iteration)

        builder.add_edge(START, "exp_preparing")
        builder.add_edge("exp_preparing", "chat")
        builder.add_edge("chat", "eval")
        builder.add_edge("eval", "agg_eval")
        builder.add_edge("agg_eval", "problem_organizing")
        builder.add_edge("problem_organizing", "prompt_improver")
        builder.add_edge("prompt_improver", "finish_exp_iteration")
        builder.add_conditional_edges(
            "finish_exp_iteration",
            lambda state: "exp_preparing" if int(state['exp_iteration']) < self.preconfig.max_iteration else END,
            {"exp_preparing": "exp_preparing", END: END}
        )

        graph = builder.compile(checkpointer=InMemorySaver())
        return graph

    def run_graph_agents(self, thread_id, recursion_limit=10000):
        config = {
            "configurable": {
                "thread_id": thread_id
            }, 
            "recursion_limit": recursion_limit
        }

        result = self.graph.invoke(
            {
                'thread_id': thread_id,
                'input_prompt_ver': self.preconfig.initial_prompt_ver,
                'case_index_list': random.sample([f"{i:02d}" for i in range(len(self.preconfig.tester_persona_set))], k=self.preconfig.case_sampling_k),
                'exp_iteration': '00',
            },
            config=config
        )
        return result