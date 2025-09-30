from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class CounselorOutput(BaseModel):
    front_message: str = Field(description="Only output the message to be delivered to the conversation partner.")
    back_thinking: str = Field(description="Briefly summarize the Chain-of-Thought process.")

# --- Judge 모델의 Structured Output : Pydantic 모델 정의 ---
class ScoringItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    score: int = Field(ge=0, le=10, description="0~10 척도 점수")
    reason: str = Field(description="해당 점수 선택 이유")

class EvalOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scoring1: ScoringItem = Field(description="item 1: 요구 파악")
    scoring2: ScoringItem = Field(description="item 2: 정확한 정보")
    scoring3: ScoringItem = Field(description="item 3: 정보 수집(충분성)")
    scoring4: ScoringItem = Field(description="item 4: 정보 수집(질문 품질)")
    scoring5: ScoringItem = Field(description="item 5: 원인 추론")
    scoring6: ScoringItem = Field(description="item 6: 개인화")
    scoring7: ScoringItem = Field(description="item 7: 적절성")
    scoring8: ScoringItem = Field(description="item 8: 구체성")
    scoring9: ScoringItem = Field(description="item 9: 설득")
    scoring10: ScoringItem = Field(description="item 10: 어조/말투")
    scoring11: ScoringItem = Field(description="item 11: 장황성/반복")
    
class SetParams:
    def __init__(self, 
            cami_prompt,
            counseling_guide,
            tester_persona,
            eval_score_table,
            cami_direction):
        self.cami_prompt = cami_prompt
        self.counseling_guide = counseling_guide
        self.tester_persona = tester_persona
        self.eval_score_table = eval_score_table
        self.cami_direction = cami_direction

        self.params = {
            'counselor': self.counselor(),
            'tester': self.tester(),
            'judge1': self.judges()[1],
            'judge2': self.judges()[2], 
            'problem_organizer': self.problem_organizer(),
            'prompt_improver': self.prompt_improver(),
        }
        
    def counselor(self):
        return {
            "system_instruction": self.cami_prompt + '\n\n --- \n\n' + self.counseling_guide,
            "model": "gemini-2.5-flash",    
            "temperature": 0.6,
            "structured_output": CounselorOutput,
            "max_output_tokens": 8192
        }
        
    def tester(self):
        system_instruction = f"""
### 역할
- 당신은 반려동물 양육 상담 챗봇 "까미"를 이용하는 반려견 보호자입니다.
- <보호자 페르소나>가 당신의 정보이니, 반드시 따르세요.
- 상대방과의 대화는 휴대폰으로 채팅을 하고 있는 상황입니다.
- 당신은 사람이기 때문에 때로는 실수를 합니다. 챗봇의 말에 논점을 이탈하기도 하거나, 몇몇 질문은 깜박하고 무시해 버릴 수도 있어요.
- 또, 대화를 주도적으로 이끌기 위해 화제를 전환하거나 본인의 감정 상태를 직접 표현하고, 궁금한 내용을 적극적으로 질문해요.
- 실망해서 단답을 하기도 하고, 이탈하기도 합니다.
- 챗봇이 양육팁을 제시했을 때 본인의 상황과 안 맞는 점이 있다면 이야기하세요. 납득이 된 경우에는 이를 한번 해보겠다고 답변하세요.

### 시간의 흐름
- 당신은 챗봇을 한 번만 이용하지 않습니다. 당신은 챗봇이 맘에 들 경우 시간이 지나도 계속 이용할 수 있어요.
- 챗봇이 양육 솔루션을 제공하고, 그에 대해 납득을 하였을 경우에는, 시간이 며칠 뒤로 흘렀다고 가정하고 훈련을 진행한 후기를 들려주세요. 그에 따른 후속 질문을 해주세요.
- 챗봇이 같은 답을 반복할 수 있는데요. 그런 경우 강력하게 비판하거나 시간이 흐른 상황을 만들어서 대화 주제를 바꾸세요. 또는 이탈을 할 수도 있습니다.

### 이탈
- 당신은 챗봇에 대한 만족도가 매우 낮을 경우, 챗봇을 떠날 수도 있습니다. 그런 경우에는 반드시 다음과 같이 '[챗봇 이탈]'과 이탈 이유를 출력하세요.
- '[챗봇 이탈]:{{이탈 이유}}'

### 제한
- 당신은 챗봇을 이용하고 있는 유저(반려견 보호자)라는 것을 절대 잊지 마세요.
- **이전 대화 내용을 똑같이 반복하는 것을 반드시 피하고, 까미 챗봇처럼 말하지 마세요.**

### 출력 양식
- [챗봇 이탈] 혹은 당신이 까미 챗봇에 전할 말을 출력합니다.

{self.tester_persona}
"""
        return {
            "system_instruction": system_instruction,
            "model": "gemini-2.5-flash",
            "temperature": 0.5
        }
    
    def judges(self):
        system_instruction = """
### 역할
당신은 반려견 행동학 전문 수의사입니다.
반려견 양육상담 챗봇 까미의 상담 능력을 평가하고 있어요.
까미와 보호자의 채팅 기록을 입력할테니 까미의 상담 능력을 평가해 주세요.
**채점 기준**을 제공할테니, 해당 기준에 따라 **까미의 점수**를 채점하세요.
**각 항목별 점수 기준은 상담이 지향하는 바를 나타냅니다. 점수 평가와 개선 제안을 반드시 이에 기반하여 제시해 주세요.**

### 출력
- scoring1..scoring11: 아래 [채점 기준]에 따라 **까미의 점수**를 평정하세요.
    - 각 항목은 {"score": 0~10, "reason": string} 객체입니다.
    - "reason"에서는 **직접 주요 대화를 자세히 인용하며** 평가 이유를 설명하고, "score"에는 평가 점수를 출력합니다. ["채점 기준" 준수.] 
    - 채점할만한 대화 내용이 없으면 항목별로 'when_missing_score' 컬럼의 점수값을 score로 기입하고, reason은 "대상 내용 없음"으로 기입하세요.
- json 형식으로 출력. "```json"는 출력에 절대 포함하지 마세요.

#### 출력 예시
'{
    "scoring1": {"score": 3, "reason": "전체적으로 큰 문제는 없었지만, 상담가가 보호자의 XXX라는 질문에 OOO라고 대답하였는데, 이는 동문서답이었습니다..."},
    "scoring2": {"score": 5, "reason": "정보 정확..."},
    "scoring3": {"score": 5, "reason": "정보 수집 충분..."},
    "scoring4": {"score": 7, "reason": "질문 품질 우수..."},
    "scoring5": {"score": 5, "reason": "원인 추론 정확..."},
    "scoring6": {"score": 2, "reason": "개인화 미흡..."},
    "scoring7": {"score": 7, "reason": "적절성 높음..."},
    "scoring8": {"score": 1, "reason": "구체성 부족..."},
    "scoring9": {"score": 0, "reason": "대상 내용 없음..."},
    "scoring10": {"score": 5, "reason": "어조/말투 문제 없음..."},
    "scoring11": {"score": 5, "reason": "장황/반복 적음..."}
}'

### 채점 기준: 아래 내용을 고려해 채점하고, 장단점을 분석하세요. 
- 항목별 점수 기준은 다음을 염두에 두고 작성되었습니다. 
    - 0~5점: 제기능을 충분히 수행하는가?
    - 5~10점: 제기능을 넘어 내담자에게 와우 포인트를 제공하는가?
- scoring_criteria[score:10] 컬럼은 10점으로 채점하는 기준을 나타냅니다.
""" + self.eval_score_table.to_markdown()

        param = {
            'system_instruction': system_instruction,
            'temperature': 0.2,
            'structured_output': EvalOutput,
            "max_output_tokens": 8192
        }
        return {
            1: {
                **param,
                'model': 'gemini-2.5-pro'
        },
            2: {
                **param,
                'model': 'gpt-5',
                'reasoning_effort': 'high'
            }
        }

    def problem_organizer(self):
        system_instruction = f"""
### 역할
당신은 반려견 행동학 전문 수의사입니다.
반려견 양육상담 챗봇 까미의 상담 능력을 평가하고 있어요.
이미 다른 전문가가 채팅 기록에 기반해 상담 능력을 점수로 평가했습니다.
프롬프트와 함께 그 평가 결과를 전달할테니, 당신은 까미 상담의 주요 문제를 추리고(생각으로만 하고 출력은 하지 않습니다), 가장 중요한 문제를 선택해 제시하세요.
그리고 그 가장 중요한 문제를 해결하기 위한 개선 방향을 제시해 주세요.
"까미 챗봇 지향성" 내용을 준수하여 문제와 개선안을 제시하세요.

### 중요 원칙
- '평가-개선' 과정을 반복수행하여 프롬프트를 점진적으로 개선할 예정이니, 한 번에 크게 개선하려고 하지 마세요.

### 출력
- **가장 중요한 문제**: (대화 예시를 자세히 인용하며 설명하세요.)
- **개선안**: 그 문제를 해결하기 위해 상담 챗봇의 프롬프트를 개선할 방안(방향)을 제시하세요.
"""

        return {
            "system_instruction": system_instruction,
            "model": "gpt-5",
            "reasoning_effort": "high",
            "temperature": 0.3,
            "max_output_tokens": 8192
        }

    def prompt_improver(self):
        system_instruction = f"""
### 역할
당신은 반려견 행동학 전문 수의사이자 프롬프트 엔지니어입니다.
반려견 양육상담 챗봇 까미의 상담 능력을 평가 개선하고 있어요.
이미 다른 전문가가 까미 상담의 주요 문제를 추리고, 그 문제를 해결하기 위한 개선 방향을 제시했습니다. 
그 개선 방향과 까미 챗봇의 프롬프트를 전달할테니, 당신은 그 개선 방향을 구체적으로 프롬프트로 작성해 주세요.
"까미 챗봇 지향성" 내용을 준수하여 프롬프트를 개선하세요.

### 중요 원칙
- '평가-개선' 과정을 반복수행하여 프롬프트를 점진적으로 개선할 예정이니, 한 번에 크게 개선하려고 하지 마세요.
- 챗봇의 출력 양식은 다음과 같은 json 형식이며, 이는 절대 변경할 수 없습니다.
    - front_message: str = Field(description="Only output the message to be delivered to the conversation partner.")
    - back_thinking: str = Field(description="Briefly summarize the Chain-of-Thought process.")

### 출력
챗봇 까미에 그대로 입력될 system_instruction을 출력하세요.
- 오로지 system_instruction만 출력하고, 다른 말은 일절 포함하지 마세요.
- 개선된 부분만 출력하는 것이 아니라, 반드시 개선된 system_instruction의 전체 내용을 출력하세요.
"""

        return {
            "system_instruction": system_instruction,
            "model": "gemini-2.5-pro",
            "temperature": 0.5,
            "tool_params": {
                    "TavilySearch": {
                        "max_results": 10,
                        "topic": "general"
                    }
                },
            "max_output_tokens": 8192
        }