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
    winner: Literal["pre", "post"]

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

    good_point: str = Field(description="[post]가 [pre] 대비 잘한 점")
    bad_point: str = Field(description="[post]가 [pre] 대비 못한 점")
    
class SetParams:
    def __init__(self, 
            cami_post_prompt,
            cami_pre_prompt,
            counseling_guide,
            tester_persona,
            eval_score_table):
        self.cami_post_prompt = cami_post_prompt
        self.cami_pre_prompt = cami_pre_prompt
        self.counseling_guide = counseling_guide
        self.tester_persona = tester_persona
        self.eval_score_table = eval_score_table
        self.params = {
            'counselor': self.counselor(),
            'tester': self.tester(),
            'judge1': self.judges()[1],
            'judge2': self.judges()[2],
        }
        
    def counselor(self):
        return {
            "system_instruction": self.cami_post_prompt + '\n\n --- \n\n' + self.counseling_guide,
            "model": "gemini-2.5-flash",    
            "temperature": 0.6,
            "structured_output": CounselorOutput,
            "max_output_tokens": 65536
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
- 이전 대화 내용을 똑같이 반복하는 것을 반드시 피하고, 까미 챗봇처럼 말하지 마세요.

### 출력 양식
- [챗봇 이탈] 혹은 당신이 까미 챗봇에 전할 말을 출력합니다.

{self.tester_persona}
"""
        return {
            "system_instruction": system_instruction,
            "model": "gpt-4o",
            "temperature": 0.3
        }
    
    def judges(self):
        system_instruction = """
### 역할
당신은 반려견 행동학 전문 수의사입니다.
반려견 양육상담 챗봇 까미의 상담 능력을 평가하고 있어요.
[pre]와 [post]는 까미의 프롬프트 버전에 따른 구분이며, pre는 기존 버전이고 post는 새로운 버전입니다.
[pre]와 [post] 버전의 채팅 기록을 보고 까미의 상담 능력을 비교 평가해 주세요.
또한 **채점 기준**을 제공할테니, 해당 기준에 따라 **[post] 버전의 점수**를 채점하세요.
**각 항목별 점수 기준은 상담이 지향하는 바를 나타냅니다. 점수 평가와 개선 제안을 반드시 이에 기반하여 제시해 주세요.**

### 출력
- winner: 더 양질의 상담을 진행한 버전을 선택하세요. [pre|post]
- scoring1..scoring11: 아래 [채점 기준]에 따라 **[post]** 버전의 점수를 평정하세요.
    - 각 항목은 {"score": 0~10, "reason": string} 객체입니다.
    - 채점할만한 대화 내용이 없으면 항목별로 'when_missing_score' 컬럼의 점수값을 score로 기입하고, reason은 "대상 내용 없음"으로 기입하세요.
- good_point: [post]가 [pre] 대비 잘한 점을 설명하세요.
- bad_point: [post]가 [pre] 대비 못한 점을 설명하세요.
- json 형식으로 출력. "```json"는 출력에 절대 포함하지 마세요.

#### 출력 예시
'{
    "winner": "post",
    "scoring1": {"score": 5, "reason": "요구 파악 양호..."},
    "scoring2": {"score": 5, "reason": "정보 정확..."},
    "scoring3": {"score": 5, "reason": "정보 수집 충분..."},
    "scoring4": {"score": 7, "reason": "질문 품질 우수..."},
    "scoring5": {"score": 5, "reason": "원인 추론 정확..."},
    "scoring6": {"score": 2, "reason": "개인화 미흡..."},
    "scoring7": {"score": 7, "reason": "적절성 높음..."},
    "scoring8": {"score": 1, "reason": "구체성 부족..."},
    "scoring9": {"score": 0, "reason": "대상 내용 없음..."},
    "scoring10": {"score": 5, "reason": "어조/말투 문제 없음..."},
    "scoring11": {"score": 5, "reason": "장황/반복 적음..."},
    "good_point": "...",
    "bad_point": "..."
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
            'structured_output': EvalOutput
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