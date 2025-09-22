from langchain_core.pydantic_v1 import BaseModel, Field

# --- Judge 모델의 Structured Output : Pydantic 모델 정의 ---
class EvalOutput(BaseModel):
    winner: str = Field(description="post와 pre 챗봇 중 누가 더 양질의 상담을 진행하였는지 판단하세요.")
    good_point: str = Field(description="post가 pre 대비 개선된 점을 설명하세요.")
    revision_point: str = Field(description="post를 개선하기 위해 수정해야 할 점을 설명하세요.")
    suggestion: str = Field(description="post를 개선하기 위해 제안하는 방법을 설명하세요.(개선의 범위는 프롬프트에 한정됩니다.)")
    score: int = Field(description="점수 평정")
    
class SetParams:
    def __init__(self, 
            cami_post_prompt,
            cami_pre_prompt,
            tester_persona):
        self.cami_post_prompt = cami_post_prompt
        self.cami_pre_prompt = cami_pre_prompt
        self.tester_persona = tester_persona
        self.params = {
            'counselor': self.counselor(),
            'tester': self.tester(),
            'judge1': self.judges()[1],
            'judge2': self.judges()[2],
        }
        
    def counselor(self):
        return {
            "system_instruction": self.cami_post_prompt,
            "model": "gemini-2.5-flash",    
            "temperature": 0.6
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
            "model": "gemini-2.5-flash",
            "temperature": 0.5
        }
    
    def judges(self):
        system_instruction = """
"""
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