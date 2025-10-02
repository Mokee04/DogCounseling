### GUIDING_EXPERT : 상담 가이드 작성용 모델
### 입력 -> pre-questionnaire_summary.md 파일에 있는 내용
### 출력 -> 플레인 텍스트(상담 진행용 모델에 전달할 상담 가이드 문서)
guiding_expert = {
    "model": 'gemini-2.5-pro',
    "system_instruction": GUIDING_EXPERT_SYSTEM_INSTRUCTION, # guiding_expert_system_instruction.md
    "temperature": 0.2,
    "max_output_tokens": 16384
}

### COUNSELOR : 상담 진행용 모델
### System Instruction : COUNSELOR_PROMPT + '\n\n' + GUIDING_EXPERT_OUTPUT
###     - GUIDER_EXPERT 모델이 출력한 문자열 데이터(가이드 문서)를 COUNSELOR_PROMPT와 결합하여 System Instruction으로 입력합니다.
### 초기 입력 : 채팅방 개설 시, counselor_initial_input_message.md 파일에 있는 내용을 모델에 보내고, 그 첫번째 응답을 채팅방에 표시합니다.
### 출력 -> JSON {front_message : 모델이 유저에게 전달하는 말, back_thinking : 내부 사고 과정}
COUNSELOR_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "front_message": {
            "type": "string",
            "description": "Only output the message to be delivered to the conversation partner."
        },
        "back_thinking": {
            "type": "string",
            "description": "Briefly summarize the Chain-of-Thought process."
        }
    },
    "required": ["front_message", "back_thinking"]
}
counselor = {
    "model": "gemini-2.5-flash",
    "system_instruction": COUNSELOR_SYSTEM_INSTRUCTION + '\n\n' + GUIDING_EXPERT_OUTPUT, # counselor_system_instruction.md + 한줄띄기 + guiding expert 모델 출력 결과
    "temperature": 0.6,
    "response_mime_type": "application/json",
    "response_schema": COUNSELOR_OUTPUT_SCHEMA,
    "max_output_tokens": 16384
}