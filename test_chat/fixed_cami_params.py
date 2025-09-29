from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class CounselorOutput(BaseModel):
    front_message: str = Field(description="Only output the message to be delivered to the conversation partner.")
    back_thinking: str = Field(description="Briefly summarize the Chain-of-Thought process.")

class FixedCamiParams:
    def __init__(self):
        wdir = '/dog_counseling/test_chat'
        self.guide_prompt = open(f'{wdir}/guide_prompt.md', 'r').read()
        self.counselor_prompt = open(f'{wdir}/counselor_prompt.md', 'r').read()
        self.initial_message = open(f'{wdir}/initial_message.md', 'r').read()
        
        self.params = {
            'counseling_guide': self.counseling_guide(),
            'counselor': self.counselor(),
        }
        
    def counseling_guide(self):
        return {
            'system_instruction': self.guide_prompt,
            'model': 'gemini-2.5-pro',
            'temperature': 0.2
        }
        
    def counselor(self):
        return {
            'system_instruction': self.counselor_prompt,
            'model': 'gemini-2.5-flash',
            'temperature': 0.6,
            'structured_output': CounselorOutput,
        }