import sys, os, warnings, gc, datetime, json, pickle
import pandas as pd, numpy as np
from pydantic import TypeAdapter
from google import genai
from google.genai import types
import google.generativeai as genai_simp
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload, MediaIoBaseDownload
from io import BytesIO

# 거슬리는 경고 메시지 안 보이게 처리
warnings.simplefilter('ignore')

class LoadPrequestions:
    """구글 스프레드시트에서 로그인 인증만 담당하는 클래스"""
    
    def __init__(self, login_info):
        """로그인 하고 구글 시트에서 사용자 데이터 확인"""
        # 구글 API 연결
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = st.secrets["connections"]["gcs"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(f'https://docs.google.com/spreadsheets/d/{st.secrets["prequestions_id"]}')
        worksheet = sheet.worksheet('응답')
        
        # 스프레드시트 데이터 변환
        data = worksheet.get_all_records()
        raw = pd.DataFrame(data)
        raw = raw.rename(columns=raw.loc[0].to_dict()).loc[1:].reset_index(drop=True)
        
        # 로그인 검증
        id, password = login_info['id'], login_info['password']
        user_data = raw[(raw['email'] == id) & (raw['password'] == password)]
        
        # 없는 계정이면 에러
        if len(user_data) == 0:
            raise ValueError("존재하지 않는 아이디 또는 비밀번호입니다.")
        
        # 인스턴스 변수로 저장
        self.pet_name = user_data['pet_name'].values[0]
        self.user_context = user_data['user_context'].values[0]
        
    def get_user_context(self):
        """사용자 컨텍스트 반환"""
        return self.user_context

def upload_to_drive(file_data, filename, folder_id, mime_type, credentials):
    """
    Google Drive에 파일 업로드하는 함수 - 같은 이름 파일은 덮어쓰기
    
    file_data: 업로드할 파일 데이터 (문자열 또는 바이트)
    filename: 저장할 파일 이름
    folder_id: 저장할 Google Drive 폴더 ID
    mime_type: 파일 형식 (application/json 등)
    credentials: 서비스 계정 인증 정보
    
    업로드된 파일의 ID 반환
    """
    # Drive API 서비스 생성
    drive_service = build('drive', 'v3', credentials=credentials)
    
    # 같은 이름의 파일이 폴더에 있는지 검색
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    results = drive_service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    
    # 메모리에서 파일 데이터 처리
    if isinstance(file_data, str):
        file_data = file_data.encode('utf-8')
    
    # 메모리 기반 미디어 객체 생성
    fh = BytesIO(file_data)
    media = MediaIoBaseUpload(fh, mimetype=mime_type, resumable=True)
    
    # 기존 파일이 있으면 업데이트, 없으면 새로 만들기
    files = results.get('files', [])
    if files:
        # 기존 파일 업데이트
        file_id = files[0]['id']
        drive_service.files().update(
            fileId=file_id,
            media_body=media
        ).execute()
        print(f"기존 파일 업데이트")
        return file_id
    else:
        # 새 파일 생성
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f"새 파일 생성")
        return file.get('id')

def download_instruction_file(credentials):
    """
    구글 드라이브에서 system_instruction 관련 파일 다운로드
    
    credentials: 서비스 계정 인증 정보
    
    system_instruction, model_name, initial_message가 포함된 딕셔너리 반환
    """
    try:
        # Drive API 서비스 생성
        drive_service = build('drive', 'v3', credentials=credentials)
        
        # 폴더 내 system_instruction.json 파일 검색 (output_drive_id 사용)
        folder_id = st.secrets['drive_folder_id']
        query = f"name='system_instruction.json' and '{folder_id}' in parents and trashed=false"
        results = drive_service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        files = results.get('files', [])
        if not files:
            print("system_instruction 파일을 찾을 수 없습니다.")
            return {
                "system_instruction": "오류가 났어요 ㅜㅜ.",
                "model_name": "models/gemini-2.0-flash",
                "initial_message": "오류로 인한 상담 불가를 안내해 주세요."
            }
        
        # 파일 다운로드
        file_id = files[0]['id']
        request = drive_service.files().get_media(fileId=file_id)
        file_bytes = BytesIO()
        downloader = MediaIoBaseDownload(file_bytes, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            
        # 파일 내용 파싱
        file_bytes.seek(0)
        instruction_data = json.loads(file_bytes.getvalue().decode('utf-8'))
        
        print("system_instruction 로드 완료")
        return instruction_data
        
    except Exception as e:
        print(f"시스템 지시문 파일 로드 실패: {e}")
        # 기본값 제공
        return {
                "system_instruction": "오류가 났어요 ㅜㅜ.",
                "model_name": "models/gemini-2.0-flash",
                "initial_message": "오류로 인한 상담 불가를 안내해 주세요."
            }

class CounselingWithGemini:
    """Gemini로 반려견 상담하는 클래스 - 챗봇 엔진"""
    
    def __init__(self, user_context, google_ai_studio_key):
        """
        챗봇 모델 초기화하기
        """
        self.google_ai_studio_key = google_ai_studio_key
        self.client = None  # API 클라이언트
        self.chatmodel = None  # 채팅 모델
        
        # 인증 정보 가져오기
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = st.secrets["connections"]["gcs"]
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        
        # 구글 드라이브에서 시스템 지시문 파일 다운로드
        self.instruction_data = download_instruction_file(credentials)
        system_instruction = self.instruction_data.get('system_instruction', "반려견 양육 상담을 진행합니다.")
        
        # 시스템 지시문과 사용자 컨텍스트 결합
        self.sys_inst = f"{system_instruction}\n\n{user_context}"
        
    def define_model(self, model_name=None):
        """
        Gemini 모델 설정하고 초기화하기
        
        model_name: 사용할 Gemini 모델
        """
        # 모델명이 파라미터로 넘어오지 않았다면 드라이브에서 가져온 값 사용
        if model_name is None:
            model_name = self.instruction_data.get('model_name', "models/gemini-2.0-flash-thinking-exp-01-21")
        
        # 안전 설정 - 위험한 내용 블록
        safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]

        # 모델 응답 설정
        generation_config = types.GenerateContentConfig(
            system_instruction=self.sys_inst,  # 시스템 지시문 (상담사 역할 정의)
            response_mime_type="text/plain",   # 텍스트 응답
            temperature=0.7,                   # 응답 다양성
            top_p=0.95,                        # 토큰 선택 범위
            max_output_tokens=2048,            # 최대 토큰 수
            safety_settings=safety_settings     # 안전 설정
        )
        
        # Gemini API 클라이언트 생성하고 채팅 모델 초기화
        self.client = genai.Client(api_key=self.google_ai_studio_key)
        self.chatmodel = self.client.chats.create(
            model=model_name,
            config=generation_config
        )
    
    def send_question(self, question):
        """
        질문을 모델에 보내고 응답 받기
        
        question: 사용자가 한 질문
        
        응답 텍스트를 반환하고 콘솔에도 출력 (디버깅용)
        """
        # 질문 보내고 응답 받기
        response = self.chatmodel.send_message(question).text
        
        # 콘솔에 질문-응답 로그 출력 (디버깅용)
        display_text = f"""
Q. {question}
A. {response}"""
        print(display_text)
        
        return response
    
    def get_chat_info(self):
        """
        현재 채팅 모델 설정 정보 반환
        
        모델명과 설정값을 딕셔너리로 반환 (디버깅용)
        """
        if self.chatmodel:
            return {
                "model": self.chatmodel._model,     # 모델명
                "config": self.chatmodel._config,   # 설정값
            }
        return None
    
    def save_chatmodel(self, folder_id, user_id=None, file_id=None, credentials=None):
        """
        채팅 모델 상태를 Google Drive에 저장
        
        folder_id: 저장할 Google Drive 폴더 ID
        user_id: 사용자 ID (파일명 생성에 사용)
        file_id: 기존 파일 ID (있으면 그 파일 업데이트)
        credentials: 인증 정보 (없을 경우 기본 인증 정보 사용)
        
        저장된 파일 ID 반환, 실패 시 None
        """
        if self.chatmodel:
            try:
                # 현재 상태를 딕셔너리로 저장
                state = {
                    'history': self.chatmodel.get_history(),
                    'model_name': self.chatmodel._model,
                    'config': self.chatmodel._config,
                    'google_ai_studio_key': self.google_ai_studio_key,
                    'sys_inst': self.sys_inst
                }
                
                # 파일명 생성 - 사용자 ID 기반으로 변경
                if user_id:
                    # 사용자 ID로 안전한 파일명 생성
                    safe_id = "".join(c if c.isalnum() else "_" for c in user_id)
                    filename = f'{safe_id}_model.pkl'
                else:
                    # 기존 방식 유지 (사용자 ID가 없을 경우 날짜 사용)
                    today = datetime.date.today()
                    filename = f'chatmodel_state_{today.strftime("%Y%m%d")}.pkl'
                
                # 피클 데이터를 메모리에 직렬화
                pickle_data = BytesIO()
                pickle.dump(state, pickle_data)
                pickle_data.seek(0)  # 처음으로 되돌리기
                
                # 인증 정보가 없으면 기본 인증 정보 사용
                if credentials is None:
                    # 구글 API 연결 설정
                    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
                    creds_dict = st.secrets["connections"]["gcs"]
                    credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
                
                # Drive에 업로드 - 기존 파일 ID가 있으면 먼저 직접 업데이트 시도
                if file_id:
                    try:
                        drive_service = build('drive', 'v3', credentials=credentials)
                        drive_service.files().update(
                            fileId=file_id,
                            media_body=MediaIoBaseUpload(pickle_data, mimetype='application/octet-stream')
                        ).execute()
                        print(f"기존 파일 업데이트 (직접)")
                        return file_id
                    except Exception as e:
                        print(f"기존 파일 업데이트 실패, 일반 업로드로 진행: {e}")
                        # 실패하면 일반 업로드로 진행
                
                # Drive에 업로드 (기존 파일 ID 없거나 직접 업데이트 실패 시)
                file_id = upload_to_drive(
                    file_data=pickle_data.getvalue(),
                    filename=filename,
                    folder_id=folder_id,
                    mime_type='application/octet-stream',
                    credentials=credentials
                )
                
                print(f"채팅 모델 상태 Google Drive에 저장 완료")
                return file_id
                
            except Exception as e:
                print(f"Google Drive 저장 중 오류 발생: {e}")
                return None
        
        return None
    
    @classmethod
    def load_chatmodel_from_drive(cls, file_id, credentials, google_ai_studio_key=None):
        """
        Google Drive에서 채팅 모델 불러오기
        
        file_id: Google Drive 파일 ID
        credentials: 인증 정보
        google_ai_studio_key: API 키 (없으면 저장된 키 사용)
        """
        try:
            # Drive API 서비스 생성
            drive_service = build('drive', 'v3', credentials=credentials)
            
            # 파일 다운로드
            request = drive_service.files().get_media(fileId=file_id)
            file_bytes = BytesIO()
            downloader = MediaIoBaseDownload(file_bytes, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                
            # 파일 내용 가져오기
            file_bytes.seek(0)
            state = pickle.load(file_bytes)
            
            # 새 인스턴스 생성
            instance = cls("", state['google_ai_studio_key'] if google_ai_studio_key is None else google_ai_studio_key)
            instance.sys_inst = state['sys_inst']  # 시스템 지시문 복원
            
            # API 클라이언트 초기화
            instance.client = genai.Client(api_key=instance.google_ai_studio_key)
            
            # 채팅 모델 초기화
            instance.chatmodel = instance.client.chats.create(
                model=state['model_name'],
                config=state['config']
            )
            
            # 채팅 기록 복원
            for content in state['history']:
                if content.role == 'user':
                    instance.chatmodel.send_message(content.parts[0].text)
            
            print(f"Google Drive에서 채팅 모델 로드 완료")
            return instance
            
        except Exception as e:
            print(f"Google Drive에서 모델 로드 중 오류 발생: {e}")
            return None
