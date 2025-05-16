import streamlit as st
import os
import pandas as pd
import datetime
import json
import uuid
from counseling_lib import LoadPrequestions, CounselingWithGemini, upload_to_drive
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from io import BytesIO
import gspread

# 페이지 설정
st.set_page_config(
    page_title="반려견 상담 챗봇",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 환경 변수 로드
base_path = os.path.dirname(os.path.abspath(__file__))
google_ai_studio_key = st.secrets['GOOGLE_AI_STUDIO_KEY']
output_drive_id = st.secrets['output_drive_id']  # 모든 데이터를 저장할 폴더 ID

# Google Drive에서 파일 다운로드하는 함수
def download_from_drive(file_id, credentials):
    """
    Google Drive에서 파일 다운로드하는 함수
    
    file_id: 다운로드할 파일의 ID
    credentials: 서비스 계정 인증 정보
    
    파일 내용 바이트로 반환
    """
    # Drive API 서비스 생성
    drive_service = build('drive', 'v3', credentials=credentials)
    
    # 파일 다운로드
    request = drive_service.files().get_media(fileId=file_id)
    file_bytes = BytesIO()
    
    # 요청 실행 및 파일 내용 저장
    downloader = MediaIoBaseDownload(file_bytes, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        
    # 파일 내용 반환
    file_bytes.seek(0)
    return file_bytes.getvalue()

# 인증 정보 가져오기 함수
def get_credentials():
    """Streamlit secrets에서 구글 API 인증 정보 가져오기"""
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_dict = st.secrets["connections"]["gcs"]
    return ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

# 세션 상태 초기화
if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

if 'login_error' not in st.session_state:
    st.session_state['login_error'] = False

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'user_id' not in st.session_state:
    st.session_state['user_id'] = ""

# 피드백 상태 초기화
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = {}

# 사용자별 Google Drive 파일 ID를 저장/관리하기 위한 상태
if 'model_file_id' not in st.session_state:
    st.session_state['model_file_id'] = None

# 채팅 모델 파일 이름 생성 함수
def get_model_filename(user_id):
    """사용자 ID에 기반한 모델 파일명 생성"""
    safe_id = "".join(c if c.isalnum() else "_" for c in user_id)
    return f'{safe_id}_model.pkl'

# Google Drive에서 사용자 모델 파일 ID 찾기
def find_model_file_id(user_id, credentials):
    """Drive에서 사용자 모델 파일 ID 찾기"""
    filename = get_model_filename(user_id)
    
    try:
        # Drive API 서비스 생성
        drive_service = build('drive', 'v3', credentials=credentials)
        
        # 폴더 내 파일 검색 - output_drive_id 사용
        query = f"name='{filename}' and '{output_drive_id}' in parents and trashed=false"
        results = drive_service.files().list(
            q=query, 
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        files = results.get('files', [])
        if files:
            return files[0]['id']  # 첫 번째 매칭 파일의 ID 반환
        return None
    
    except Exception as e:
        print(f"모델 파일 ID 찾기 중 오류 발생: {e}")
        return None

# 채팅 및 피드백 데이터 파일명 생성
def get_chat_data_filename(user_id):
    """사용자 ID에 기반한 채팅 데이터 파일명 생성"""
    safe_id = "".join(c if c.isalnum() else "_" for c in user_id)
    return f'{safe_id}_chat_data.json'

# Google Drive에서 사용자 채팅 데이터 파일 ID 찾기
def find_chat_data_file_id(user_id, credentials):
    """Drive에서 사용자 채팅 데이터 파일 ID 찾기"""
    filename = get_chat_data_filename(user_id)
    
    try:
        # Drive API 서비스 생성
        drive_service = build('drive', 'v3', credentials=credentials)
        
        # 폴더 내 파일 검색 (output_drive_id 사용)
        query = f"name='{filename}' and '{output_drive_id}' in parents and trashed=false"
        results = drive_service.files().list(
            q=query, 
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        files = results.get('files', [])
        if files:
            return files[0]['id']  # 첫 번째 매칭 파일의 ID 반환
        return None
    
    except Exception as e:
        print(f"채팅 데이터 파일 ID 찾기 중 오류 발생: {e}")
        return None

# 채팅 및 피드백 데이터 저장 함수
def save_chat_data(user_id):
    """채팅 기록과 피드백을 함께 저장"""
    chat_data = {
        "chat_history": st.session_state['chat_history'],
        "feedback": st.session_state['feedback']
    }
    
    try:
        # 인증 정보 가져오기
        credentials = get_credentials()
        
        # JSON 데이터 직렬화
        json_data = json.dumps(chat_data, ensure_ascii=False)
        
        # 파일 이름 생성
        filename = get_chat_data_filename(user_id)
        
        # Drive에 업로드 (output_drive_id 사용)
        file_id = upload_to_drive(
            file_data=json_data,
            filename=filename,
            folder_id=output_drive_id,
            mime_type='application/json',
            credentials=credentials
        )
        
        print(f"채팅 데이터 Google Drive에 저장 완료")
        return True
    except Exception as e:
        print(f"채팅 데이터 저장 중 오류 발생: {e}")
        return False

# 채팅 및 피드백 데이터 로드 함수
def load_chat_data(user_id):
    """저장된 채팅 기록과 피드백을 로드하고 필요시 마이그레이션"""
    try:
        # 인증 정보 가져오기
        credentials = get_credentials()
        
        # 파일 ID 찾기
        file_id = find_chat_data_file_id(user_id, credentials)
        
        if not file_id:
            return None
        
        # 파일 다운로드
        file_content = download_from_drive(file_id, credentials)
        chat_data = json.loads(file_content.decode('utf-8'))
        
        # ID가 없는 메시지에 ID 추가 (마이그레이션)
        chat_history = chat_data.get('chat_history', [])
        for idx, message in enumerate(chat_history):
            if "id" not in message or message.get("id") is None:
                message["id"] = f"migrated_{idx}_{str(uuid.uuid4())}"
                
        # 피드백 마이그레이션 (인덱스 기반 -> ID 기반)
        feedback = chat_data.get('feedback', {})
        new_feedback = {}
        
        # 인덱스 기반 피드백 처리
        for key, value in list(feedback.items()):
            # 숫자 인덱스인지 확인
            if isinstance(key, str) and key.isdigit():
                idx = int(key)
                if idx < len(chat_history) and chat_history[idx]["role"] == "assistant":
                    message_id = chat_history[idx]["id"]
                    new_feedback[message_id] = value
                    # 기존 키 삭제
                    feedback.pop(key, None)
            
        # 남은 피드백 정보 유지
        new_feedback.update(feedback)
        chat_data['feedback'] = new_feedback
        
        return chat_data
    except Exception as e:
        print(f"채팅 데이터 로드 중 오류 발생: {e}")
        return None

# 로그인 함수 재구현
def login():
    st.session_state['login_error'] = False
    user_id = st.session_state.user_id_input
    user_password = st.session_state.user_password
    
    try:
        # 로그인 정보로 LoadPrequestions 클래스 초기화
        login_info = {'id': user_id, 'password': user_password}
        process = LoadPrequestions(login_info=login_info)
        
        # 사용자 정보 저장
        st.session_state['user_id'] = user_id
        st.session_state['user_info'] = process.pet_name
        
        # 사용자 컨텍스트 가져오기
        user_context = process.get_user_context()
        
        # 인증 정보 가져오기
        credentials = get_credentials()
        
        # Google Drive에서 사용자 모델 파일 ID 찾기
        model_file_id = find_model_file_id(user_id, credentials)
        st.session_state['model_file_id'] = model_file_id
        
        if model_file_id:
            # 저장된 모델이 있으면 Drive에서 불러오기
            counselor = CounselingWithGemini.load_chatmodel_from_drive(
                file_id=model_file_id, 
                credentials=credentials,
                google_ai_studio_key=google_ai_studio_key
            )
            
            if counselor is None:
                # 모델 로드 실패 시 새 모델 생성
                st.warning("이전 상담 모델을 불러오는 데 실패했습니다. 새 모델을 생성합니다.")
                st.session_state['counselor'] = CounselingWithGemini(
                    user_context=user_context, 
                    google_ai_studio_key=google_ai_studio_key
                )
                st.session_state['counselor'].define_model()
                st.session_state['loaded_existing_model'] = False
            else:
                st.session_state['counselor'] = counselor
                st.session_state['loaded_existing_model'] = True
            
            # 저장된 채팅 데이터가 있는지 확인
            chat_data = load_chat_data(user_id)
            
            if chat_data:
                # 저장된 채팅 데이터가 있으면 로드
                st.session_state['chat_history'] = chat_data.get('chat_history', [])
                st.session_state['feedback'] = chat_data.get('feedback', {})
            else:
                # 저장된 채팅 데이터가 없으면 모델에서 대화 기록만 가져오기
                st.session_state['chat_history'] = []
                
                if st.session_state['counselor'].chatmodel:
                    history = st.session_state['counselor'].chatmodel.get_history()
                    for content in history:
                        try:
                            if len(content.parts) > 0 and hasattr(content.parts[0], 'text'):
                                role = "assistant" if content.role == "model" else "user"
                                # 고유 ID 생성
                                message_id = str(uuid.uuid4())
                                st.session_state['chat_history'].append({
                                    "id": message_id,
                                    "role": role,
                                    "content": content.parts[0].text
                                })
                        except (IndexError, AttributeError) as e:
                            print(f"메시지 처리 중 오류 발생: {e}")
                            continue
                
                st.session_state['feedback'] = {}
            
        else:
            # 저장된 모델이 없으면 새로 생성
            st.session_state['counselor'] = CounselingWithGemini(
                user_context=user_context, 
                google_ai_studio_key=google_ai_studio_key
            )
            st.session_state['counselor'].define_model()
            
            # 초기 메시지는 구글 드라이브에서 가져온 값 사용하여 모델에 전송
            initial_message = st.session_state['counselor'].instruction_data.get(
                'initial_message',
                "오류로 상담 진행이 불가함을 안내해 주세요."
            )
            
            # 초기 메시지를 모델에 보내고 응답 받기 (JSON 응답 처리)
            try:
                json_response_str = st.session_state['counselor'].send_question(initial_message)
                response_data = json.loads(json_response_str) # JSON 파싱
                bot_response_content = response_data.get("front_message", "오류: 응답 내용을 가져올 수 없습니다.") # front_message 추출

                # 백엔드 데이터 추출 (필요시)
                backend_data = {
                    key: response_data.get(key)
                    for key in response_data.keys()
                    if key.startswith('back_')
                }

            except json.JSONDecodeError:
                st.error("모델 응답을 처리하는 중 오류가 발생했습니다. (JSON 형식 오류)")
                bot_response_content = "죄송합니다. 답변을 처리하는 데 문제가 발생했습니다."
                backend_data = {} # 오류 시 빈 데이터
            except Exception as e:
                st.error(f"모델 응답 처리 중 예상치 못한 오류 발생: {e}")
                bot_response_content = "죄송합니다. 답변을 처리하는 데 문제가 발생했습니다."
                backend_data = {} # 오류 시 빈 데이터

            message_id = str(uuid.uuid4())
            st.session_state['chat_history'] = [{
                "id": message_id,
                "role": "assistant",
                "content": bot_response_content, # front_message만 content로 저장
                "backend_data": backend_data # 백엔드 데이터 추가 저장
            }]
            st.session_state['feedback'] = {}
            st.session_state['loaded_existing_model'] = False
        
        # 로그인 상태 변경
        st.session_state['is_logged_in'] = True
        
    except ValueError as e:
        # 로그인 실패 시 오류 상태 설정
        st.session_state['login_error'] = True
        st.session_state['error_message'] = str(e)

# 피드백 함수 재구현
def handle_like(message_id):
    """추천 버튼 클릭 처리"""
    if st.session_state['feedback'].get(message_id) == 'like':
        # 이미 좋아요 상태면 취소
        st.session_state['feedback'].pop(message_id)
    else:
        # 좋아요 설정
        st.session_state['feedback'][message_id] = 'like'
    # 피드백 즉시 저장
    if st.session_state['user_id']:
        save_chat_data(st.session_state['user_id'])

def handle_dislike(message_id):
    """반대 버튼 클릭 처리"""
    if st.session_state['feedback'].get(message_id) == 'dislike':
        # 이미 싫어요 상태면 취소
        st.session_state['feedback'].pop(message_id)
    else:
        # 싫어요 설정
        st.session_state['feedback'][message_id] = 'dislike'
    # 피드백 즉시 저장
    if st.session_state['user_id']:
        save_chat_data(st.session_state['user_id'])

# 피드백 및 새로고침 처리 함수 추가
def handle_refresh(message_id):
    """새로고침 버튼 클릭 처리"""
    try:
        # 수정할 메시지 인덱스 찾기
        target_index = -1
        for i, msg in enumerate(st.session_state['chat_history']):
            if msg.get("id") == message_id and msg.get("role") == "assistant":
                target_index = i
                break

        if target_index == -1 or target_index == 0:
            st.warning("이전 사용자 질문을 찾을 수 없어 답변을 다시 생성할 수 없습니다.")
            return

        # 이전 사용자 질문 찾기
        user_message = st.session_state['chat_history'][target_index - 1]
        if user_message.get("role") != "user":
            st.warning("이전 메시지가 사용자 질문이 아니므로 답변을 다시 생성할 수 없습니다.")
            return

        user_query = user_message["content"]

        # 봇 응답 재생성
        with st.spinner("답변 다시 생성 중..."):
            try:
                json_response_str = st.session_state['counselor'].send_question("오류가 발생했어요. 다시 생성해주세요:  \n" + user_query)
                response_data = json.loads(json_response_str) # JSON 파싱
                new_response_content = response_data.get("front_message", "오류: 응답 내용을 가져올 수 없습니다.") # front_message 추출

                # 백엔드 데이터 추출 (필요시)
                new_backend_data = {
                    key: response_data.get(key)
                    for key in response_data.keys()
                    if key.startswith('back_')
                }

                # 해당 메시지 내용 및 백엔드 데이터 업데이트
                st.session_state['chat_history'][target_index]["content"] = new_response_content
                st.session_state['chat_history'][target_index]["backend_data"] = new_backend_data # 백엔드 데이터도 업데이트
                # 기존 피드백 제거
                st.session_state['feedback'].pop(message_id, None)

                # 변경사항 저장 (채팅 데이터 및 모델)
                if st.session_state['user_id']:
                    chat_data_saved = save_chat_data(st.session_state['user_id'])

                    # 모델 저장 로직 추가 (CounselingWithGemini.save_chatmodel 사용)
                    credentials = get_credentials()
                    model_saved = False
                    if st.session_state.get('model_file_id'):
                        # 기존 파일 업데이트 시도
                        file_id_or_success = st.session_state['counselor'].save_chatmodel(
                            folder_id=output_drive_id,
                            user_id=st.session_state['user_id'],
                            file_id=st.session_state['model_file_id'],
                            credentials=credentials
                        )
                        # save_chatmodel은 성공 시 file_id를 반환하므로 True/False로 변환
                        model_saved = bool(file_id_or_success)
                    else:
                        # 새 파일 생성 시도
                        file_id = st.session_state['counselor'].save_chatmodel(
                            folder_id=output_drive_id,
                            user_id=st.session_state['user_id'],
                            credentials=credentials
                        )
                        if file_id: # 새 파일 ID가 반환되면 성공
                             st.session_state['model_file_id'] = file_id
                             model_saved = True
                        else: # None이 반환되면 실패
                             model_saved = False

                    # 저장 결과 피드백
                    if chat_data_saved and model_saved:
                        # 새로고침 시에는 별도 토스트 메시지 생략 가능 (UI가 바로 업데이트되므로)
                        pass
                    elif model_saved:
                        st.warning("상담 내용은 저장되었으나 채팅 데이터 저장에 실패했습니다.")
                    elif chat_data_saved:
                         st.warning("채팅 데이터는 저장되었으나 상담 내용 저장에 실패했습니다.")
                    else:
                         st.error("데이터 저장 중 오류가 발생했습니다.")

            except json.JSONDecodeError:
                # 실패한 원본 문자열 로깅 추가
                print(f"JSON 파싱 실패. 원본 응답: {json_response_str}")
                st.error("모델 응답을 처리하는 중 오류가 발생했습니다. (JSON 형식 오류)")
            except Exception as e:
                st.error(f"답변 재생성 중 오류 발생: {e}")

        # UI 갱신
        st.rerun()

    except Exception as e:
        st.error(f"답변 재생성 처리 중 오류 발생: {e}") # 함수 전체 감싸는 에러 핸들링

# 로그아웃 함수 재구현
def logout():
    # 로그아웃 시 현재 모델과 채팅 데이터 저장
    if 'counselor' in st.session_state and st.session_state['user_id']:
        # 모델 저장 - 이제 Drive 폴더 ID 사용
        credentials = get_credentials()
        model_saved = False  # 변수 초기화
        
        if st.session_state.get('model_file_id'):
            # 기존 파일 업데이트
            model_saved = st.session_state['counselor'].save_chatmodel(
                folder_id=output_drive_id,
                user_id=st.session_state['user_id'],
                file_id=st.session_state['model_file_id'],
                credentials=credentials
            )
        else:
            # 새 파일 생성
            file_id = st.session_state['counselor'].save_chatmodel(
                folder_id=output_drive_id,
                user_id=st.session_state['user_id'],
                credentials=credentials
            )
            st.session_state['model_file_id'] = file_id  # 새 ID 저장
            model_saved = bool(file_id)  # 저장 성공 여부
        
        chat_data_saved = save_chat_data(st.session_state['user_id'])
        
        if model_saved and chat_data_saved:
            st.toast("상담 내용과 피드백이 저장되었습니다.")
        elif model_saved:
            st.warning("상담 내용은 저장되었으나 채팅 데이터 저장에 실패했습니다.")
        else:
            st.error("저장 중 오류가 발생했습니다.")
    
    # 세션 상태 초기화
    keys_to_clear = ['is_logged_in', 'counselor', 'chat_history', 'user_info', 'loaded_existing_model', 
                     'feedback', 'model_file_id']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # user_id는 초기화하되 삭제하지 않음
    st.session_state['user_id'] = ""
    st.session_state['is_logged_in'] = False
    st.session_state['chat_history'] = []
    st.session_state['feedback'] = {}
    
# 메인 UI 구성
st.title("🐶반려견 상담")

# 로그인 상태에 따라 다른 UI 표시
if not st.session_state['is_logged_in']:
    # 로그인 폼 표시
    st.subheader("로그인")
    
    with st.form("login_form"):
        st.text_input("이메일", key="user_id_input", placeholder="이메일을 입력하세요")
        st.text_input("비밀번호", key="user_password", type="password", placeholder="비밀번호를 입력하세요")
        submit = st.form_submit_button("로그인", on_click=login)
    
    # 로그인 오류 메시지 표시
    if st.session_state['login_error']:
        st.error(f"로그인 실패: {st.session_state['error_message']}")
        
else:
    # 로그인 성공 시 상담 인터페이스 표시

    # 사이드바에 사용자 정보 및 로그아웃 버튼
    with st.sidebar:
        st.write(f"반려견: {st.session_state['user_info']}")

        # 이전 대화 모델을 불러왔을 경우 표시
        if st.session_state.get('loaded_existing_model', False):
            st.success("이전 상담 내용을 불러왔습니다.")

        # 모델 저장 버튼 제거됨

        # 로그아웃 버튼
        st.button("로그아웃", on_click=logout)

    # 채팅 메시지 표시
    # 메시지 렌더링 함수 정의
    def render_message(message):
        """
        채팅 메시지와 피드백/새로고침 버튼을 함께 렌더링하는 함수
        버튼은 메시지 내용 하단 왼쪽에 표시. 마지막 어시스턴트 메시지에만 새로고침 버튼 표시.
        버튼 순서: 좋아요 - 싫어요 - 새로고침
        """
        if message["role"] == "assistant":
            # ID 확인/생성
            if "id" not in message or message.get("id") is None:
                message["id"] = str(uuid.uuid4())

            message_id = message["id"]
            current_feedback = st.session_state['feedback'].get(message_id, None)

            # 버튼 스타일 설정
            like_style = "primary" if current_feedback == 'like' else "secondary"
            dislike_style = "primary" if current_feedback == 'dislike' else "secondary"

            # 마지막 메시지인지 확인
            is_last_assistant_message = (
                st.session_state['chat_history'] and
                message == st.session_state['chat_history'][-1]
            )

            # 메시지 표시 컨테이너
            with st.chat_message(message["role"], avatar="🐶"):
                # 1. 메시지 내용 먼저 표시
                content_with_breaks = message["content"].replace("\\n", "\n")
                st.markdown(content_with_breaks, unsafe_allow_html=False)

                # 2. 버튼 영역 (내용 아래) - st.columns를 사용하여 왼쪽 정렬
                # 내용과의 간격 조절을 위해 작은 공백 추가
                st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                if is_last_assistant_message:
                    # 마지막 메시지: 좋아요 - 싫어요 - 새로고침 순서
                    # 버튼 컬럼 뒤에 넓은 빈 컬럼을 두어 왼쪽 정렬 효과
                    col1, col2, col3, _ = st.columns([0.05, 0.05, 0.05, 0.85])
                    with col1: # 좋아요
                        st.button("👍", key=f"like_{message_id}",
                                on_click=handle_like, args=(message_id,),
                                type=like_style)
                    with col2: # 싫어요
                        st.button("👎", key=f"dislike_{message_id}",
                                on_click=handle_dislike, args=(message_id,),
                                type=dislike_style)
                    with col3: # 새로고침
                         st.button("🔄", key=f"refresh_{message_id}",
                                on_click=handle_refresh, args=(message_id,),
                                help="답변 다시 생성하기")
                else:
                    # 이전 메시지: 좋아요 - 싫어요 순서
                    # 버튼 컬럼 뒤에 넓은 빈 컬럼을 두어 왼쪽 정렬 효과
                    col1, col2, _ = st.columns([0.05, 0.05, 0.9])
                    with col1: # 좋아요
                         st.button("👍", key=f"like_{message_id}",
                                on_click=handle_like, args=(message_id,),
                                type=like_style)
                    with col2: # 싫어요
                        st.button("👎", key=f"dislike_{message_id}",
                                on_click=handle_dislike, args=(message_id,),
                                type=dislike_style)

        else:
            # 사용자 메시지
            with st.chat_message(message["role"], avatar="😀"):
                # 사용자 메시지도 동일하게 처리 (만약을 위해)
                content_with_breaks = message["content"].replace("\\n", "  \n")
                st.markdown(content_with_breaks, unsafe_allow_html=False)

    # 모든 메시지를 하나의 명령으로 렌더링
    with st.container():
        # 한번에 모든 채팅 메시지를 처리하여 UI 안정성 향상
        for message in st.session_state['chat_history']:
            render_message(message)

    # 사용자 입력
    user_input = st.chat_input("반려견에 대해 무엇이든 물어보세요!")

    # 사용자 입력 처리
    if user_input:
        # 사용자 메시지 즉시 표시 - 반응성 향상
        with st.chat_message("user", avatar="😀"):
            st.markdown(user_input, unsafe_allow_html=False)
        
        # 사용자 메시지 저장 - UUID로 고유 ID 할당하여 추적 가능
        user_message_id = str(uuid.uuid4())
        st.session_state['chat_history'].append({
            "id": user_message_id,
            "role": "user",
            "content": user_input
        })

        # 봇 응답 생성 및 즉시 저장
        with st.spinner("생각 중..."):
            try:
                # API 호출하여 응답 생성 (JSON 응답 처리)
                json_response_str = st.session_state['counselor'].send_question(user_input)
                response_data = json.loads(json_response_str) # JSON 파싱
                bot_response_content = response_data.get("front_message", "오류: 응답 내용을 가져올 수 없습니다.") # front_message 추출

                # 백엔드 데이터 추출 (필요시)
                backend_data = {
                    key: response_data.get(key)
                    for key in response_data.keys()
                    if key.startswith('back_')
                }

                bot_message_id = str(uuid.uuid4())

                # 응답을 세션에 먼저 저장
                st.session_state['chat_history'].append({
                    "id": bot_message_id,
                    "role": "assistant",
                    "content": bot_response_content, # front_message만 content로 저장
                    "backend_data": backend_data # 백엔드 데이터 추가 저장
                })

                # Google Drive에 즉시 저장 (채팅 데이터 및 모델)
                if st.session_state['user_id']:
                    chat_data_saved = save_chat_data(st.session_state['user_id'])

                    # 모델 저장 로직 추가 (CounselingWithGemini.save_chatmodel 사용)
                    credentials = get_credentials()
                    model_saved = False
                    if st.session_state.get('model_file_id'):
                        # 기존 파일 업데이트 시도
                        file_id_or_success = st.session_state['counselor'].save_chatmodel(
                            folder_id=output_drive_id,
                            user_id=st.session_state['user_id'],
                            file_id=st.session_state['model_file_id'],
                            credentials=credentials
                        )
                        model_saved = bool(file_id_or_success)
                    else:
                        # 새 파일 생성 시도
                        file_id = st.session_state['counselor'].save_chatmodel(
                            folder_id=output_drive_id,
                            user_id=st.session_state['user_id'],
                            credentials=credentials
                        )
                        if file_id:
                            st.session_state['model_file_id'] = file_id
                            model_saved = True
                        else:
                            model_saved = False

                    # 저장 결과 피드백 (문제가 있을 경우에만)
                    if not chat_data_saved and not model_saved:
                        st.warning("채팅 및 상담 내용 저장 중 문제가 발생했습니다.")
                    elif not model_saved:
                        st.warning("상담 내용 저장 중 문제가 발생했습니다.")
                    elif not chat_data_saved:
                        st.warning("채팅 내용 저장 중 문제가 발생했습니다.")

            except json.JSONDecodeError:
                # 실패한 원본 문자열 로깅 추가
                print(f"JSON 파싱 실패. 원본 응답: {json_response_str}")
                st.error("모델 응답을 처리하는 중 오류가 발생했습니다. (JSON 형식 오류)")
                # 오류 발생 시에도 사용자에게 피드백을 줄 수 있도록 처리
                error_message_id = str(uuid.uuid4())
                st.session_state['chat_history'].append({
                    "id": error_message_id,
                    "role": "assistant",
                    "content": "죄송합니다. 답변을 처리하는 데 문제가 발생했습니다.",
                    "backend_data": {"error": "JSONDecodeError"}
                })
                # 오류 발생 시 채팅 데이터만 저장 시도
                if st.session_state['user_id']:
                    save_chat_data(st.session_state['user_id']) # 오류 상태도 저장
            except Exception as e:
                st.error(f"모델 응답 처리 중 예상치 못한 오류 발생: {e}")
                error_message_id = str(uuid.uuid4())
                st.session_state['chat_history'].append({
                    "id": error_message_id,
                    "role": "assistant",
                    "content": "죄송합니다. 답변을 처리하는 데 문제가 발생했습니다.",
                    "backend_data": {"error": str(e)}
                })
                # 오류 발생 시 채팅 데이터만 저장 시도
                if st.session_state['user_id']:
                    save_chat_data(st.session_state['user_id']) # 오류 상태도 저장

        # 페이지 강제 재로딩
        st.rerun()