import cv2
import mediapipe as mp
import math
import os  # 파일 존재 여부 확인을 위해 추가

# Mediapipe 손 인식 솔루션 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# --------------------------------------------------------------------------
# 2단계: 외부 지식 증강 (RAG - Bridge Part)
# --------------------------------------------------------------------------
def setup_mock_vector_db():
    """
        Creates a mock vector DB mapping gesture descriptions to cultural meanings.
        """
    db = {
        "The thumb is extended upwards, while the other four fingers are in a fist": {
            "description": "Generally means 'good', 'positive', or 'agreement'.",
            "warning": "However, be cautious as this gesture can be highly offensive in parts of Brazil, West Africa, and the Middle East."
        },
        "The index and middle fingers are extended in a V shape, while the other fingers are folded": {
            "description": "Commonly represents 'peace' or 'victory'.",
            "warning": "However, if the back of the hand faces the other person, it can be an insulting gesture in the UK and some Commonwealth countries."
        },
    }
    return db


def query_knowledge_base(description: str, db: dict):
    """
    Queries the mock DB with the input text and returns relevant information.
    """
    return db.get(description, None)


# --------------------------------------------------------------------------
# 1단계: 시각 정보의 구조적 텍스트화 (Vision Part)
# --------------------------------------------------------------------------
def get_structured_text_from_landmarks(hand_landmarks):
    """
    Analyzes landmark coordinates from MediaPipe to convert a gesture's shape
    into an objective text description.
    """
    landmarks = hand_landmarks.landmark
    thumb_is_up = landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y
    index_finger_is_up = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[
        mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_finger_is_up = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_finger_is_up = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[
        mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_is_up = landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y

    if thumb_is_up and not index_finger_is_up and not middle_finger_is_up and not ring_finger_is_up and not pinky_is_up:
        return "The thumb is extended upwards, while the other four fingers are in a fist"
    if index_finger_is_up and middle_finger_is_up and not ring_finger_is_up and not pinky_is_up and not thumb_is_up:
        return "The index and middle fingers are extended in a V shape, while the other fingers are folded"
    return "Unknown Gesture"


# --------------------------------------------------------------------------
# 3단계: 통제된 최종 답변 생성 (Language Part)
# --------------------------------------------------------------------------
def generate_final_response(rag_context):
    """
    Generates the final response based on the augmented information from RAG,
    following the system prompt rules.
    """
    if not rag_context:
        return "Could not find the meaning of the gesture in the database.\nThe meaning is ambiguous."

    common_meaning = rag_context.get("description", "")
    warning = rag_context.get("warning", "")

    response = "Gesture Analysis Result:\n"
    response += f" - Common Meaning: {common_meaning}\n"
    if warning:
        response += f" - ! Cultural Warning: {warning}"
    return response


# --------------------------------------------------------------------------
# 메인 워크플로우 실행
# --------------------------------------------------------------------------
def main():
    # 2단계: RAG DB 준비
    knowledge_db = setup_mock_vector_db()

    # --- 이미지 파일 불러오기로 변경된 파트 ---
    IMAGE_FILE = 'gesture_image.jpg'  # 분석할 이미지 파일명을 입력하세요.

    # 파일 존재 여부 확인
    if not os.path.exists(IMAGE_FILE):
        print(f"Error: '{IMAGE_FILE}' not found. Please check the filename and path.")
        return

    # MediaPipe Hands 모델 로드
    with mp_hands.Hands(
            static_image_mode=True,  # 이미지 모드 활성화
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:

        # 이미지 파일 읽기
        image = cv2.imread(IMAGE_FILE)

        # BGR to RGB 변환
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 최종 결과 출력 초기화
        final_output = "Could not find a hand in the image."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 손 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                # --- 아키텍처 워크플로우 실행 ---
                # 1단계: 시각 정보 -> 구조적 텍스트
                structured_text = get_structured_text_from_landmarks(hand_landmarks)

                if structured_text != "Unknown Gesture":
                    # 2단계: 텍스트 -> RAG로 지식 검색
                    rag_context = query_knowledge_base(structured_text, knowledge_db)

                    # 3단계: 보강된 정보 -> 최종 답변 생성
                    final_output = generate_final_response(rag_context)
                else:
                    final_output = ("This is not an analyzable gesture.")

        # 화면에 결과 텍스트 출력
        y0, dy = 50, 40
        for i, line in enumerate(final_output.split('\n')):
            y = y0 + i * dy
            cv2.putText(image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)  # Outline
            cv2.putText(image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Text

        cv2.imshow('Vision-RAG-LLM Gesture Recognition Demo', image)
        # Save the final image with the results
        cv2.imwrite('result_image_.jpg', image)
        print("Press any key to close the result window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # # --- 기존 카메라 실행 파트 (주석 처리) ---
    # cap = cv2.VideoCapture(0)
    # with mp_hands.Hands(
    #     max_num_hands=1,
    #     min_detection_confidence=0.7,
    #     min_tracking_confidence=0.7) as hands:
    #     while cap.isOpened():
    #         success, image = cap.read()
    #         if not success:
    #             continue
    #         image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    #         image.flags.writeable = False
    #         results = hands.process(image)
    #         image.flags.writeable = True
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         final_output = "손을 카메라에 보여주세요."
    #         if results.multi_hand_landmarks:
    #             for hand_landmarks in results.multi_hand_landmarks:
    #                 mp_drawing.draw_landmarks(
    #                     image,
    #                     hand_landmarks,
    #                     mp_hands.HAND_CONNECTIONS)
    #                 structured_text = get_structured_text_from_landmarks(hand_landmarks)
    #                 if structured_text != "알 수 없는 제스처":
    #                     rag_context = query_knowledge_base(structured_text, knowledge_db)
    #                     final_output = generate_final_response(rag_context)
    #                 else:
    #                     final_output = "분석 가능한 제스처가 아닙니다."

    #         y0, dy = 50, 40
    #         for i, line in enumerate(final_output.split('\n')):
    #             y = y0 + i * dy
    #             cv2.putText(image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5) # Outline
    #             cv2.putText(image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) # Text
    #         cv2.imshow('Vision-RAG-LLM Gesture Recognition Demo', image)
    #         if cv2.waitKey(5) & 0xFF == 27: # ESC 키 누르면 종료
    #             break
    # cap.release()


if __name__ == '__main__':
    main()