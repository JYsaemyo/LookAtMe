``` Python
 
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# LSTM 모델 로드
model = load_model('lstm_model_v2.h5')
actions = ['DUMMY_1', 'DUMMY_2', 'rH', 'lH', 'YaleYale', 'UJJOOJJOO', 'DORA']

# 입력 데이터 전처리 함수
def preprocess_frame(frame, sequence, max_seq_length=30):
    # Mediapipe로 손 관절 피처 추출
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            joint_data = []
            for landmark in hand_landmarks.landmark:
                joint_data.extend([landmark.x, landmark.y, landmark.z])  # x, y, z 좌표를 리스트로 저장
            sequence.append(joint_data)
            if len(sequence) > max_seq_length:
                sequence.pop(0)
            return sequence, results, True
    return sequence, results, False

# 실시간 웹캠 입력
cap = cv2.VideoCapture(0)
sequence = []
current_action = "None"
action_start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 전처리
    sequence, results, detected = preprocess_frame(frame, sequence)
    
    if detected and len(sequence) == 30:  # 모델 입력 조건 만족
        input_data = np.expand_dims(sequence, axis=0)  # (1, 30, 99)
        input_data = np.array(input_data, dtype=np.float32)

        # 모델 예측
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        action = actions[predicted_class]
        
        # 현재 동작이 변경되었을 때 알림
        if action != current_action:
            current_action = action
            action_start_time = cv2.getTickCount()
        
        # 화면에 동작과 함께 시간 표시
        cv2.putText(frame, f'Action: {action}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mediapipe 손 관절 그리기
    if detected and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 화면에 동작이 나타나는 시간
    if action_start_time:
        action_time = (cv2.getTickCount() - action_start_time) / cv2.getTickFrequency()
        cv2.putText(frame, f'Time: {action_time:.2f}s', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 프레임 출력
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

 
 
```
