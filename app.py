import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai

st.set_page_config(page_title="AI Real Estate Advisor", layout="wide")

# --------------------- CẤU HÌNH ---------------------
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "LightGBM": "models/best_lgb_model.pkl",
        "XGBoost": "models/best_xgb_model.pkl",
        "HistGradientBoosting": "models/best_hgb_model.pkl"
    }
    if model_name not in model_paths:
        st.error(f"Mô hình {model_name} không tồn tại!")
        return None
    try:
        return joblib.load(model_paths[model_name])
    except FileNotFoundError:
        st.error(f"File mô hình {model_paths[model_name]} không tồn tại!")
        return None

# Cấu hình Google Gemini API
# genai.configure(api_key=st.secrets["general"]["GEMINI_API_KEY"])
genai.configure(api_key="AIzaSyDBbom8P1ip9cc0bWDyfj5-s51S9f1P7uk")
model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")

# --------------------- XỬ LÝ DỮ LIỆU ---------------------
def process_single_real_estate_input(input_data):
    df = pd.DataFrame([input_data])

    # Xử lý diện tích
    df['Diện tích'] = pd.to_numeric(
        df['Diện tích'].astype(str).str.replace('m2', '').str.strip(),
        errors='coerce'
    ).fillna(0)

    # Xử lý đường trước nhà
    def extract_width(value):
        value = str(value).replace('m', '').strip()
        if '-' in value:
            nums = [float(x) for x in value.split('-')]
            return sum(nums) / len(nums)
        return float(value) if value else 0
    df['Đường trước nhà'] = df['Đường trước nhà'].apply(extract_width)

    # Trích xuất thành phố
    df['Thành phố'] = df['Địa chỉ'].str.split(',').str[-1].str.strip()

    # Chuyển các cột số
    for col in ['Phòng ngủ', 'Số tầng', 'Số toilet', 'Số phòng khách']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Ánh xạ giá trị phân loại
    mapping_dicts = {
        'Pháp lý': {'Sổ đỏ': 0, 'Sổ hồng': 1, 'Hợp đồng mua bán': 2},
        'Hướng nhà': {'Đông': 0, 'Tây': 1, 'Nam': 2, 'Bắc': 3},
        'Loại địa ốc': {
            'Nhà phố': 0, 'Nhà riêng': 1, 'Biệt thự': 2,
            'căn hộ chung cư': 3, 'nhà hàng - khách sạn': 4, 'căn hộ mini - dịch vụ': 5
        },
        'Thành phố': {
            'Hồ Chí Minh': 0, 'Hà Nội': 1, 'Đà Nẵng': 2,
            'Tiền Giang': 3, 'Bình Dương': 4, 'Đồng Nai': 5
        }
    }

    for col, mapping in mapping_dicts.items():
        df[col] = df[col].map(mapping).fillna(0).astype(int)

    # Định nghĩa features
    features = [
        'Diện tích', 'Phòng ngủ', 'Số tầng', 'Số toilet', 'Số phòng khách',
        'Đường trước nhà', 'Pháp lý', 'Hướng nhà', 'Loại địa ốc', 'Thành phố'
    ]

    return df[features]

# --------------------- DỰ ĐOÁN GIÁ ---------------------
def predict_house_price(X_input, selected_model):
    try:
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input)

        if X_input.shape[1] != 10:
            st.error("Sai số lượng đặc trưng đầu vào!")
            return None

        return np.expm1(selected_model.predict(X_input))
    except Exception as e:
        st.error(f"Lỗi dự đoán: {str(e)}")
        return None

# --------------------- TẠO PROMPT CHO GEMINI ---------------------
def create_gemini_prompt(input_data, predicted_price):
    prompt = f"""
    [THÔNG TIN BẤT ĐỘNG SẢN]
    - Địa chỉ: {input_data['Địa chỉ']}
    - Diện tích: {input_data['Diện tích']}
    - Phòng ngủ: {input_data['Phòng ngủ']}
    - Số tầng: {input_data['Số tầng']}
    - Số toilet: {input_data['Số toilet']}
    - Số phòng khách: {input_data['Số phòng khách']}
    - Đường trước nhà: {input_data['Đường trước nhà']}
    - Pháp lý: {input_data['Pháp lý']}
    - Hướng nhà: {input_data['Hướng nhà']}
    - Loại địa ốc: {input_data['Loại địa ốc']}

    [GIÁ DỰ ĐOÁN]
    {predicted_price:,.2f} tỷ đồng

    Hãy phân tích:
    1. Định giá cho bất động sản này dựa trên thông tin cung cấp, giải thích lý do.
    2. Đánh giá mức giá dự đoán (cao/thấp/phù hợp) so với mặt bằng khu vực, tập trung vào cơ hội từ mức giá này.
    3. Ưu điểm nổi bật và nhược điểm (nếu có) của bất động sản, nhấn mạnh các yếu tố tích cực.
    4. Tiềm năng đầu tư, nêu rõ cơ hội sinh lời hoặc lợi ích lâu dài.
    5. Khuyến nghị: CÓ NÊN MUA? (Giải thích rõ lý do, ưu tiên góc nhìn tích cực về giá trị bất động sản).
    6. 3 lưu ý quan trọng nếu quyết định mua, giúp tối ưu hóa giá trị đầu tư.

    [LƯU Ý]
    - Đi thẳng vào phân tích, không chào hỏi, không giới thiệu
    - Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc.
    - Ưu tiên dữ liệu thực tế về thị trường.
    - Đưa ra con số ước lượng cụ thể nếu có thể.
    - Nhấn mạnh các khía cạnh tích cực của mức giá dự đoán và giá trị bất động sản.
    """
    return prompt

# --------------------- GIAO DIỆN STREAMLIT ---------------------
st.title('🏠 AI ĐỊNH GIÁ & TƯ VẤN BẤT ĐỘNG SẢN')
st.markdown("""
    <style>
    .stTextInput input, .stSelectbox select {
        font-size: 16px !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.form("real_estate_form"):
    col1, col2 = st.columns(2)

    with col1:
        dia_chi = st.text_input("Địa chỉ (có dấu phẩy cuối, VD: 'Quận 1, Hồ Chí Minh')", "Quận 1, Hồ Chí Minh")
        dien_tich = st.text_input("Diện tích (VD: 50 m2)", "50 m2")
        duong_truoc_nha = st.text_input("Đường trước nhà (VD: 5m hoặc 4-6m)", "5m")
        phong_ngu = st.text_input("Số phòng ngủ", "2")
        so_tang = st.text_input("Số tầng", "1")

    with col2:
        so_toilet = st.text_input("Số toilet", "1")
        so_phong_khach = st.text_input("Số phòng khách", "1")
        phap_ly = st.selectbox("Pháp lý", ["Sổ đỏ", "Sổ hồng", "Hợp đồng mua bán"])
        huong_nha = st.selectbox("Hướng nhà", ["Đông", "Tây", "Nam", "Bắc"])
        loai_dia_oc = st.selectbox("Loại địa ốc", [
            "Nhà phố", "Nhà riêng", "Biệt thự",
            "căn hộ chung cư", "nhà hàng - khách sạn", "căn hộ mini - dịch vụ"
        ])
        model_choice = st.selectbox("Chọn mô hình dự đoán", ["LightGBM", "XGBoost", "HistGradientBoosting"])

    submitted = st.form_submit_button("🚀 Dự đoán giá & Nhận tư vấn AI")

if submitted:
    with st.spinner('Đang phân tích...'):
        # Tải mô hình được chọn
        selected_model = load_model(model_choice)
        if selected_model is None:
            st.error("Không thể tải mô hình!")
        else:
            input_data = {
                'Địa chỉ': dia_chi,
                'Diện tích': dien_tich,
                'Đường trước nhà': duong_truoc_nha,
                'Phòng ngủ': phong_ngu,
                'Số tầng': so_tang,
                'Số toilet': so_toilet,
                'Số phòng khách': so_phong_khach,
                'Pháp lý': phap_ly,
                'Hướng nhà': huong_nha,
                'Loại địa ốc': loai_dia_oc
            }

            processed_data = process_single_real_estate_input(input_data)
            predicted_price = predict_house_price(processed_data, selected_model)

            if predicted_price is not None:
                predicted_price = predicted_price[0]

                # Hiển thị kết quả
                st.success(f"**💰 Giá dự đoán (mô hình {model_choice}):** {predicted_price:,.2f} tỷ đồng")

                # Tư vấn từ Gemini
                with st.spinner('AI đang tư vấn...'):
                    prompt = create_gemini_prompt(input_data, predicted_price)
                    try:
                        response = model.generate_content(prompt)
                    except Exception as e:
                        st.error(f"Lỗi khi gọi Gemini API: {str(e)}")
                        response = None

                    if response:
                        st.subheader("🤖 Tư vấn chuyên gia AI")
                        st.markdown(response.text)

                        # Lưu vào lịch sử
                        if 'history' not in st.session_state:
                            st.session_state.history = []

                        st.session_state.history.append({
                            'input': input_data,
                            'price': predicted_price,
                            'model': model_choice,
                            'advice': response.text
                        })

# Hiển thị lịch sử
if 'history' in st.session_state and st.session_state.history:
    st.divider()
    st.subheader("📚 Lịch sử dự đoán")

    for i, item in enumerate(st.session_state.history[::-1], 1):
        with st.expander(f"#{i}: {item['input']['Địa chỉ']} - {item['price']:,.2f} tỷ (Mô hình: {item['model']})"):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.json(item['input'], expanded=False)
            with col2:
                st.markdown(f"**💡 Tư vấn:**\n{item['advice']}")
