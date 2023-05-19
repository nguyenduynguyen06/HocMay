import streamlit as st

st.set_page_config(page_title="Machine Learning", page_icon=":guardsman:", layout="wide")
page_bg_img = f"""
   <style>

   [data-testid="stSidebar"] > div:first-child {{
   background-image: url("https://mega.com.vn/media/news/0106_hinh-nen-may-tinh-full-hd63.jpg");
   background-position: center; 
   }}
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://scr.vn/wp-content/uploads/2020/07/H%C3%ACnh-n%E1%BB%81n-desktop-%C4%91%E1%BB%99-ph%C3%A2n-gi%E1%BA%A3i-l%E1%BB%9Bn-scaled-2048x1280.jpg");
     background-size: cover;
         background-position: top left;
         background-repeat: no-repeat;
         background-attachment: local;
    }}
   </style>
   """

st.markdown(page_bg_img, unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: white' >Hãy chọn chức năng phía dưới: </p>", unsafe_allow_html=True)
def intro():
    import streamlit as st
    st.write("# Chào mừng đến Project cuối kì môn Học máy! 👋")


    st.markdown(
        """
        Project này làm về đề tài "Tạo trang web cho môn học Machine Learing dùng Streamlit""

        **👈 Select a demo from the dropdown on the left** to see some examples
        of what Streamlit can do!

        Thành viên của nhóm

        1. Lê Quang Dương - 20110454
        2. Nguyễn Duy Nguyễn - 20110530

        Link github của Project:

        *chưa gắn link*"""
    )

def mapping_demo():
    import streamlit as st
    import pandas as pd
    import pydeck as pdk

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart)
to display geospatial data.
"""
    )

    @st.cache_data
    def from_data_file(filename):
        url = (
            "http://raw.githubusercontent.com/streamlit/"
            "example-data/master/hello/v1/%s" % filename
        )
        return pd.read_json(url)

    try:
        ALL_LAYERS = {
            "Bike Rentals": pdk.Layer(
                "HexagonLayer",
                data=from_data_file("bike_rental_stats.json"),
                get_position=["lon", "lat"],
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
            ),
            "Bart Stop Exits": pdk.Layer(
                "ScatterplotLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_color=[200, 30, 0, 160],
                get_radius="[exits]",
                radius_scale=0.05,
            ),
            "Bart Stop Names": pdk.Layer(
                "TextLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_text="name",
                get_color=[0, 0, 0, 200],
                get_size=15,
                get_alignment_baseline="'bottom'",
            ),
            "Outbound Flow": pdk.Layer(
                "ArcLayer",
                data=from_data_file("bart_path_stats.json"),
                get_source_position=["lon", "lat"],
                get_target_position=["lon2", "lat2"],
                get_source_color=[200, 30, 0, 160],
                get_target_color=[200, 30, 0, 160],
                auto_highlight=True,
                width_scale=0.0001,
                get_width="outbound",
                width_min_pixels=3,
                width_max_pixels=30,
            ),
        }
        st.sidebar.markdown("### Map Layers")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)
        ]
        if selected_layers:
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state={
                        "latitude": 37.76,
                        "longitude": -122.4,
                        "zoom": 11,
                        "pitch": 50,
                    },
                    layers=selected_layers,
                )
            )
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

def plotting_demo():
    import streamlit as st
    import numpy as np
    import cv2 as cv
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image('logo.png', width=150)
    with col2:
        st.markdown('<h4 style="color: white">Trường Đại học Sư Phạm Kỹ Thuật TP.HCM</h4>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color: white">Môn học: Machine Learning &nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp; GVHD: Trần Tiến Đức</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="color: white">Thành viên tham gia:</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="color: white">Nguyễn Duy Nguyễn - 20110530</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="color: white">Lê Quang Dương - 20110515</p>',
            unsafe_allow_html=True)
    global index
    st.markdown('<h3 style="color: white">Phát hiện khuôn mặt</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: white">Nhấn nút bên dưới để bật camera và xem kết quả</p>',
        unsafe_allow_html=True)

    def run_face_detection(running_flag):
        deviceId = 0
        cap = cv.VideoCapture(deviceId)
        FRAME_WINDOW = st.image([])
        if 'Dừng lại' not in st.session_state:
            st.session_state.stop = False
            stop = False

        press = st.button('❗Dừng lại')
        if press:
            if st.session_state.stop == False:
                st.session_state.stop = True
                cap.release()
            else:
                st.session_state.stop = False

        print('Trang thai nhan Stop', st.session_state.stop)

        if 'frame_stop' not in st.session_state:
            frame_stop = cv.imread('PhatHienKhuonMat_Facebook_Streamlit/stop.jpg')
            st.session_state.frame_stop = frame_stop
            print('Đã load stop.jpg')

        if st.session_state.stop == True:
            FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')
        detector = cv.FaceDetectorYN.create(
            'PhatHienKhuonMat_Facebook_Streamlit/face_detection_yunet_2022mar.onnx',
            "",
            (320, 320),
            0.9,
            0.3,
            5000
        )

        tm = cv.TickMeter()
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize([frameWidth, frameHeight])

        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            frame = cv.resize(frame, (frameWidth, frameHeight))

            # Inference
            tm.start()
            faces = detector.detect(frame)  # faces is a tuple
            tm.stop()

            # Draw results on the input image
            visualize(frame, faces, tm.getFPS())

            # Visualize results
            FRAME_WINDOW.image(frame, channels='BGR')

            if not running_flag:
                break  # thoát khỏi vòng lặp nếu running_flag là False

        cap.release()  # release camera resources


    def visualize(input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0),
                             thickness)
                cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
        cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    start_button = st.button('🚀 Bắt đầu')
    if start_button:
        run_face_detection(True)


def nhandangchuso():
    import streamlit as st
    from PIL import Image
    import base64
    import numpy as np
    from tensorflow import keras
    from keras.models import model_from_json
    from keras.optimizers import SGD
    import cv2
    model_architecture = "NhanDangChuSo/digit_config.json"
    model_weights = "NhanDangChuSo/digit_weight.h5"
    model = model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)

    optim = SGD()
    model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

    mnist = keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_test_image = X_test

    RESHAPED = 784

    X_test = X_test.reshape(10000, RESHAPED)
    X_test = X_test.astype('float32')

    # normalize in [0,1]
    X_test /= 255

    index = np.random.randint(0, 9999, 150)

    def generate_and_predict():
        global index
        # Generate random image
        digit_random = np.zeros((10 * 28, 15 * 28), dtype=np.uint8)
        for i in range(0, 150):
            m = i // 15
            n = i % 15
            digit_random[m * 28:(m + 1) * 28, n * 28:(n + 1) * 28] = X_test_image[index[i]]
        cv2.imwrite('NhanDangChuSo/digit_random.jpg', digit_random)
        image = Image.open('NhanDangChuSo/digit_random.jpg')
        # Predict
        X_test_sample = np.zeros((150, 784), dtype=np.float32)
        for i in range(0, 150):
            X_test_sample[i] = X_test[index[i]]
        prediction = model.predict(X_test_sample)
        s = ''
        for i in range(0, 150):
            ket_qua = np.argmax(prediction[i])
            s = s + str(ket_qua) + ' '
            if (i + 1) % 15 == 0:
                s = s + '\n'
        return image, s

    def app():
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image('logo.png', width=150)
        with col2:
            st.markdown('<h4 style="color: white">Trường Đại học Sư Phạm Kỹ Thuật TP.HCM</h4>', unsafe_allow_html=True)
            st.markdown(
                '<p style="color: white">Môn học: Machine Learning &nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp; GVHD: Trần Tiến Đức</p>',
                unsafe_allow_html=True)
            st.markdown(
                '<p style="color: white">Thành viên tham gia:</p>',
                unsafe_allow_html=True)
            st.markdown(
                '<p style="color: white">Nguyễn Duy Nguyễn - 20110530</p>',
                unsafe_allow_html=True)
            st.markdown(
                '<p style="color: white">Lê Quang Dương - 20110515</p>',
                unsafe_allow_html=True)
        global index
        st.markdown('<h3 style="color: white">Nhận dạng chữ số viết tay</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color: white">Nhấn nút bên dưới để tạo ra ảnh ngẫu nhiên và xem kết quả</p>',
            unsafe_allow_html=True)
        st.write('')

        container = st.container()
        index = np.random.randint(0, 9999, 150)
        image, result = generate_and_predict()
        with container:
            btn_random = st.button('🌆 Tạo ảnh mới')
            col1, col2 = st.columns([1,1])
            image_placeholder = col1.empty()
            if btn_random:
                image_placeholder.image(image, width=500)
                with col2:
                    st.markdown('<h5 style="color: white">Kết quả</h5>', unsafe_allow_html=True)
                    st.markdown('<p style="color: white">' + result + '</p>', unsafe_allow_html=True)

    if __name__ == '__main__':
        app()


def cali():
    import joblib
    import streamlit as st
    import pandas as pd
    import numpy as np
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image('logo.png', width=150)
    with col2:
        st.markdown('<h4 style="color: white">Trường Đại học Sư Phạm Kỹ Thuật TP.HCM</h4>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color: white">Môn học: Machine Learning &nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp; GVHD: Trần Tiến Đức</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="color: white">Thành viên tham gia:</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="color: white">Nguyễn Duy Nguyễn - 20110530</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="color: white">Lê Quang Dương - 20110515</p>',
            unsafe_allow_html=True)
    global index
    st.markdown('<h3 style="color: white">Dự báo giá nhà California</h3>', unsafe_allow_html=True)
    st.write('')
    def my_format(x):
        s = "{:,.0f}".format(x)
        L = len(s)
        if L < 14:
            s = '&nbsp' * (14 - L) + s
        return s

    forest_reg = joblib.load("HoiQuyRungNgauNhien_Streamlit/forest_reg_model.pkl")

    column_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                    'total_bedrooms', 'population', 'households', 'median_income',
                    'rooms_per_household', 'population_per_household',
                    'bedrooms_per_room', 'ocean_proximity_1',
                    'ocean_proximity_2', 'ocean_proximity_3',
                    'ocean_proximity_4', 'ocean_proximity_5']
    x_test = pd.read_csv('HoiQuyRungNgauNhien_Streamlit/x_test.csv', header=None, names=column_names)
    y_test = pd.read_csv('HoiQuyRungNgauNhien_Streamlit/y_test.csv', header=None)
    y_test = y_test.to_numpy()
    N = len(x_test)
    st.dataframe(x_test)
    get_5_rows = st.button('💭 Lấy 5 hàng ngẫu nhiên và dự báo')
    if get_5_rows:
        st.markdown(
            '<p style="color: white">Kết quả: </p>',
            unsafe_allow_html=True)
        index = np.random.randint(0, N - 1, 5)
        some_data = x_test.iloc[index]
        st.dataframe(some_data)
        result = 'y_test:' + '&nbsp&nbsp&nbsp&nbsp'
        for i in index:
            s = my_format(y_test[i, 0])
            result = result + s
        result = '<p style="font-family:Consolas; color:White   ; font-size: 15px;">' + result + '</p>'
        st.markdown(result, unsafe_allow_html=True)

        some_data = some_data.to_numpy()
        y_pred = forest_reg.predict(some_data)
        result = 'y_predict:' + '&nbsp'
        for i in range(0, 5):
            s = my_format(y_pred[i])
            result = result + s
        result = '<p style="font-family:Consolas; color:White ; font-size: 15px;">' + result + '</p>'
        st.markdown(result, unsafe_allow_html=True)



page_names_to_funcs = {
    "—": intro,
    "Phát hiện khuôn mặt": plotting_demo,
    "Nhận dạng khuôn mặt của chính mình": mapping_demo,
    "Nhận dạng 10 chữ số viết tay": nhandangchuso,
    "Dự báo nhà Cali": cali,
}

demo_name = st.sidebar.selectbox("", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()