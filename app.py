import streamlit as st
from ultralytics import YOLO
from  PIL import Image
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from collections import Counter
import io


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Supermarket Product Identification",
    page_icon="🛒",
    layout="wide"
)

MODEL_PATH = Path("models/best.pt")

# Fixed 17-category mapping used during training
CLASS_NAMES = [
    "alcohol",
    "candy",
    "canned_food",
    "chocolate",
    "dessert",
    "dried_food",
    "dried_fruit",
    "drink",
    "gum",
    "instant_drink",
    "instant_noodles",
    "milk",
    "personal_hygiene",
    "puffed_food",
    "seasoner",
    "stationery",
    "tissue"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained YOLO11s model once."""
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found: {MODEL_PATH}\n\n"
            "Please make sure best.pt is inside the models folder."
        )
        st.stop()

    return YOLO(str(MODEL_PATH))


model = load_model()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("Smart Supermarket Product Identification System")

st.write(
    """
    Upload one or more supermarket/product images. The system detects
    individual products using the trained YOLO11s model and generates
    category-wise product statistics.
    """
)

st.info(
    "The application performs inference locally using the trained YOLO model. "
    "The uploaded images are not used for training."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.write(
    f"Current threshold: **{confidence_threshold:.2f}**"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

st.sidebar.write("Model: YOLO11s")
st.sidebar.write("Categories: 17")
st.sidebar.write("Inference: Local")


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "Upload supermarket/product images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_category_name(class_id):
    """
    Convert YOLO class ID into the 17-category name.
    """
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]

    return f"class_{class_id}"


def calculate_statistics(detected_categories):
    """
    Calculate total products, category counts and percentages.
    """

    total_products = len(detected_categories)

    counts = Counter(detected_categories)

    rows = []

    for category in CLASS_NAMES:
        count = counts.get(category, 0)

        if total_products > 0:
            percentage = (count / total_products) * 100
        else:
            percentage = 0.0

        rows.append(
            {
                "Category": category,
                "Count": count,
                "Percentage": percentage
            }
        )

    df = pd.DataFrame(rows)

    # Show categories with detections first
    df = df.sort_values(
        by=["Count", "Category"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return total_products, df


def draw_detections(image, results):
    """
    Draw bounding boxes, category labels and confidence
    values on the detected image.
    """

    annotated = image.copy()

    boxes = results.boxes

    for box in boxes:

        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Class ID
        class_id = int(box.cls[0].cpu().item())

        # Confidence
        confidence = float(box.conf[0].cpu().item())

        category = get_category_name(class_id)

        # Label
        label = f"{category} {confidence:.2f}"

        # Bounding box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 200, 0),
            2
        )

        # Text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        # Keep label inside image
        label_y = max(y1, text_height + baseline + 5)

        # Label background
        cv2.rectangle(
            annotated,
            (x1, label_y - text_height - baseline - 5),
            (x1 + text_width + 5, label_y),
            (0, 200, 0),
            -1
        )

        # Label text
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return annotated


def create_detection_dataframe(results):
    """
    Create a detailed table containing every detected object.
    """

    rows = []

    boxes = results.boxes

    for index, box in enumerate(boxes, start=1):

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        class_id = int(box.cls[0].cpu().item())

        confidence = float(box.conf[0].cpu().item())

        category = get_category_name(class_id)

        rows.append(
            {
                "Detection": index,
                "Category": category,
                "Confidence": round(confidence, 4),
                "X1": round(float(x1), 1),
                "Y1": round(float(y1), 1),
                "X2": round(float(x2), 1),
                "Y2": round(float(y2), 1)
            }
        )

    return pd.DataFrame(rows)


def dataframe_to_csv(df):
    """
    Convert dataframe to downloadable CSV.
    """
    return df.to_csv(index=False).encode("utf-8")


def image_to_bytes(image):
    """
    Convert OpenCV/PIL image to PNG bytes.
    """

    success, buffer = cv2.imencode(
        ".png",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )

    if not success:
        return None

    return io.BytesIO(buffer.tobytes()).getvalue()


# ============================================================
# PROCESS UPLOADED IMAGES
# ============================================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} image(s) uploaded successfully."
    )

    # Store results for all uploaded images
    all_image_statistics = []

    # Store all detected categories from all images
    all_categories = []

    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for image_number, uploaded_file in enumerate(
        uploaded_files,
        start=1
    ):

        image = Image.open(uploaded_file).convert("RGB")

        image_array = np.array(image)

        # ----------------------------------------------------
        # RUN YOLO DETECTION
        # ----------------------------------------------------

        results = model.predict(
            source=image_array,
            conf=confidence_threshold,
            verbose=False
        )[0]

        # ----------------------------------------------------
        # EXTRACT DETECTIONS
        # ----------------------------------------------------

        detected_categories = []

        for box in results.boxes:

            class_id = int(
                box.cls[0].cpu().item()
            )

            category = get_category_name(class_id)

            detected_categories.append(category)

        # Add to overall results
        all_categories.extend(detected_categories)

        # ----------------------------------------------------
        # CALCULATE STATISTICS
        # ----------------------------------------------------

        total_products, statistics_df = calculate_statistics(
            detected_categories
        )

        # ----------------------------------------------------
        # CREATE ANNOTATED IMAGE
        # ----------------------------------------------------

        annotated_image = draw_detections(
            image_array.copy(),
            results
        )

        # ----------------------------------------------------
        # DETAILED DETECTION TABLE
        # ----------------------------------------------------

        detection_df = create_detection_dataframe(results)

        # ----------------------------------------------------
        # SAVE IMAGE STATISTICS
        # ----------------------------------------------------

        all_image_statistics.append(
            {
                "Image": uploaded_file.name,
                "Total Products": total_products
            }
        )

        # ====================================================
        # IMAGE SECTION
        # ====================================================

        st.markdown("---")

        st.header(
            f"Image {image_number}: {uploaded_file.name}"
        )

        # ----------------------------------------------------
        # BASIC IMAGE INFORMATION
        # ----------------------------------------------------

        width, height = image.size

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.metric(
                "Image Width",
                f"{width} px"
            )

        with info_col2:
            st.metric(
                "Image Height",
                f"{height} px"
            )

        with info_col3:
            st.metric(
                "Total Products Detected",
                total_products
            )

        # ----------------------------------------------------
        # IMAGE COMPARISON
        # ----------------------------------------------------

        image_col1, image_col2 = st.columns(2)

        with image_col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with image_col2:

            st.subheader("Detected Products")

            st.image(
                annotated_image,
                use_container_width=True
            )

        # ----------------------------------------------------
        # CATEGORY STATISTICS
        # ----------------------------------------------------

        st.subheader("Category-wise Product Statistics")

        statistics_display = statistics_df.copy()

        statistics_display["Percentage"] = (
            statistics_display["Percentage"]
            .map(lambda x: f"{x:.2f}%")
        )

        st.dataframe(
            statistics_display,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DETECTED CATEGORY SUMMARY
        # ----------------------------------------------------

        non_zero_df = statistics_df[
            statistics_df["Count"] > 0
        ].copy()

        if not non_zero_df.empty:

            st.subheader("Detected Categories")

            category_columns = st.columns(
                min(4, len(non_zero_df))
            )

            for i, (_, row) in enumerate(
                non_zero_df.iterrows()
            ):

                with category_columns[
                    i % len(category_columns)
                ]:

                    st.metric(
                        row["Category"],
                        int(row["Count"]),
                        f"{row['Percentage']:.2f}%"
                    )

        # ----------------------------------------------------
        # INDIVIDUAL DETECTION DETAILS
        # ----------------------------------------------------

        st.subheader("Individual Detection Details")

        if not detection_df.empty:

            st.dataframe(
                detection_df,
                use_container_width=True,
                hide_index=True
            )

            csv_data = dataframe_to_csv(
                detection_df
            )

            st.download_button(
                label="Download Detection Details (CSV)",
                data=csv_data,
                file_name=f"{Path(uploaded_file.name).stem}_detections.csv",
                mime="text/csv"
            )

        else:

            st.warning(
                "No products were detected in this image."
            )

        # ----------------------------------------------------
        # DOWNLOAD ANNOTATED IMAGE
        # ----------------------------------------------------

        annotated_bytes = image_to_bytes(
            annotated_image
        )

        if annotated_bytes:

            st.download_button(
                label="Download Annotated Image",
                data=annotated_bytes,
                file_name=f"{Path(uploaded_file.name).stem}_annotated.png",
                mime="image/png"
            )


    # ========================================================
    # OVERALL MULTI-IMAGE SUMMARY
    # ========================================================

    if len(uploaded_files) > 1:

        st.markdown("---")

        st.header("Overall Summary")

        overall_total = len(all_categories)

        overall_counts = Counter(
            all_categories
        )

        overall_rows = []

        for category in CLASS_NAMES:

            count = overall_counts.get(
                category,
                0
            )

            percentage = (
                (count / overall_total) * 100
                if overall_total > 0
                else 0
            )

            overall_rows.append(
                {
                    "Category": category,
                    "Count": count,
                    "Percentage": percentage
                }
            )

        overall_df = pd.DataFrame(
            overall_rows
        )

        overall_df = overall_df.sort_values(
            by=["Count", "Category"],
            ascending=[False, True]
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # OVERALL METRICS
        # ----------------------------------------------------

        metric1, metric2 = st.columns(2)

        with metric1:

            st.metric(
                "Images Processed",
                len(uploaded_files)
            )

        with metric2:

            st.metric(
                "Total Products Detected",
                overall_total
            )

        # ----------------------------------------------------
        # IMAGE-LEVEL SUMMARY
        # ----------------------------------------------------

        st.subheader("Image-wise Summary")

        image_summary_df = pd.DataFrame(
            all_image_statistics
        )

        st.dataframe(
            image_summary_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # OVERALL CATEGORY SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Overall Category-wise Statistics"
        )

        overall_display = overall_df.copy()

        overall_display["Percentage"] = (
            overall_display["Percentage"]
            .map(lambda x: f"{x:.2f}%")
        )

        st.dataframe(
            overall_display,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # OVERALL CSV
        # ----------------------------------------------------

        overall_csv = dataframe_to_csv(
            overall_df
        )

        st.download_button(
            label="Download Overall Statistics (CSV)",
            data=overall_csv,
            file_name="overall_product_statistics.csv",
            mime="text/csv"
        )

else:

    st.markdown("---")

    st.subheader("How to use the system")

    st.markdown(
        """
        1. Click **Browse files**.
        2. Select one or more supermarket/product images.
        3. Adjust the confidence threshold if required.
        4. The YOLO model detects the products.
        5. Bounding boxes and category labels are displayed.
        6. The system calculates the total number of products.
        7. Category-wise counts are generated.
        8. Product distribution percentages are calculated.
        9. Detection details and annotated images can be downloaded.
        """
    )

    st.info(
        "You can upload images from the test dataset or completely new "
        "real-world supermarket/product images."
    )