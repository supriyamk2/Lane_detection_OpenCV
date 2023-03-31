import cv2
import numpy as np
import streamlit as st
from PIL import Image


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (width, height),
        (width//2, height//2)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges


def draw_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return lines_image


def combine_images(image, lines_image):
    return cv2.addWeighted(image, 0.8, lines_image, 1, 1)


def main():
    st.title("Lane Detection")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image = np.array(image)
            
            edges = process_image(image)
            roi = region_of_interest(edges)
            lines = cv2.HoughLinesP(roi, 2, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=100)
            lines_image = draw_lines(image, lines)
            result = combine_images(image, lines_image)

            col1, col2 = st.columns(2)
            col1.header("Original Image")
            col1.image(image, use_column_width=True)
            col2.header("Lane Detection")
            col2.image(result, use_column_width=True)
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
