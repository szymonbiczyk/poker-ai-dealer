import cv2


def preprocess_for_card_contour(image):
    # Convert the image into clean edges for contour detection.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    return edges


def approximate_card_contour(edges):
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return contours, None, None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    return contours, largest_contour, contour_area, perimeter, approx
