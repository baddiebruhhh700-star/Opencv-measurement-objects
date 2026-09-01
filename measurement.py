import cv2
import numpy as np
import csv
import os
import math


# ============================================================
# 1. SETTINGS
# ============================================================

IMAGE_PATH = r"C:\Users\DELL\Downloads\tools count.jfif"

OUTPUT_FOLDER = r"C:\Users\DELL\Documents\perception-folder\output"

MIN_AREA = 2000

THRESHOLD_VALUE = 200


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 3. LOAD IMAGE
# ============================================================

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(
        f"Could not load image:\n{IMAGE_PATH}"
    )

print("Image loaded successfully")
print("Image shape:", img.shape)


# ============================================================
# 4. GRAYSCALE
# ============================================================

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# 5. SEGMENTATION
# ============================================================

_, mask = cv2.threshold(
    gray,
    THRESHOLD_VALUE,
    255,
    cv2.THRESH_BINARY_INV
)


# ============================================================
# 6. MORPHOLOGICAL CLEANUP
# ============================================================

kernel = np.ones(
    (5, 5),
    dtype=np.uint8
)

# Remove small noise
mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

# Close small gaps
mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)


# ============================================================
# 7. FIND CONTOURS
# ============================================================

contours, hierarchy = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Contours found:", len(contours))


# ============================================================
# 8. CREATE OUTPUT IMAGE
# ============================================================

output = img.copy()

object_count = 0


# ============================================================
# 9. CREATE CSV FILE
# ============================================================

CSV_FILE = os.path.join(
    OUTPUT_FOLDER,
    "measurements.csv"
)

csv_file = open(
    CSV_FILE,
    "w",
    newline=""
)

csv_writer = csv.writer(
    csv_file
)

csv_writer.writerow([
    "Object_ID",
    "Area_px2",
    "Perimeter_px",
    "Centroid_X",
    "Centroid_Y",
    "Bounding_X",
    "Bounding_Y",
    "Width_px",
    "Height_px",
    "Circularity",
    "Hull_Area_px2",
    "Solidity"
])


# ============================================================
# 10. PROCESS EACH CONTOUR
# ============================================================

for contour in contours:

    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    area = cv2.contourArea(contour)

    if area < MIN_AREA:
        continue

    object_count += 1


    # --------------------------------------------------------
    # PERIMETER
    # --------------------------------------------------------

    perimeter = cv2.arcLength(
        contour,
        True
    )


    # --------------------------------------------------------
    # CENTROID
    # --------------------------------------------------------

    M = cv2.moments(contour)

    if M["m00"] != 0:

        cx = int(
            M["m10"] / M["m00"]
        )

        cy = int(
            M["m01"] / M["m00"]
        )

    else:

        cx = 0
        cy = 0


    # --------------------------------------------------------
    # BOUNDING BOX
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        contour
    )


    # --------------------------------------------------------
    # CIRCULARITY
    # --------------------------------------------------------

    if perimeter > 0:

        circularity = (
            4 * math.pi * area
        ) / (
            perimeter ** 2
        )

    else:

        circularity = 0


    # --------------------------------------------------------
    # CONVEX HULL
    # --------------------------------------------------------

    hull = cv2.convexHull(
        contour
    )


    # --------------------------------------------------------
    # HULL AREA
    # --------------------------------------------------------

    hull_area = cv2.contourArea(
        hull
    )


    # --------------------------------------------------------
    # SOLIDITY
    # --------------------------------------------------------

    if hull_area > 0:

        solidity = (
            area / hull_area
        )

    else:

        solidity = 0


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n------------------------------")

    print(
        "Object:",
        object_count
    )

    print(
        "Area:",
        round(area, 2)
    )

    print(
        "Perimeter:",
        round(perimeter, 2)
    )

    print(
        "Centroid:",
        (cx, cy)
    )

    print(
        "Bounding box:",
        (x, y, w, h)
    )

    print(
        "Width:",
        w
    )

    print(
        "Height:",
        h
    )

    print(
        "Circularity:",
        round(circularity, 4)
    )

    print(
        "Hull area:",
        round(hull_area, 2)
    )

    print(
        "Solidity:",
        round(solidity, 4)
    )


    # ========================================================
    # SAVE TO CSV
    # ========================================================

    csv_writer.writerow([
        object_count,
        round(area, 2),
        round(perimeter, 2),
        cx,
        cy,
        x,
        y,
        w,
        h,
        round(circularity, 4),
        round(hull_area, 2),
        round(solidity, 4)
    ])


    # ========================================================
    # ANNOTATION 1 — GREEN CONTOUR
    # ========================================================

    cv2.drawContours(
        output,
        [contour],
        -1,
        (0, 255, 0),
        2
    )


    # ========================================================
    # ANNOTATION 2 — MAGENTA CONVEX HULL
    # ========================================================

    cv2.polylines(
        output,
        [hull],
        True,
        (255, 0, 255),
        2
    )


    # ========================================================
    # ANNOTATION 3 — BLUE BOUNDING BOX
    # ========================================================

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )


    # ========================================================
    # ANNOTATION 4 — RED CENTROID
    # ========================================================

    cv2.circle(
        output,
        (cx, cy),
        6,
        (0, 0, 255),
        -1
    )


    # ========================================================
    # ANNOTATION 5 — OBJECT NUMBER
    # ========================================================

    cv2.putText(
        output,
        f"Object {object_count}",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # ========================================================
    # ANNOTATION 6 — AREA
    # ========================================================

    cv2.putText(
        output,
        f"Area: {area:.0f}",
        (x, y + h + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


# ============================================================
# 11. TOTAL OBJECT COUNT
# ============================================================

print("\n====================================")
print("TOTAL OBJECTS:", object_count)
print("====================================")


# ============================================================
# 12. DRAW TOTAL OBJECT COUNT
# ============================================================

cv2.putText(
    output,
    f"TOTAL OBJECTS: {object_count}",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (0, 255, 255),
    3
)


# ============================================================
# 13. SAVE ORIGINAL IMAGE
# ============================================================

original_path = os.path.join(
    OUTPUT_FOLDER,
    "original.jpg"
)

cv2.imwrite(
    original_path,
    img
)


# ============================================================
# 14. SAVE MASK IMAGE
# ============================================================

mask_path = os.path.join(
    OUTPUT_FOLDER,
    "mask.jpg"
)

cv2.imwrite(
    mask_path,
    mask
)


# ============================================================
# 15. SAVE ANNOTATED IMAGE
# ============================================================

annotated_path = os.path.join(
    OUTPUT_FOLDER,
    "annotated_objects.jpg"
)

cv2.imwrite(
    annotated_path,
    output
)


# ============================================================
# 16. CLOSE CSV
# ============================================================

csv_file.close()


# ============================================================
# 17. DISPLAY IMAGES
# ============================================================

cv2.imshow(
    "Original",
    img
)

cv2.imshow(
    "Mask",
    mask
)

cv2.imshow(
    "Annotated Objects",
    output
)

cv2.waitKey(0)

cv2.destroyAllWindows()


# ============================================================
# 18. SHOW SAVED FILE LOCATIONS
# ============================================================

print("\n====================================")
print("FILES SAVED SUCCESSFULLY")
print("====================================")

print("\nOriginal:")
print(original_path)

print("\nMask:")
print(mask_path)

print("\nAnnotated:")
print(annotated_path)

print("\nCSV:")
print(CSV_FILE)
