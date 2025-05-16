import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import glob
from scipy.signal import find_peaks, peak_widths
import os
import ast 

def identify_4_points(image, gray, output_image_path):
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=60, threshold2=100)

    # Find contours in the edges image
    all_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the image for visualization
    image_with_contours = image.copy()

    # Filter and draw the selected contours on the copy
    selected_contours = []
    prev_contour = None  # To keep track of the previous contour
    coordinates = []  # List to store (x, y, w, h) tuples

    for contour in all_contours:
        # Calculate the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the bounding rectangle touches the left or right edge and has a width greater than 50 pixels
        if (x <= 5 or x + w >= image.shape[1] - 5) and w > 50:  ###10 10  20
            # If previous contour exists and has the same X, then check Y difference
            if prev_contour and x == prev_contour[0] and abs(y - prev_contour[1]) > 20:  ####5
                print("x:", x, "y:", y, "w:", w, "h:", h)
            # If the previous contour doesn't exist or X is different, print anyway
            elif not prev_contour or x != prev_contour[0]:
                print("x:", x, "y:", y, "w:", w, "h:", h)

                # Store the values of x, y, w, and h in the coordinates list
                coordinates.append((x, y, w, h))

            selected_contours.append(contour)
            cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 1)

            prev_contour = (x, y, w, h)  # Update the previous contour

    # # Display the image with selected contours using pyplot
    # plt.figure(figsize=(12, 20))  # Optional: Set the figure size
    # plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Contours')
    # plt.axis('off')  # Optional: Turn off the axis labels
    # plt.show()

    # Save the modified image with contours
    cv2.imwrite(output_image_path, image_with_contours)

    # Return the coordinates list
    return coordinates


def connect_middle_100(image, output_image_path, coordinates):
    left = []
    right = []
    image1 = image.copy()

    for x, y, w, h in coordinates:
        if x <= 10:
            left.append((x, y))
        else:
            right.append((x + w, y))

    left_top = [point for point in left if point[1] < sum(y for _, y in left) / len(left)]
    left_bottom = [point for point in left if point[1] >= sum(y for _, y in left) / len(left)]

    right_top = [point for point in right if point[1] < sum(y for _, y in right) / len(right)]
    right_bottom = [point for point in right if point[1] >= sum(y for _, y in right) / len(right)]
    print("left_top =", left_top)
    print("left_bottom = ", left_bottom)
    print("right_top = ", right_top)
    print("right_bottom = ", right_bottom)
    # Calculate the increments for dividing the sides into 100 pieces
    left_y_increment = (left_bottom[0][1] - left_top[0][1]) / 100
    right_y_increment = (right_bottom[0][1] - right_top[0][1]) / 100
    print("left_side_increament = ", left_y_increment)
    print("right_side_increament = ", right_y_increment)
    # Create an array to store all the points of the connecting lines
    connecting_points = []

    # Generate the points for connecting lines
    for i in range(100+1):
        left_point = (int(left_top[0][0]), int(left_top[0][1] + i * left_y_increment))
        right_point = (int(right_top[0][0]), int(right_top[0][1] + i * right_y_increment))
        connecting_points.append((left_point, right_point))

    # Convert the points to a NumPy array
    connecting_points = np.array(connecting_points)
    print("total number of line in middle =", len(connecting_points))
    # Draw the connecting lines on the image
    for left_point, right_point in connecting_points:
        cv2.line(image1, left_point, right_point, (0, 0, 255), 1)  # Red color (0, 0, 255)

    # Save the modified image
    # cv2.imwrite(output_image_path, image1)

    # # Display the modified image using pyplot
    # plt.figure(figsize=(48, 80))  # Optional: Set the figure size
    # plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Contours')
    # plt.axis('off')  # Optional: Turn off the axis labels
    # plt.show()
    return left_bottom, right_bottom, left_y_increment, right_y_increment,left_top, right_top
    
def connect_bottom(image, output_image_path, left_bottom, right_bottom, left_y_increment, right_y_increment):
    # Get the image dimensions
    image_height, image_width, _ = image.shape
    image2 = image.copy()
    print(image_height)

    # Create an array to store all the points of the connecting lines
    connecting_points = []

    # Determine the maximum number of iterations based on the available space
    max_iterations = int(min((image_height - left_bottom[0][1]) / left_y_increment, (image_height - right_bottom[0][1]) / right_y_increment))
    print("bottom part =", max_iterations)

    # Generate the points for connecting lines
    for i in range(max_iterations + 1):
        left_point = (int(left_bottom[0][0]), int(left_bottom[0][1] + i * left_y_increment))
        right_point = (int(right_bottom[0][0]), int(right_bottom[0][1] + i * right_y_increment))

        connecting_points.append((left_point, right_point))

    # Convert the points to a NumPy array
    connecting_points = np.array(connecting_points)

    # Draw the connecting lines on the image
    for left_point, right_point in connecting_points:
        cv2.line(image2, left_point, right_point, (0, 0, 255), 1)  # Red color (0, 0, 255)

    # Save the modified image
    cv2.imwrite(output_image_path, image2)

    # # Display the modified image using pyplot
    # plt.figure(figsize=(48, 80))  # Optional: Set the figure size
    # plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Contours')
    # plt.axis('off')  # Optional: Turn off the axis labels
    # plt.show()

def connect_top(image, output_image_path, left_top, right_top, left_y_increment, right_y_increment):
    # Get the image dimensions
    image_height, image_width, _ = image.shape
    image3 = image.copy()
    
    # Create an array to store all the points of the connecting lines
    connecting_points = []
    
    
    # Determine the maximum number of iterations based on the available space
    max_iterations = int(min(left_top[0][1]/left_y_increment , right_top[0][1]/right_y_increment))
    print("top part = ",max_iterations)
    
    # Generate the points for connecting lines
    for i in range(max_iterations+1):
        left_point = (int(left_top[0][0]), int(left_top[0][1] - i * left_y_increment))
        right_point = (int(right_top[0][0]), int(right_top[0][1] - i * right_y_increment))
    
        connecting_points.append((left_point, right_point))
    
    # Convert the points to a NumPy array
    connecting_points = np.array(connecting_points)
    
    # Draw the connecting lines
    for left_point, right_point in connecting_points:
        cv2.line(image3, left_point, right_point, (0, 0, 255), 1)  # Red color (0, 0, 255)
    
    # Save the modified image
    output_path = 'top_with_lines.jpg'
    cv2.imwrite(output_path, image3)
    
    # # Display the image with selected and first/last contours using pyplot
    # plt.figure(figsize=(48, 80))  # Optional: Set the figure size
    # plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Contours')
    # plt.axis('off')  # Optional: Turn off the axis labels
    # plt.show()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def connect_lines(image, output_image_path,coordinates):
    image4 = image.copy()
    ## to store all connecting points
    # Initialize an empty NumPy array to store all connecting points
    all_connecting_points = np.empty((0, 2, 2), dtype=int)
    top_lines = []
    bottom_lines = []
    # Get the image dimensions
    image_height, image_width, _ = image.shape

    # Left and right points categorization
    left = []
    right = []

    for x, y, w, h in coordinates:
        if x <= 10:
            left.append((x, y))
        else:
            right.append((x + w, y))

    left_top = [point for point in left if point[1] < sum(y for _, y in left) / len(left)]
    left_bottom = [point for point in left if point[1] >= sum(y for _, y in left) / len(left)]

    right_top = [point for point in right if point[1] < sum(y for _, y in right) / len(right)]
    right_bottom = [point for point in right if point[1] >= sum(y for _, y in right) / len(right)]

    # Calculate the increments for dividing the sides into 100 pieces
    left_y_increment = (left_bottom[0][1] - left_top[0][1]) / 100
    right_y_increment = (right_bottom[0][1] - right_top[0][1]) / 100
    
    #print_values
    print("left_top =", left_top)
    print("left_bottom = ", left_bottom)
    print("right_top = ", right_top)
    print("right_bottom = ", right_bottom)
    print("left_side_increament = ", left_y_increment)
    print("right_side_increament = ", right_y_increment)

    ### for center lines
    # Create an array to store all the points of the connecting lines
    connecting_points = []

    # Generate the points for connecting lines
    for i in range(100+1):
        left_point = (int(left_top[0][0]), int(left_top[0][1] + i * left_y_increment))
        right_point = (int(right_top[0][0]), int(right_top[0][1] + i * right_y_increment))
        connecting_points.append((left_point, right_point))
        all_connecting_points = np.vstack((all_connecting_points, connecting_points))

    ### for the bottom lines
    # Create an array to store all the points of the connecting lines
    connecting_points = []

    # Determine the maximum number of iterations based on the available space
    max_iterations = int(min((image_height - left_bottom[0][1])/left_y_increment, (image_height - right_bottom[0][1])/right_y_increment))

    # Generate the points for connecting lines
    for i in range(max_iterations+1):
        left_point = (int(left_bottom[0][0]), int(left_bottom[0][1] + i*left_y_increment))
        right_point = (int(right_bottom[0][0]), int(right_bottom[0][1] + i*right_y_increment))
        connecting_points.append((left_point, right_point))
        all_connecting_points = np.vstack((all_connecting_points, connecting_points))
    bottom_lines = max_iterations

    ### for top lines
    # Create an array to store all the points of the connecting lines
    connecting_points = []

    # Determine the maximum number of iterations based on the available space
    max_iterations = int(min(left_top[0][1]/left_y_increment , right_top[0][1]/right_y_increment))

    # Generate the points for connecting lines
    for i in range(max_iterations+1):
        left_point = (int(left_top[0][0]), int(left_top[0][1] - i * left_y_increment))
        right_point = (int(right_top[0][0]), int(right_top[0][1] - i * right_y_increment))
        connecting_points.append((left_point, right_point))
        all_connecting_points = np.vstack((all_connecting_points, connecting_points))
    top_lines = max_iterations

    # Draw the connecting lines on the image
    for left_point, right_point in all_connecting_points:
        cv2.line(image4, tuple(left_point), tuple(right_point), (0, 0, 255), 1)  # Red color (0, 0, 255)

    # Save the modified image
    cv2.imwrite(output_image_path, image4)

    # # Display the modified image using pyplot
    # plt.figure(figsize=(48, 80))  # Optional: Set the figure size
    # plt.imshow(cv2.cvtColor(image4, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Contours')
    # plt.axis('off')  # Optional: Turn off the axis labels
    # plt.show()    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return all_connecting_points,top_lines, bottom_lines

def connect_lines_minimum(image, output_image_path,coordinates,min_top_line,min_bottom_line):
    image4 = image.copy()
    ## to store all connecting points
    # Initialize an empty NumPy array to store all connecting points
    all_connecting_points_minimum = np.empty((0, 2, 2), dtype=int)

    # Get the image dimensions
    image_height, image_width, _ = image.shape

    # Left and right points categorization
    left = []
    right = []

    for x, y, w, h in coordinates:
        if x <= 10:
            left.append((x, y))
        else:
            right.append((x + w, y))

    left_top = [point for point in left if point[1] < sum(y for _, y in left) / len(left)]
    left_bottom = [point for point in left if point[1] >= sum(y for _, y in left) / len(left)]

    right_top = [point for point in right if point[1] < sum(y for _, y in right) / len(right)]
    right_bottom = [point for point in right if point[1] >= sum(y for _, y in right) / len(right)]

    # Calculate the increments for dividing the sides into 100 pieces
    left_y_increment = (left_bottom[0][1] - left_top[0][1]) / 100
    right_y_increment = (right_bottom[0][1] - right_top[0][1]) / 100
    
    #print_values
    print("left_top =", left_top)
    print("left_bottom = ", left_bottom)
    print("right_top = ", right_top)
    print("right_bottom = ", right_bottom)
    print("left_side_increament = ", left_y_increment)
    print("right_side_increament = ", right_y_increment)

    ### for center lines
    # Create an array to store all the points of the connecting lines
    connecting_points = []

    # Generate the points for connecting lines
    for i in range(100+1):
        left_point = (int(left_top[0][0]), int(left_top[0][1] + i * left_y_increment))
        right_point = (int(right_top[0][0]), int(right_top[0][1] + i * right_y_increment))
        connecting_points.append((left_point, right_point))
        all_connecting_points_minimum = np.vstack((all_connecting_points_minimum, connecting_points))

    ### for the bottom lines
    # Create an array to store all the points of the connecting lines
    connecting_points = []

    # Determine the maximum number of iterations based on the available space
    max_iterations_minimum = min_bottom_line

    # Generate the points for connecting lines
    for i in range(max_iterations_minimum+1):
        left_point = (int(left_bottom[0][0]), int(left_bottom[0][1] + i*left_y_increment))
        right_point = (int(right_bottom[0][0]), int(right_bottom[0][1] + i*right_y_increment))
        connecting_points.append((left_point, right_point))
        all_connecting_points_minimum = np.vstack((all_connecting_points_minimum, connecting_points))


    ### for top lines
    # Create an array to store all the points of the connecting lines
    connecting_points = []

    # Determine the maximum number of iterations based on the available space
    max_iterations_minimum = min_top_line

    # Generate the points for connecting lines
    for i in range(max_iterations_minimum+1):
        left_point = (int(left_top[0][0]), int(left_top[0][1] - i * left_y_increment))
        right_point = (int(right_top[0][0]), int(right_top[0][1] - i * right_y_increment))
        connecting_points.append((left_point, right_point))
        all_connecting_points_minimum = np.vstack((all_connecting_points_minimum, connecting_points))


    # Draw the connecting lines on the image
    for left_point, right_point in all_connecting_points_minimum:
        cv2.line(image4, tuple(left_point), tuple(right_point), (0, 0, 255), 1)  # Red color (0, 0, 255)

    # Save the modified image
    cv2.imwrite(output_image_path, image4)

    # # Display the modified image using pyplot
    # plt.figure(figsize=(48, 80))  # Optional: Set the figure size
    # plt.imshow(cv2.cvtColor(image4, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Contours')
    # plt.axis('off')  # Optional: Turn off the axis labels
    # plt.show()    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return all_connecting_points_minimum


def all_lines_contour(image, gray, output_path, all_connecting_points_minimum):
    
    # Create a copy of the original image for drawing contours
    contour_image = image.copy()
    
    # Define the lower and upper threshold for contour detection (adjust these values as needed)
    #lower_threshold = np.array([0, 0, 0])
    #upper_threshold = np.array([100, 100, 100])
  
    # Threshold the grayscale image to create a binary mask
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Create a mask for the lines (your previously drawn lines)
    line_mask = np.zeros_like(binary_mask)
    
    # Draw the lines on the line_mask
    for left_point, right_point in all_connecting_points_minimum:
        cv2.line(line_mask, left_point, right_point, 255, 2)  # Draw lines in white (255)
    
    # Subtract the line_mask from the binary_mask to exclude lines from contours
    binary_mask = cv2.subtract(binary_mask, line_mask)
    
    # Find contours in the modified binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the contours on the contour_image
    for contour in contours:
        cv2.drawContours(contour_image, [contour], -1, (0, 0, 255), 1)  # Red color (0, 0, 255)
    
    # Save the modified image with contours
    cv2.imwrite(output_path, contour_image)
    
    # # Display the image with selected and first/last contours using pyplot
    # plt.figure(figsize=(48, 80))  # Optional: Set the figure size
    # plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Contours')
    # plt.axis('off')  # Optional: Turn off the axis labels
    
    # plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def vertical_lines_detection(image, gray, output_path):
    # Apply edge detection (optional, but can improve results)
    edges = cv2.Canny(gray, 950, 700, apertureSize=5)
    
    # Apply Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 15, np.pi/500, threshold=150, minLineLength=100, maxLineGap=3)
    
    # Draw the detected lines on a copy of the original image
    line_mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if the line meets the minimum length requirement
            line_length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
            if line_length >= 100:
                # Calculate the equation of the line: y = mx + b
                m = (y2 - y1) / (x2 - x1 + 1e-6)
                b = y1 - m * x1
    
                # Calculate the y-coordinates for top and bottom of the image
                y_top = 0
                y_bottom = image.shape[0]
    #             print(y_bottom)
    
                # Check if the slope is close to zero
                x_top = int((y_top - b) / (m + 1e-6))
                if abs(m) < 1e-6:
                    x_bottom = int(x1)
                else:
                    # Calculate the x-coordinates using the equation of the line
                    
                    x_bottom = int((y_bottom - b) / (m + 1e-6))
    
    
                # Draw the line from top to bottom
                # Draw the line from top to bottom
                cv2.line(line_mask, (int(x_top), int(y_top)), (int(x_bottom), int(y_bottom)), 255, 25)  
    
    # Invert the line mask to create a mask for the background
    background_mask = cv2.bitwise_not(line_mask)
    
    # Find contours excluding the lines
    contours, _ = cv2.findContours(background_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Overlay the contours on the original image
    image_with_contours = image.copy()
    for contour in contours:
        # Calculate the width of the contour's bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if the height is less than 100 pixels then height of image
        if h > (image.shape[0] - 100):
            cv2.drawContours(image_with_contours, [contour], -1, (0, 0, 255), 2)
    
    # Save the image with highlighted lines and filtered contours
    cv2.imwrite(output_path, image_with_contours)
    
    # # Display the image with highlighted lines and filtered contours using pyplot
    # plt.figure(figsize=(12, 8))
    # plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Probabilistic Hough Lines, Vertical Lines, and Contours (Filtered)')
    # plt.axis('off')
    # plt.show()  
    return line_mask
    return image_with_contours
    
def clear_horizontal_contour(image, gray, all_connecting_points,output_path):
    # Threshold the grayscale image to create a binary mask for the contour area
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # Create a mask for the contour area
    contour_mask = np.zeros_like(binary_mask)
    for contour in contours:
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Create a mask for the lines (your previously drawn lines)
    line_mask_horizontal = np.zeros_like(binary_mask)
    
    # Draw the lines on the line_mask_horizontal
    for left_point, right_point in all_connecting_points:
        cv2.line(line_mask_horizontal, left_point, right_point, 255, 2)  # Draw lines in white (255)
    
    # Subtract the line_mask_horizontal from the contour_mask to exclude lines from the contour area
    contour_mask = cv2.subtract(contour_mask, line_mask_horizontal)
    
    # Create a masked image by applying the contour mask to the original image
    masked_image_horizontal = cv2.bitwise_and(image, image, mask=contour_mask)
    
    # Save the masked image
    cv2.imwrite(output_path, masked_image_horizontal)
    
    # # Display the image with highlighted contours using pyplot
    # plt.figure(figsize=(48, 80))  # Optional: Set the figure size
    # plt.imshow(cv2.cvtColor(masked_image_horizontal, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Highlighted Contours')
    # plt.axis('off')  # Optional: Turn off the axis labels
    # plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return masked_image_horizontal
    
def clear_vertical_contour(image, gray, line_mask, output_path):
    # Convert line_mask to uint8
    line_mask = line_mask.astype(np.uint8)
    
    # Invert the line mask to create a mask for the background
    background_mask = cv2.bitwise_not(line_mask)
    
    # Find contours excluding the lines
    contours, _ = cv2.findContours(background_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the contour area
    contour_mask = np.zeros_like(gray)  
    for contour in contours:
        # Calculate the width of the contour's bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if the width is greater than or equal to 50 pixels
        if h > (image.shape[0] - 100):
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Create a masked image by applying the contour mask to the original color image
    masked_image_vertical = cv2.bitwise_and(image, image, mask=contour_mask)
    
    # Save the masked image
    cv2.imwrite(output_path, masked_image_vertical)
    
    # # Display the image with highlighted contours using pyplot
    # plt.figure(figsize=(48, 80))  # Optional: Set the figure size
    # plt.imshow(cv2.cvtColor(masked_image_vertical, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Highlighted Contours')
    # plt.axis('off')  # Optional: Turn off the axis labels
    # plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return masked_image_vertical

    
    # Display the image with highlighted contours using pyplot
    # plt.figure(figsize=(48, 80))  # Optional: Set the figure size
    # plt.imshow(cv2.cvtColor(masked_image_vertical, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Highlighted Contours')
    # plt.axis('off')  # Optional: Turn off the axis labels
    # plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return masked_image_vertical
    
def combine_both_clear_contours(masked_image_vertical, masked_image_horizontal, output_path):
    contours_image_gray = cv2.cvtColor(masked_image_vertical, cv2.COLOR_BGR2GRAY)

    # Create a mask from the contour image
    _, mask = cv2.threshold(contours_image_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through the contours
    for i, contour in enumerate(contours):
        # Create a mask for the current contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, (255), thickness=cv2.FILLED)
     
    # Apply the inverse mask to extract the background
    vertical_and_horizontal_wiped_combined = cv2.bitwise_and(masked_image_horizontal, masked_image_horizontal, mask=mask)
    
    # # Display the image with highlighted contours using pyplot
    # plt.figure(figsize=(48, 80))  # Optional: Set the figure size
    # plt.imshow(cv2.cvtColor(vertical_and_horizontal_wiped_combined, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Highlighted Contours')
    # plt.axis('off')  # Optional: Turn off the axis labels
    # plt.show()
    
    # Save the modified image
    cv2.imwrite(output_path, vertical_and_horizontal_wiped_combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return vertical_and_horizontal_wiped_combined
    
def contours_combine_both_clear_contours(masked_image_vertical, masked_image_horizontal,vertical_and_horizontal_wiped_combined,output_path):
    # Load the contour image
    contours_image_gray = cv2.cvtColor(masked_image_vertical, cv2.COLOR_BGR2GRAY)
    #target_image_gray = cv2.cvtColor(masked_image_horizontal, cv2.COLOR_BGR2GRAY)

    image1 = masked_image_vertical.copy()
    
   
    # Create a mask from the contour image
    _, mask = cv2.threshold(contours_image_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through the contours
    for i, contour in enumerate(contours):
        # Create a mask for the current contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, (255), thickness=cv2.FILLED)
    
        # Apply the contour mask to extract the ROI from the original image
        roi = cv2.bitwise_and(masked_image_horizontal, masked_image_horizontal, mask=contour_mask)
    
        # Convert the ROI to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
        # Find contours within the ROI
        roi_contours, _ = cv2.findContours(roi_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Draw the contours found within the ROI on the original target image
        cv2.drawContours(image1, roi_contours, -1, (0, 0, 255), 2)
    
    # Display the image with highlighted contours using pyplot
    # plt.figure(figsize=(48, 80))  # Optional: Set the figure size
    # plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Highlighted Contours')
    # plt.axis('off')  # Optional: Turn off the axis labels
    # plt.show()
    
    # Save the modified image
    cv2.imwrite(output_path, image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def intensity_of_each_lane_each_box(masked_image_vertical, masked_image_horizontal,image):
    # height, width,_ = image.shape

    # Print the height and width
    # print(f"Height: {height}, Width: {width}")
    
   # contour_3_boundaries = []
    # Load the contour image
    contours_image_gray = cv2.cvtColor(masked_image_vertical, cv2.COLOR_BGR2GRAY)
    #target_image_gray = cv2.cvtColor(masked_image_horizontal, cv2.COLOR_BGR2GRAY)
    
    # Create a mask from the contour image
    _, mask = cv2.threshold(contours_image_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #define empty list for roi, counter and intensity
    roi_list = []        # List for ROI
    contour_list = []    # List for Contour
    mean_intensity_list = []  # List for Mean Intensity
    sum_intensity_list = []  # List for sum Intensity

    # Iterate through the contours
    for i, contour in enumerate(contours):
        if i < 2 or i >= len(contours) - 2:
            continue
        # Create a mask for the current contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, (255), thickness=cv2.FILLED)
            
        # Apply the contour mask to extract the ROI from the original image
        roi = cv2.bitwise_and(masked_image_horizontal, masked_image_horizontal, mask=contour_mask)
    
        # Convert the ROI to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
        # Find contours within the ROI
        roi_contours, _ = cv2.findContours(roi_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Create a copy of the ROI image for visualization
        roi_image_with_contours = roi.copy()
    
        # Draw the contours found within the ROI on the ROI image
        cv2.drawContours(roi_image_with_contours, roi_contours, -1, (0, 0, 255), 2)
    
        # Initialize lists to store intensity values and x-coordinates
        intensity_values = []
        x_coordinates = []
    
        # Calculate the intensity values and x-coordinates for each contour
        for j, roi_contour in enumerate(roi_contours):
            if j == 0 or j == len(roi_contours) - 1:
                continue

            #Mask to extract pixels within the contour
            mask_within_contour = np.zeros_like(roi_gray)
            cv2.drawContours(mask_within_contour, [roi_contour], -1, 255, thickness=cv2.FILLED)
    
            # Extract the pixels within the contour
            pixels_within_contour = cv2.bitwise_and(roi_gray, roi_gray, mask=mask_within_contour)
            pixels_within_contour = 255-pixels_within_contour
            # #pixels_within_contour = 255 - pixels_within_contour
            # print("length")
            # print(len(pixels_within_contour))
            # print("pixel with in contour")
            # print(pixels_within_contour)
            # print("minimum and maximum pixel in a range")
            # print(np.min(pixels_within_contour), np.max(pixels_within_contour))

            # print("data_type")
            # print(pixels_within_contour.dtype)

            # Get the height and width of the pixels_within_contour array
            # height, width = pixels_within_contour.shape

            # Print the height and width
            # print(f"Height: {height}, Width: {width}")


            # # # Count the number of 0 values
            # num_white_pixels = np.sum(pixels_within_contour == 255)

            # # # Count the number of pixels greater than 0
            # num_non_white_pixels = np.sum(pixels_within_contour < 255)
            # x= np.count_nonzero(mask_within_contour)
            # y = num_white_pixels + num_non_white_pixels
            # print(f"total pixel = {y}")


            # # # Print the results
            # print(f"Number of pixels with value 255: {num_white_pixels}")
            # print(f"Number of pixels with value less than 255: {num_non_white_pixels}")
            # print(f"Number of pixels within the countour: {x}")


            # print("new_intensity_mean")
            mean_intensity_new = np.mean(pixels_within_contour[pixels_within_contour < 255])
            sum_intensity_new = np.sum(pixels_within_contour[pixels_within_contour < 255])
            # print(mean_intensity_new)

            # Calculate the mean intensity of pixels within the contour
            #mean_intensity = np.mean(pixels_within_contour)
                        
            # Print the mean intensity of the contour
            # print(f"ROI {i+1}, Contour {j+1}: Mean Intensity = {mean_intensity_new}")
    
            # Append the intensity value to the list
            intensity_values.append(mean_intensity_new)
    
            # Append the x-coordinate (position within the contour) to the list
            x_coordinates.append(j)
            
            # Append the information to the respective lists
            roi_list.append(f'ROI {i+1}')
            contour_list.append(f'Contour {j+1}')
            mean_intensity_list.append(round(mean_intensity_new, 4))       
            sum_intensity_list.append(sum_intensity_new)
        #create a dataframe #.round(4)
        data_mean = {
            'ROI': roi_list,
            'Contour': contour_list,
            'Intensity': mean_intensity_list
        }
        df_mean = pd.DataFrame(data_mean)
        # print(df_mean)
        # df.to_csv(output_intensity, index=False)


        #create a dataframe #.round(4)
        data_sum = {
            'ROI': roi_list,
            'Contour': contour_list,
            'Intensity': sum_intensity_list
        }
        df_sum = pd.DataFrame(data_sum)
        # print(df_sum)
    
        # # Plot the intensity graph for the current ROI
        # plt.figure(figsize=(24, 12))  # Set the figure size
        # plt.plot(x_coordinates, intensity_values, color='blue', marker='o', linestyle='-')
        # plt.title(f'Intensity Graph for ROI {i+1}')
        # plt.xlabel('Contour Position')
        # plt.ylabel('Intensity')
        # plt.grid(True)
        # plt.show()
    
        # # Display the ROI image with its contours
        # plt.figure(figsize=(6, 6))  # Set the figure size
        # plt.imshow(cv2.cvtColor(roi_image_with_contours, cv2.COLOR_BGR2RGB))
        # plt.title(f'ROI {i+1} with Contours')
        # plt.axis('off')
        # plt.show()
    # Return the DataFrame
    return df_mean, df_sum

#define a function to sort the intensity file
def intensity_sorting(df,output_pivoted_intensity_sorted):
    print("intensity_sorting_step")
    print(df)
    # Pivot the DataFrame
    pivot_df = df.pivot(index='Contour', columns='ROI', values='Intensity')
    
    # Optionally, reset the index if you want 'Contour' to be a regular column
    pivot_df.reset_index(inplace=True)
    
    # Display the pivoted DataFrame
    print(pivot_df)
    
    # Save the pivoted DataFrame to a new CSV file
    # pivot_df.to_csv(output_pivoted_intensity, index=False)
    
    # Extract the numeric part from the 'Contour' column
    pivot_df['Contour_Num'] = pivot_df['Contour'].str.extract('(\d+)').astype(int)
    
    
    # Sort the DataFrame based on the numeric part of 'Contour'
    pivot_df_sorted = pivot_df.sort_values(by='Contour_Num')
    
    # Drop the temporary 'Contour_Num' column
    pivot_df_sorted.drop('Contour_Num', axis=1, inplace=True)
    
    # Display the sorted DataFrame
    print(pivot_df_sorted)
    
    #save the sorted_intensity as csv file
    pivot_df_sorted.to_csv(output_pivoted_intensity_sorted, index=False)
    

    
def get_contour_number(contour):
    """Extracts the numerical part of the contour for sorting."""
    return int(contour.split()[-1])

def process_and_save_csv_files(file_pattern='sample_name_*.csv'):
    """Processes each CSV file matching the pattern, sorts, transposes, and saves them."""
    csv_files = glob.glob(file_pattern)
    processed_dfs = []  # List to store processed DataFrames
    
    for file in csv_files:
        # Read CSV file into a DataFrame
        df = pd.read_csv(file)


        # Apply sorting logic
        df['Contour_Num'] = df['Contour'].apply(get_contour_number)
        df = df.sort_values(by='Contour_Num').drop(columns=['Contour_Num'])

        # Set 'Contour' column as the index
        df.set_index('Contour', inplace=True)

        # Transpose the DataFrame
        df_transposed = df.transpose()

        # Reset the index to keep the original column names
        df_transposed = df_transposed.reset_index()
       
        # Append the processed DataFrame to the list
        processed_dfs.append(df_transposed)
        
        print(f"Processed DataFrame from: {file}")
    
    return processed_dfs  # Return the list of processed DataFrames

def merge_and_sort_files(processed_dfs):
    """Merges and sorts all DataFrames from the list of processed DataFrames."""
    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(processed_dfs, ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_file = 'merged_sorted_transposed_step2.csv'
    merged_df.to_csv(merged_file, index=False)
    print(f"Merged DataFrame saved to: {merged_file}")
    
    # Sort the merged DataFrame by contour number
    merged_df["Contour_Num"] = merged_df["index"].apply(get_contour_number)
    sorted_df = merged_df.sort_values(by="Contour_Num").drop(columns=["Contour_Num"])

    # Save the final sorted DataFrame to another CSV file
    sorted_file = 'ready_for_normalization.csv'
    sorted_df.to_csv(sorted_file, index=False)
    print(f"Final sorted DataFrame saved to: {sorted_file}")

    return sorted_df

def normalization_on_merged_files(sorted_df):
    df = sorted_df
    # Get the number of columns
    num_columns = df.shape[1]
    # Initialize a new DataFrame with the first column
    df_new = pd.DataFrame(df.iloc[:, 0])
    # Loop through columns starting from the second column
    for i in range(1, num_columns):
        # Concatenate the first column with the current column
        df_new = pd.concat([df_new, df.iloc[:, 0], df.iloc[:, i]], axis=1)
    
    df = df_new
    
    # Number of columns in the original DataFrame
    num_columns = len(df.columns)
    
    # Create new DataFrames for pairs of columns
    list_of_dfs = [df.iloc[:, i:i+2] for i in range(1, num_columns, 2)]
    
    # Sort each DataFrame in list_of_dfs by the second column
    sorted_dfs = [new_df.sort_values(by=new_df.columns[1]) for new_df in list_of_dfs]
    print(sorted_dfs)
    
    # Print the sorted DataFrames
    for idx, sorted_df in enumerate(sorted_dfs):
        print(f'Sorted DataFrame {idx + 1}:')
        print(sorted_df)
        print('\n')
    
    
    # Extract the second column for each DataFrame and stack them horizontally
    second_columns = [df.iloc[:, 1] for df in sorted_dfs]
    stacked_columns = np.column_stack(second_columns)
    
    # Calculate the row-wise average
    averages = np.round(np.mean(stacked_columns, axis=1), 3)
    
    # Create a new DataFrame with the averages
    average_df = pd.DataFrame({'Average': averages})
    
    # Print the resulting DataFrame
    print(average_df)
    print("###############")
    print(sorted_df)
    
    # Iterate through sorted_dfs
    for idx, sorted_df in enumerate(sorted_dfs):
        # Find the column starting with 'ROI'
        roi_column = [col for col in sorted_df.columns if col.startswith('Contour ')][0]
    
        # Replace 'ROI' column values with corresponding average values
        sorted_df[roi_column] = averages
    
    # Print the modified DataFrames
    for idx, sorted_df in enumerate(sorted_dfs):
        print(f'Modified DataFrame {idx + 1}:')
        print(sorted_df)
        print('\n')
    
    # Concatenate all the modified DataFrames along columns
    combined_df = pd.concat(sorted_dfs, axis=1)
    
    # Keep only one "Contour" column and remove the others
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    # Print the modified DataFrame
    print(combined_df)
    
    # Save the sorted DataFrame to a new CSV file
    combined_df.to_csv('normalized_df.csv', index=False)
    return combined_df

def ready_for_ml(combined_df,classification_file):
    df= combined_df

    # Filter rows where the 'index' column starts with 'val'
    val_df = df[df['index'].str.startswith('val')]
    
    # Filter rows where the 'index' column does not start with 'val'
    non_val_df = df[~df['index'].str.startswith('val')]
    
    # Save each subset to separate CSV files
    val_df.to_csv('normalized_validation.csv', index=False)
    non_val_df.to_csv('normalized_samples.csv', index=False)
    
    df2 = non_val_df
    df1 = classification_file
    print(df1)
    
    # Remove duplicates and handle missing values in df2
    df2 = df2.drop_duplicates(subset='index').dropna(subset=['index'])
    print(df2)
    
    
    # Create a new "cluster" column in df2 and assign corresponding values from df1
    df2['cluster'] = df2['index'].map(df1.set_index('sample')['cluster'])
    
    # Drop rows where 'cluster' has NaN values
    df2 = df2.dropna(subset=['cluster'])
    
    # Convert the 'ok' column values to integers
    df2['cluster'] = df2['cluster'].astype(int)
    
    # Print the modified df2
    print(df2)

        # Extract the numeric part from the "index" column
    df2['sort_key'] = df2['index'].str.extract('(\d+)').astype(int)

    # Sort the DataFrame by the numeric part
    df2_sorted = df2.sort_values(by='sort_key')

    # Drop the temporary "sort_key" column (optional)
    df2_sorted = df2_sorted.drop(columns=['sort_key'])
    
    df2_sorted.to_csv('ready_for_ml_class_1.csv', index=False)

   # functions.py



# Function to calculate adaptive parameters based on intensity values
def calculate_adaptive_parameters(intensity_values):
    mean_intensity = intensity_values.mean()
    std_intensity = intensity_values.std()
    data_length = len(intensity_values)
    
    # Adaptive parameters
    smoothing_window = max(5, min(data_length // 100, 15))  # Window size as a fraction of data points
    height_thresh = 0.005 * mean_intensity  # 0.5% of the mean intensity
    prominence_thresh = 0.01 * std_intensity  # 1% of the standard deviation
    distance = max(1, data_length // 150)  # Spacing proportional to data length
    
    return smoothing_window, height_thresh, prominence_thresh, distance

# Function to normalize intensity values using Min-Max normalization
def normalize_intensity(intensity_values):
    min_val = intensity_values.min()
    max_val = intensity_values.max()
    normalized_values = (intensity_values - min_val) / (max_val - min_val)
    return normalized_values

# Function to process intensity data for a single lane
def process_lane(lane_name, intensity_values):
    # Calculate adaptive parameters
    smoothing_window, height_thresh, prominence_thresh, distance = calculate_adaptive_parameters(intensity_values)
    
    # Smooth the intensity data
    smoothed_intensity = intensity_values.rolling(window=smoothing_window, min_periods=1).mean()
    
    # Detect peaks
    peaks, properties = find_peaks(
        smoothed_intensity, 
        height=height_thresh, 
        prominence=prominence_thresh, 
        distance=distance
    )
    
    # Measure peak widths at half prominence
    widths_results = peak_widths(smoothed_intensity, peaks, rel_height=0.5)
    left_bases = np.floor(widths_results[2]).astype(int)
    right_bases = np.ceil(widths_results[3]).astype(int)
    
    return smoothed_intensity, peaks, properties, left_bases, right_bases


def filter_close_ranges(contour_ranges, gap_threshold):
    filtered_ranges = []
    for i in range(len(contour_ranges)):
        if i == 0:  # Always keep the first range
            filtered_ranges.append(contour_ranges[i])
        else:
            prev_range = filtered_ranges[-1]
            current_range = contour_ranges[i]
            
            # Check the gap between the current range and the previous one
            gap = current_range[0] - prev_range[1]
            if gap < gap_threshold:
                # Compare lengths and keep the longer range
                prev_length = prev_range[1] - prev_range[0]
                current_length = current_range[1] - current_range[0]
                if current_length > prev_length:
                    filtered_ranges[-1] = current_range  # Replace with the current range
            else:
                filtered_ranges.append(current_range)  # No overlap, keep the current range

    return filtered_ranges

# Function to filter common ranges across ROIs
def filter_common_ranges_keep_original(contour_ranges_all, tolerance=20):
    lanes = list(contour_ranges_all.keys())
    if not lanes:
        return {}

    reference_ranges = contour_ranges_all[lanes[0]]
    common_ranges = []
    for ref_range in reference_ranges:
        is_common = all(
            any(is_within_tolerance(ref_range, lane_range, tolerance) for lane_range in contour_ranges_all[lane])
            for lane in lanes[1:]
        )
        if is_common:
            common_ranges.append(ref_range)

    filtered_ranges = {lane: [] for lane in lanes}
    for lane in lanes:
        for original_range in contour_ranges_all[lane]:
            if any(is_within_tolerance(original_range, common_range, tolerance) for common_range in common_ranges):
                filtered_ranges[lane].append(original_range)

    return filtered_ranges

# Helper function to check if two ranges are within a tolerance
def is_within_tolerance(range1, range2, tolerance):
    return (
        abs(range1[0] - range2[0]) <= tolerance and
        abs(range1[1] - range2[1]) <= tolerance
    )




def contours_of_choice(image,image_name,masked_image_vertical, masked_image_horizontal,contour_index_df, output_path):

    contour_boundaries = []
    # print(contour_index_df)
    # Specify the row index you want to process
    row_to_process = 0  # For example, process the third row (index 2)
    # Load the contour image
    contours_image_gray = cv2.cvtColor(masked_image_vertical, cv2.COLOR_BGR2GRAY)

    # Create a mask from the contour image
    _, mask = cv2.threshold(contours_image_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    

    # Iterate through the contours
    for i, contour in enumerate(contours):

        if i < 2 or i >= len(contours) - 2:
            continue
        if image_name == "101-112" and i == 13:
            continue
        if image_name == "126-136" and i == 7:
            continue
        if image_name == "validation_samples_1-10" and i in [14,8,7]:
            continue

        
   
        # Create a mask for the current contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, (255), thickness=cv2.FILLED)
            
        # Apply the contour mask to extract the ROI from the original image
        roi = cv2.bitwise_and(masked_image_horizontal, masked_image_horizontal, mask=contour_mask)
    
        # Convert the ROI to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
        # Find contours within the ROI
        roi_contours, _ = cv2.findContours(roi_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Create a copy of the ROI image for visualization
        roi_image_with_contours = roi.copy()
    
        # Draw the contours found within the ROI on the ROI image
        cv2.drawContours(roi_image_with_contours, roi_contours, -1, (0, 0, 255), 2)
    


        # Counter for alternating colors
        color_counter = 0

        # Loop through the DataFrame
        for index, row in contour_index_df.iterrows():
            if index == row_to_process:  # Check if it's the desired row
                contour_index = ast.literal_eval(row['Ranges'])
                # print(f"Processing row {index}: {contour_index}")
                # print(contour_index)
        row_to_process = row_to_process +1
                
        
        # Iterate through each list in contour_index
        for inner_list in contour_index:
            # Initialize the list to store contour boundaries for the current list
            contour_boundaries = []
        
            # Calculate the intensity values and x-coordinates for each contour in the current list
            for j, roi_contour in enumerate(roi_contours):
                # Check if j is in the current inner list within contour_index
                if isinstance(inner_list, list) and j in inner_list:
                    contour_boundaries.append(roi_contour)
        
            # Check if color_counter is less than the length of contour_index
            if color_counter < len(contour_index):
                # Determine the color based on the counter (0 for green, 1 for red)
                color = (0, 255, 0) if color_counter % 2 == 0 else (0, 0, 255)
        
                # Draw contours for the current list on a single image with alternating colors
                cv2.drawContours(image, contour_boundaries, -1, color, 2)
        
                # Increment the color counter only once for each inner list
                color_counter += 1
        
        contour_index =[]
        contour_boundaries = []
  
       
    # Display the image with all contour
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB if the image is in color
    plt.title('Drawing merged contours')
    plt.axis('off')  # Turn off axis labels
    plt.show()
    
    
    # Save the modified image
    cv2.imwrite(output_path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()