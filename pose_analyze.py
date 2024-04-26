import numpy as np
from scipy.signal import find_peaks
import math

is_standing = True  # Initialize the global state

def calculate_angle(a, b, c):
    """Calculate the angle between the vectors from a to b and from b to c."""
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def is_person_standing(person, previous_nose_y):
    """Determine whether a person is standing based on the keypoints."""
    # Get the keypoints
    nose = person.keypoints[0]
    left_knee = person.keypoints[13]
    right_knee = person.keypoints[14]
    left_ankle = person.keypoints[15]
    right_ankle = person.keypoints[16]

    # Define the threshold for the y coordinate change
    threshold = ((left_knee.y - left_ankle.y) + (right_knee.y - right_ankle.y)) // 2

    # If the y coordinates of the knees are higher than the ankles and the y coordinate of the nose and knees are higher than the previous ones by a certain threshold, the person is standing
    if nose.y < previous_nose_y - threshold:
        return True
    else:
        return False

def squat_count(list_persons_history):
  # Initialize a list to store the y coordinates of the nose keypoint
  nose_y_coordinates = []
  knee_y_coordinates = []  # Initialize a list to store the y coordinates of the knee keypoints
  action_count = 0
  correction_info = 'No correction info'
  previous_nose_y = None  # Initialize the previous nose y coordinate
  previous_knee_y = None  # Initialize the previous knee y coordinate

  # For each pose in the history
  for list_persons in list_persons_history:
    # Get the person
    person = list_persons[0]

    # If the person is performing the target action
    # if person.action == action:
    nose = person.keypoints[0]
    nose_y_coordinates.append(nose.y)

    # Save the y coordinate of the knees
    left_knee = person.keypoints[13]
    right_knee = person.keypoints[14]
    knee_y_coordinates.append((left_knee.y + right_knee.y) / 2)

    # If we have enough data to analyze
    if len(nose_y_coordinates) > 2:
      # Find the valleys in the last three y coordinates
      valleys, _ = find_peaks(-np.array(nose_y_coordinates[-3:]))

      # If a valley is found, increment the count and analyze the squat
      if len(valleys) > 0 and is_standing:
        action_count += 1
        previous_nose_y = nose.y
        # Analyze the squat here
        # Calculate the angles of the knees and hips
        left_knee_angle = calculate_angle(person.keypoints[11], person.keypoints[13], person.keypoints[15])
        right_knee_angle = calculate_angle(person.keypoints[12], person.keypoints[14], person.keypoints[16])
        hip_angle = calculate_angle(person.keypoints[5], person.keypoints[11], person.keypoints[13])

        # If the angles are not within a certain range, add a correction suggestion to the correction_info
        if not (80 <= left_knee_angle <= 100 and 80 <= right_knee_angle <= 100 and 80 <= hip_angle <= 100):
          if left_knee_angle < 80 or right_knee_angle < 80 or hip_angle < 80:
            correction_info = 'You are squatting too low.'
          else:
            correction_info = 'You are squatting too high.'
        list_persons_history.clear()
      elif is_person_standing(person, previous_nose_y):
        is_standing = True


  return action_count, correction_info

def analyze(list_persons_history, action):
  """Analyze the pose history and count the number of actions.

  Args:
    list_persons_history: A deque containing the history of poses.
    action: The target action to be detected.

  Returns:
    action_count: The number of times the action has been performed.
    correction_info: Information about how to correct the action.
  """
  action_count = 0
  correction_info = 'No correction info'

  if action == "squat":
    action_count, correction_info = squat_count(list_persons_history)
  return action_count, correction_info