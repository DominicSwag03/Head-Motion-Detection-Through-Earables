import numpy as np
import pandas as pd
df = pd.read_csv('/content/data.csv' , header=None)
df.head()
column_names = ['Timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
name = {}

for i in range(df.shape[1]):
    name[i] = column_names[i]

print(name)
newdf = df.rename(columns=name)
newdf.head()
required_columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
dt = 0.01
angles = np.zeros((len(df), 3))
angle_x = 0.0  # Roll
angle_y = 0.0  # Pitch
angle_z = 0.0  # Yaw

alpha = 0.98

for i in range(1, len(newdf)):
    # Extract accelerometer and gyroscope data from the DataFrame
    # Make sure 'ax', 'ay', 'az', 'gx', 'gy', 'gz' are valid column names in 'newdf'
    ax, ay, az = newdf.loc[i, 'ax'], newdf.loc[i, 'ay'], newdf.loc[i, 'az']
    gx, gy, gz = newdf.loc[i, 'gx'], newdf.loc[i, 'gy'], newdf.loc[i, 'gz']

    # Calculate roll and pitch from accelerometer data
    roll_acc = np.arctan2(ay, az) * (180.0 / np.pi)
    pitch_acc = np.arctan2(-ax, np.sqrt(ay**2 + az**2)) * (180.0 / np.pi)
    # Calculate yaw from gyroscope and magnetometer data (if available)
    yaw_gyro = angle_z + gz * dt
    # Apply complementary filter to estimate roll and pitch
    angle_x = alpha * (angle_x + gx * dt) + (1 - alpha) * roll_acc
    angle_y = alpha * (angle_y + gy * dt) + (1 - alpha) * pitch_acc
    angle_z = yaw_gyro
    # Store the calculated angles
    angles[i] = [angle_x, angle_y, angle_z]

print(angles)

angles_df = pd.DataFrame(angles, columns=['angle_x', 'angle_y', 'angle_z'])
angles_df.to_csv('calculated_angles.csv', index=False)
def rotate_vector(vector, angle_x, angle_y, angle_z):
    # Rotate around x-axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])

    # Rotate around y-axis
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])

    # Rotate around z-axis
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])

    # Combined rotation
    rotated_vector = Rz @ (Ry @ (Rx @ vector))
    return rotated_vector
def check_alignment(rotated_vector):
    x, y, z = rotated_vector

    # Define a threshold
    threshold = 0.1  # You can set this based on your needs

    if abs(x) < threshold and y > threshold:
        return "Straight Ahead"
    elif x < -threshold:
        return "Left"
    elif x > threshold:
        return "Right"
    else:
        return "Undefined"  # This can mean that the direction is ambiguous
def load_and_rotate_vectors(csv_file_path):
    # Load angles from the CSV file
    df = pd.read_csv("calculated_angles.csv")
    output_data = []

    for index, row in df.iterrows():
        angle_x = np.radians(row['angle_x'])
        angle_y = np.radians(row['angle_y'])
        angle_z = np.radians(row['angle_z'])

        # Original vector (z-axis unit vector)
        vector = np.array([0, 0, 1])

        # Rotate the vector
        rotated_vector = rotate_vector(vector, angle_x, angle_y, angle_z)

        # Check alignment
        alignment = check_alignment(rotated_vector)

        # Store the output data
        output_data.append({
            'angle_x': row['angle_x'],
            'angle_y': row['angle_y'],
            'angle_z': row['angle_z'],
            'rotated_x': rotated_vector[0],
            'rotated_y': rotated_vector[1],
            'rotated_z': rotated_vector[2],
            'alignment': alignment
        })

    output_df = pd.DataFrame(output_data)

    # Save the DataFrame to a new CSV file
    output_csv_path = "output_rotated_vectors.csv"  # Change as needed
    output_df.to_csv(output_csv_path, index=False)
    print(f"Output saved to {output_csv_path}") # de-indent this line to align with the start of the function block

if __name__ == "__main__":
    csv_file_path = "path/to/your/dataset.csv"  # Update this path
    load_and_rotate_vectors(csv_file_path)

