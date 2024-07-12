import numpy as np
import math

def get_concave_direction(points):
    vectors = np.diff(points, axis=0)  
    cross_products = np.cross(vectors[:-1], vectors[1:])  


    if np.sum(cross_products) < 0:
      
        concave_parts = cross_products < 0
    else:

        concave_parts = cross_products > 0

    concave_vectors = vectors[:-1][concave_parts]  

    if len(concave_vectors) == 0:
        return None 

    average_vector = np.mean(concave_vectors, axis=0) 
    unit_vector = average_vector / np.linalg.norm(average_vector)  

    return unit_vector.tolist()

def calculate_unit_normal_vector(points):
    unit_normal_vectors = []
    for i in range(len(points)):

        vector1 = (points[(i+1)%len(points)][0] - points[i][0], points[(i+1)%len(points)][1] - points[i][1])
        vector2 = (points[i][0] - points[(i-1)%len(points)][0], points[i][1] - points[(i-1)%len(points)][1])
        

        normal_vector = (vector1[1] + vector2[1], -(vector1[0] + vector2[0]))

        magnitude = math.sqrt(normal_vector[0]**2 + normal_vector[1]**2)

        unit_normal_vector = (normal_vector[0]/magnitude, normal_vector[1]/magnitude) if magnitude else (0, 0)
        
        unit_normal_vectors.append(unit_normal_vector)
    return unit_normal_vectors

def calculate_vectors_sum(vectors):
    vector_sum = [0, 0]
    for vector in vectors:
        vector_sum[0] += vector[0]
        vector_sum[1] += vector[1]
    return tuple(vector_sum)

def calculate_angle(vector1, vector2):
    dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    

    cos_theta = max(min(cos_theta, 1), -1)
    
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


points1 = np.array([(32, 13), (92, 22), (44,149), (44, 128),(10,32)])  
points2 = np.array([(10, 32), (44, 128), (44,149), (92, 22),(32,13)])  

unit_normal_vectors1 = calculate_unit_normal_vector(points1)
unit_normal_vectors2 = calculate_unit_normal_vector(points2)
vector_sum1 = calculate_vectors_sum(unit_normal_vectors1[1:-1])
vector_sum2 = calculate_vectors_sum(unit_normal_vectors2[1:-1])

angle = calculate_angle(vector_sum1, vector_sum2)
print(angle)

