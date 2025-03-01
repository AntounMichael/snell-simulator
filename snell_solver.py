from math import sin, cos, tan, sqrt, pi, asin, atan2
import time
import threading

# Thread-based timeout instead of signal-based
class TimeoutError(Exception):
    """Exception raised when a function times out."""
    pass

def timeout_handler(function, args=(), kwargs={}, timeout_duration=10, default=None):
    """
    Run a function with a timeout.
    
    Args:
        function: The function to run
        args: The function arguments
        kwargs: The function keyword arguments
        timeout_duration: Timeout in seconds
        default: Default value to return on timeout
        
    Returns:
        The function result or the default value on timeout
    """
    result = [default]
    exception = [None]
    
    def worker():
        try:
            result[0] = function(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        return default, TimeoutError("Function timed out")
    if exception[0]:
        return default, exception[0]
    return result[0], None

class MarshOptimizerSnell:
    def __init__(self, sections, target_y):
        """
        Initialize with a list of (distance, speed) tuples
        sections: [(distance1, speed1), (distance2, speed2), ...]
        """
        # Unzip the sections into separate lists
        self.DISTANCES, self.SPEEDS = zip(*sections)
        
        # Convert to lists for easier manipulation
        self.DISTANCES = list(self.DISTANCES)
        self.SPEEDS = list(self.SPEEDS)
        
        # Calculate total distance and target Y
        self.TARGET_Y = target_y 
        self.y_coords = []
        self.x_coords = [0]  # Start at x=0
        
        # Calculate x_coords based on distances
        cumulative_x = 0
        for dist in self.DISTANCES:
            cumulative_x += dist
            self.x_coords.append(cumulative_x)

    def trace_path(self, theta):
        """
        Trace the path of the light ray given initial angle theta
        Returns (y_final, total_time, path_points, segment_data)
        """
        total_time = 0
        y = 0
        self.y_coords = [0]  # Start at y=0
        x = 0
        path_points = [(0, 0)]  # Start point
        
        # Create lists to store segment-specific data
        segment_distances = []
        segment_times = []
        
        current_theta = theta  # Store initial theta
        
        for i in range(len(self.SPEEDS)):
            distance = self.DISTANCES[i]
            speed = self.SPEEDS[i]
            
            # Calculate segment distance (actual path length, not just x distance)
            segment_length = distance / cos(current_theta)
            segment_distances.append(segment_length)
            
            # Calculate time for this segment
            segment_time = segment_length / speed
            segment_times.append(segment_time)
            total_time += segment_time
            
            # Calculate y increase for this segment
            y_increase = distance * tan(current_theta)
            y += y_increase
            x += distance
            
            # Add the current point to the path
            path_points.append((x, y))
            self.y_coords.append(y)
            
            # Update theta using Snell's law for the next segment
            if i < len(self.SPEEDS) - 1:
                # domain check
                sin_ratio = self.SPEEDS[i+1] * sin(current_theta) / self.SPEEDS[i]
                if abs(sin_ratio) > 1:
                    # Create a proper segment_data dictionary for error cases
                    empty_segment_data = {
                        "distances": segment_distances,
                        "times": segment_times,
                        "total_distance": float('inf'),
                        "total_time": float('inf')
                    }
                    return (float('inf') if current_theta > 0 else float('-inf'), 
                            float('inf'), path_points, empty_segment_data)
                current_theta = asin(sin_ratio)
        
        # Create segment data dictionary with all the information we need
        segment_data = {
            "distances": segment_distances,
            "times": segment_times,
            "total_distance": sum(segment_distances),
            "total_time": total_time
        }
        
        return (y, total_time, path_points, segment_data)

    def find_optimal_angle(self, max_seconds=10):
        """
        Binary search to find theta that hits the target
        θ = 0 is perpendicular to marsh, increases counterclockwise
        
        Parameters:
        max_seconds (int): Maximum time in seconds to run the computation
        
        Returns:
        tuple: (optimal_angle, total_time, path_points, segment_data)
        """
        def binary_search():
            low = -pi/2
            high = pi/2
            best_path_points = []
            best_segment_data = {}
            
            # Add a counter to limit iterations
            max_iterations = 1000
            iteration = 0
            
            while high - low > 1e-14 and iteration < max_iterations:
                mid = (low + high) / 2
                y_final, total_time, path_points, segment_data = self.trace_path(mid)
                best_path_points = path_points
                best_segment_data = segment_data
                
                if y_final < self.TARGET_Y:
                    low = mid
                else:
                    high = mid
                
                iteration += 1
                
            # Final trace with the optimal angle to get accurate path
            _, total_time, best_path_points, best_segment_data = self.trace_path(mid)
            
            return mid, total_time, best_path_points, best_segment_data
        
        # Run the binary search with a timeout
        result, error = timeout_handler(binary_search, timeout_duration=max_seconds)
        
        if error:
            print(f"Calculation error: {error}")
            return None, None, None, None
        
        return result

def calculate_straight_path(sections, target_y):
    """Calculate metrics for a straight-line path from (0,0) to (total_x, target_y)"""
    # Calculate total x distance
    total_x = sum(distance for distance, _ in sections)
    
    # Calculate straight-line distance
    straight_distance = sqrt(total_x**2 + target_y**2)
    
    # Calculate time spent in each section
    straight_time = 0
    
    # Calculate the angle of the straight line
    straight_angle = 0 if total_x == 0 else atan2(target_y, total_x)
    
    # Current position
    current_x = 0
    current_y = 0
    
    for i, (distance, speed) in enumerate(sections):
        # Calculate intersection with next section boundary
        next_x = current_x + distance
        
        # Calculate y at the intersection based on the straight line equation
        # y = mx + b where m = target_y / total_x and b = 0
        next_y = target_y * next_x / total_x if total_x > 0 else 0
        
        # Calculate distance traveled in this section
        section_distance = sqrt((next_x - current_x)**2 + (next_y - current_y)**2)
        
        # Calculate time spent in this section
        section_time = section_distance / speed
        straight_time += section_time
        
        # Update current position
        current_x = next_x
        current_y = next_y
    
    return {
        "straight_distance": straight_distance,
        "straight_time": straight_time,
        "straight_angle": straight_angle,
        "straight_avg_speed": straight_distance / straight_time if straight_time > 0 else 0
    } 