from math import sin, cos, tan, sqrt, pi, asin
import random
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
        self.TOTAL_DISTANCE = sum(self.DISTANCES)
        self.TARGET_Y = target_y 
        self.y_coords = []

    def trace_path(self, theta):
        """
        Trace the path of the light ray given initial angle theta
        Returns (y_final, total_time)
        """
        total_time = 0
        y = 0
        self.y_coords = []
        for i in range(len(self.SPEEDS)):
            y += self.DISTANCES[i] * tan(theta)
            total_time += (self.DISTANCES[i] / cos(theta)) / self.SPEEDS[i]
            self.y_coords.append(y)
            if i < len(self.SPEEDS) - 1:
                # domain check
                if abs(self.SPEEDS[i+1] * sin(theta) / self.SPEEDS[i]) > 1:
                    return (float('inf') if theta > 0 else float('-inf'), float('inf'))
                theta = asin(self.SPEEDS[i+1] * sin(theta) / self.SPEEDS[i])

        return (y, total_time)

    def find_optimal_angle(self):
        """
        Binary search to find theta that hits the target (x=50√2, y=50√2)
        θ = 0 is perpendicular to marsh, increases counterclockwise
        """
        low = -pi/2
        high = pi/2
        while high - low > 1e-14:
            print(f"low: {low}, high: {high}")
            mid = (low + high) / 2
            y_final, total_time = self.trace_path(mid)
            if y_final < self.TARGET_Y:
                low = mid
            else:
                high = mid
        return mid, total_time, self.y_coords


def main():
        
    # Example usage with the original configuration
    original_sections = [
        ((50.0 * sqrt(2) - 50)/2, 10),  # First dry section
        (10, 9),                         # First marsh section
        (10, 8),                         # Second marsh section
        (10, 7),                         # Third marsh section
        (10, 6),                         # Fourth marsh section
        (10, 5),                         # Fifth marsh section
        ((50.0 * sqrt(2) - 50)/2, 10)   # Final dry section
    ]
    original_target_y = 50.0 * sqrt(2)

    # generate a random map with 1000 sections
    random_sections = [(random.randint(1, 100), random.randint(1, 10)) for _ in range(100)]
    random_target_y = random.randint(1, 10000)


    optimizer = MarshOptimizerSnell(random_sections, random_target_y)
    #optimizer = MarshOptimizerSnell(original_sections, original_target_y)
    optimal_theta, optimal_time, y_coords = optimizer.find_optimal_angle()
    print(f"Optimal angle: {optimal_theta:.10f} radians")
    print(f"Optimal angle: {optimal_theta * 180/pi:.10f} degrees") 
    print(f"Optimal time: {optimal_time:.10f} days")
    print(f"Boundary points: {y_coords}")

if __name__ == "__main__":
    main()
