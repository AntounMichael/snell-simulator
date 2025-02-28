import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from math import sin, cos, tan, sqrt, pi, asin
import random
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Import your existing Snell's law solver code
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
                    return (float('inf') if current_theta > 0 else float('-inf'), 
                            float('inf'), path_points, [], [])
                current_theta = asin(sin_ratio)
        
        # Create segment data dictionary with all the information we need
        segment_data = {
            "distances": segment_distances,
            "times": segment_times,
            "total_distance": sum(segment_distances),
            "total_time": total_time
        }
        
        return (y, total_time, path_points, segment_data)

    def find_optimal_angle(self):
        """
        Binary search to find theta that hits the target
        θ = 0 is perpendicular to marsh, increases counterclockwise
        """
        low = -pi/2
        high = pi/2
        best_path_points = []
        best_segment_data = {}
        
        while high - low > 1e-14:
            mid = (low + high) / 2
            y_final, total_time, path_points, segment_data = self.trace_path(mid)
            best_path_points = path_points
            best_segment_data = segment_data
            
            if y_final < self.TARGET_Y:
                low = mid
            else:
                high = mid
                
        # Final trace with the optimal angle to get accurate path
        _, total_time, best_path_points, best_segment_data = self.trace_path(mid)
        
        return mid, total_time, best_path_points, best_segment_data

def draw_path_animation(sections, target_y, animation_speed):
    """Create and display an animation of the optimal path"""
    optimizer = MarshOptimizerSnell(sections, target_y)
    optimal_theta, optimal_time, path_points, segment_data = optimizer.find_optimal_angle()
    
    # Extract x and y coordinates for plotting
    x_vals = [p[0] for p in path_points]
    y_vals = [p[1] for p in path_points]
    
    # Create figure and axis
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Calculate section boundaries
    x_boundaries = [0]
    cumulative_x = 0
    for distance, _ in sections:
        cumulative_x += distance
        x_boundaries.append(cumulative_x)
    
    # Plot section boundaries and color the sections
    for i in range(len(sections)):
        x_start = x_boundaries[i]
        x_end = x_boundaries[i+1]
        speed = sections[i][1]
        # Normalize speed to get a color (red for slow, blue for fast)
        # Assuming speeds are between 1-10
        color_val = 1 - (speed / 10.0)  # Closer to 1 means slower/more red
        ax.fill_between([x_start, x_end], [0, 0], [target_y, target_y], 
                       color=(color_val, 0.5, 1-color_val, 0.3), 
                       label=f"Speed: {speed}")
    
    # Plot target point
    ax.scatter([cumulative_x], [target_y], color='red', s=100, 
               label=f"Target ({cumulative_x:.1f}, {target_y:.1f})")
    
    # Create line that will be updated during animation
    line, = ax.plot([], [], 'r-', linewidth=2)
    point, = ax.plot([], [], 'ro', markersize=8)
    
    # Set axis limits with some padding
    max_x = cumulative_x * 1.05
    max_y = max(target_y * 1.1, max(y_vals) * 1.1)
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Height')
    ax.set_title(f'Optimal Path (Time={optimal_time:.2f} days)')
    ax.grid(True)
    ax.legend(loc='upper left')
    
    # Function to update the animation
    def update(frame):
        # For each frame, we plot up to that point in the path
        frame_index = min(frame, len(x_vals)-1)
        line.set_data(x_vals[:frame_index+1], y_vals[:frame_index+1])
        point.set_data(x_vals[frame_index], y_vals[frame_index])
        return line, point
    
    # Create animation
    frames = len(path_points)
    interval = max(50, 2000 / (animation_speed * frames))  # Adjust speed (ms per frame)
    anim = animation.FuncAnimation(fig, update, frames=frames, 
                                  interval=interval, blit=True)
    
    return fig, anim, optimal_theta, optimal_time

def draw_static_path(sections, target_y, compare_straight=False):
    """Create and display a static image of the optimal path"""
    optimizer = MarshOptimizerSnell(sections, target_y)
    optimal_theta, optimal_time, path_points, segment_data = optimizer.find_optimal_angle()
    
    # Extract x and y coordinates for plotting
    x_vals = [p[0] for p in path_points]
    y_vals = [p[1] for p in path_points]
    
    # Calculate optimal path distance
    optimal_distance = 0
    for i in range(1, len(path_points)):
        x1, y1 = path_points[i-1]
        x2, y2 = path_points[i]
        segment_distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
        optimal_distance += segment_distance
    
    # Calculate comparison data
    comparison_data = None
    straight_time = None
    straight_distance = None
    
    if compare_straight:
        # Calculate total x distance
        cumulative_x = sum(distance for distance, _ in sections)
        
        # Calculate straight line distance
        straight_distance = ((cumulative_x)**2 + (target_y)**2)**0.5
        
        # Calculate straight line time
        straight_time = 0
        current_x = 0
        current_y = 0
        
        # Angle of the straight line
        if cumulative_x > 0:
            angle = np.arctan(target_y / cumulative_x)
        else:
            angle = np.pi/2  # 90 degrees if vertical line
        
        # Calculate time for each segment
        for i, (distance, speed) in enumerate(sections):
            # Length of the path through this section
            segment_length = distance / np.cos(angle)
            segment_time = segment_length / speed
            straight_time += segment_time
        
        # Calculate additional metrics
        time_saved = straight_time - optimal_time
        time_saved_pct = (time_saved / straight_time) * 100
        
        distance_diff = straight_distance - optimal_distance
        distance_pct = (distance_diff / straight_distance) * 100 if straight_distance > 0 else 0
        
        # Average speed (distance/time)
        optimal_avg_speed = optimal_distance / optimal_time if optimal_time > 0 else 0
        straight_avg_speed = straight_distance / straight_time if straight_time > 0 else 0
        speed_diff = optimal_avg_speed - straight_avg_speed
        speed_diff_pct = (speed_diff / straight_avg_speed) * 100 if straight_avg_speed > 0 else 0
        
        comparison_data = {
            "optimal_distance": optimal_distance,
            "optimal_time": optimal_time, 
            "straight_distance": straight_distance,
            "straight_time": straight_time,
            "time_saved": time_saved,
            "time_saved_pct": time_saved_pct,
            "distance_diff": distance_diff,
            "distance_pct": distance_pct,
            "optimal_avg_speed": optimal_avg_speed,
            "straight_avg_speed": straight_avg_speed,
            "speed_diff": speed_diff,
            "speed_diff_pct": speed_diff_pct
        }
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate section boundaries
    x_boundaries = [0]
    cumulative_x = 0
    for distance, _ in sections:
        cumulative_x += distance
        x_boundaries.append(cumulative_x)
    
    # Find min and max speeds to normalize colors
    min_speed = min(speed for _, speed in sections)
    max_speed = max(speed for _, speed in sections)
    
    # Plot section boundaries and color the sections
    for i in range(len(sections)):
        x_start = x_boundaries[i]
        x_end = x_boundaries[i+1]
        speed = sections[i][1]
        # Normalize speed to get a color (red for slow, green for fast)
        if max_speed > min_speed:  # Avoid division by zero
            normalized_speed = (speed - min_speed) / (max_speed - min_speed)
        else:
            normalized_speed = 0.5  # If all speeds are the same
        ax.fill_between([x_start, x_end], [0, 0], [target_y * 1.2, target_y * 1.2], 
                       color=(1-normalized_speed, normalized_speed, 0, 0.3))
    
    # Plot target point and optimal path
    ax.scatter([cumulative_x], [target_y], color='red', s=100)
    ax.plot(x_vals, y_vals, 'r-', linewidth=2)
    
    # If comparison is enabled, plot the straight-line path
    if compare_straight:
        # Draw straight line from (0,0) to (cumulative_x, target_y)
        straight_x = [0, cumulative_x]
        straight_y = [0, target_y]
        ax.plot(straight_x, straight_y, 'b--', linewidth=2)
    
    # Set axis limits with some padding
    max_x = cumulative_x * 1.05
    max_y = max(target_y * 1.1, max(y_vals) * 1.1)
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Height')
    ax.set_title('Optimal Path vs. Straight Path' if compare_straight else 'Optimal Path')
    ax.grid(True)
    
    return fig, optimal_theta, optimal_time, optimal_distance, comparison_data

def draw_interactive_path(sections, target_y, compare_straight=False):
    """Create and display an interactive plot of the optimal path using Plotly"""
    optimizer = MarshOptimizerSnell(sections, target_y)
    optimal_theta, optimal_time, path_points, segment_data = optimizer.find_optimal_angle()
    
    # Extract directly from segment_data
    segment_distances = segment_data["distances"]
    segment_times = segment_data["times"]
    optimal_distance = segment_data["total_distance"]
    
    # Extract x and y coordinates for plotting
    x_vals = [p[0] for p in path_points]
    y_vals = [p[1] for p in path_points]
    
    # Find min and max speeds to normalize colors
    min_speed = min(speed for _, speed in sections)
    max_speed = max(speed for _, speed in sections)
    
    # Calculate maximum values for plotting boundaries
    cumulative_x = sum(distance for distance, _ in sections)
    max_x = cumulative_x * 1.05
    max_y = max(target_y * 1.1, max(segment_distances) * 1.1)
    
    # Calculate comparison data
    comparison_data = None
    straight_segments = []
    
    if compare_straight:
        # Calculate total x distance
        cumulative_x = sum(distance for distance, _ in sections)
        
        # Calculate straight line distance
        straight_distance = ((cumulative_x)**2 + (target_y)**2)**0.5
        
        # Angle of the straight line
        if cumulative_x > 0:
            angle = np.arctan(target_y / cumulative_x)
        else:
            angle = np.pi/2  # 90 degrees if vertical line
        
        # Calculate segment data for straight line
        straight_time = 0
        current_x = 0
        
        for i, (distance, speed) in enumerate(sections):
            # Length of the path through this section
            segment_length = distance / np.cos(angle)
            segment_time = segment_length / speed
            straight_time += segment_time
            
            segment_data = {
                "distance": segment_length,
                "time": segment_time,
                "speed": speed,  # Use the given speed directly
                "optimal_distance": segment_distances[i],
                "optimal_time": segment_times[i],
                "optimal_speed": speed,  # Use the same speed for optimal path
                "distance_diff": segment_distances[i] - segment_length,
                "distance_diff_pct": ((segment_distances[i] - segment_length) / segment_length * 100) if segment_length > 0 else 0,
                "time_diff": segment_times[i] - segment_time,
                "time_diff_pct": ((segment_times[i] - segment_time) / segment_time * 100) if segment_time > 0 else 0
            }
            straight_segments.append(segment_data)
            current_x += distance
        
        # Calculate overall comparison metrics
        time_saved = straight_time - optimal_time
        time_saved_pct = (time_saved / straight_time) * 100 if straight_time > 0 else 0
        
        distance_diff = straight_distance - optimal_distance
        distance_pct = (distance_diff / straight_distance) * 100 if straight_distance > 0 else 0
        
        # Average speed (distance/time)
        optimal_avg_speed = optimal_distance / optimal_time if optimal_time > 0 else 0
        straight_avg_speed = straight_distance / straight_time if straight_time > 0 else 0
        speed_diff = optimal_avg_speed - straight_avg_speed
        speed_diff_pct = (speed_diff / straight_avg_speed) * 100 if straight_avg_speed > 0 else 0
        
        comparison_data = {
            "optimal_distance": optimal_distance,
            "optimal_time": optimal_time, 
            "straight_distance": straight_distance,
            "straight_time": straight_time,
            "time_saved": time_saved,
            "time_saved_pct": time_saved_pct,
            "distance_diff": distance_diff,
            "distance_pct": distance_pct,
            "optimal_avg_speed": optimal_avg_speed,
            "straight_avg_speed": straight_avg_speed,
            "speed_diff": speed_diff,
            "speed_diff_pct": speed_diff_pct,
            "segments": straight_segments
        }
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Calculate section boundaries
    x_boundaries = [0]
    cumulative_x = 0
    for distance, _ in sections:
        cumulative_x += distance
        x_boundaries.append(cumulative_x)
    
    # Create terrain section rectangles with hover information
    for i in range(len(sections)):
        x_start = x_boundaries[i]
        x_end = x_boundaries[i+1]
        distance, speed = sections[i]
        
        # Calculate color based on speed
        if max_speed > min_speed:
            normalized_speed = (speed - min_speed) / (max_speed - min_speed)
        else:
            normalized_speed = 0.5
        
        # Set color: red for slow, green for fast
        fill_color = f'rgba({int(255*(1-normalized_speed))}, {int(255*normalized_speed)}, 0, 0.3)'
        
        # Prepare hover text for this section
        if compare_straight:
            section_data = straight_segments[i]
            
            # Determine if optimal path is shorter or longer
            comparison_word = "shorter" if section_data["distance_diff"] > 0 else "longer"
            
            hover_text = (
                f"<b>Section {i+1}</b><br>"
                f"Speed: {speed} units/day<br><br>"
                f"<b>Optimal path:</b> {section_data['optimal_distance']:.2f} units, {section_data['optimal_time']:.2f} days<br>"
                f"<b>Straight path:</b> {section_data['distance']:.2f} units, {section_data['time']:.2f} days<br><br>"
                f"Optimal path is <b>{abs(section_data['distance_diff_pct']):.1f}%</b> {comparison_word}"
            )
        else:
            hover_text = f"<b>Section {i+1}</b><br>Speed: {speed} units/day"
        
        # Add the section rectangle as the background
        fig.add_shape(
            type="rect",
            x0=x_start,
            y0=0,
            x1=x_end,
            y1=max_y * 0.95,
            fillcolor=fill_color,
            opacity=0.3,
            line_width=0,
            layer="below"
        )
        
        # Add hover area - simple and effective approach
        fig.add_trace(go.Scatter(
            x=[x_start + (x_end - x_start)/2],
            y=[max_y * 0.45],
            mode="markers",
            marker=dict(
                opacity=0.01,
                size=max_y * 0.95,
                symbol="square",
            ),
            hoverinfo="text",
            hovertext=hover_text,
            hoverlabel=dict(
                bgcolor="rgba(40, 40, 40, 0.95)",
                font=dict(size=12, color="white", family="Arial"),
                bordercolor="darkgray"
            ),
            showlegend=False,
            name=f"Section {i+1}"
        ))
    
    # Add custom modebar buttons for better user interaction
    fig.update_layout(
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255, 255, 255, 0.7)',
            color='rgba(0, 0, 0, 0.5)',
            activecolor='rgba(0, 0, 0, 1)'
        )
    )
    
    # Enable responsive behavior
    fig.update_layout(
        autosize=True,
    )
    
    # Advanced: Add custom JavaScript for section hover highlight
    # This creates a hover effect where the section being hovered becomes more saturated
    for i in range(len(sections)):
        fig.add_trace(go.Scatter(
            x=[], y=[], 
            mode="markers",
            marker=dict(size=0),
            hoverinfo="none",
            showlegend=False,
            visible=False,
            name=f"Section {i+1} Highlight",
            fill='toself',
            fillcolor='rgba(0,0,0,0)'
        ))
    
    # Add optimal path
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='Optimal Path',
            line=dict(color='red', width=3),
            hoverinfo='none'
        )
    )
    
    # Add straight path if comparison is enabled
    if compare_straight:
        fig.add_trace(
            go.Scatter(
                x=[0, cumulative_x],
                y=[0, target_y],
                mode='lines',
                name='Straight Path',
                line=dict(color='blue', width=2, dash='dash'),
                hoverinfo='none'
            )
        )
    
    # Add target point
    fig.add_trace(
        go.Scatter(
            x=[cumulative_x],
            y=[target_y],
            mode='markers',
            name='Target Point',
            marker=dict(color='red', size=12),
            hoverinfo='none'
        )
    )
    
    # Update layout 
    fig.update_layout(
        title='Optimal Path vs. Straight Path' if compare_straight else 'Optimal Path',
        xaxis_title='Distance',
        yaxis_title='Height',
        height=600,
        hovermode='closest',
        hoverdistance=100,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Set axis limits with some padding
    fig.update_xaxes(range=[0, max_x])
    fig.update_yaxes(range=[0, max_y])
    
    return fig, optimal_theta, optimal_time, optimal_distance, comparison_data

# Streamlit app
# Streamlit app
def main():
    st.title("Snell's Law Path Simulator")
    st.write("""
    This app simulates the optimal path through different terrain sections using Snell's law.
    Each section has a different speed, and the goal is to find the path that minimizes travel time.
    Hover over any section to see detailed comparisons between optimal and straight paths for that section.
    """)
    
    # Default sections
    default_sections = [
        ((50.0 * sqrt(2) - 50)/2, 10),  # First dry section
        (10, 9),                         # First marsh section
        (10, 8),                         # Second marsh section
        (10, 7),                         # Third marsh section
        (10, 6),                         # Fourth marsh section
        (10, 5),                         # Fifth marsh section
        ((50.0 * sqrt(2) - 50)/2, 10)    # Final dry section
    ]
    default_target_y = 50.0 * sqrt(2)
    
    # Simple text input for sections
    st.header("Configure Terrain Sections")
    
    # Default sections as string for text area
    default_sections_str = json.dumps(default_sections)
    
    sections_str = st.text_area(
        "Enter sections as a list of [distance, speed] pairs:",
        value=default_sections_str,
        height=200
    )
    
    # Target Y coordinate
    target_y = st.number_input("Target Y coordinate:", 1.0, 10000.0, default_target_y)
    
    # Comparison toggle
    compare_straight = st.checkbox("Compare with straight-line path", value=True)
    
    # Parse sections from text input
    try:
        sections = json.loads(sections_str.strip())
        # Validate sections format
        valid_sections = True
        for section in sections:
            if not isinstance(section, list) and not isinstance(section, tuple):
                valid_sections = False
                break
            if len(section) != 2:
                valid_sections = False
                break
            try:
                float(section[0])
                float(section[1])
            except:
                valid_sections = False
                break
        
        if not valid_sections:
            st.error("Invalid section format. Please use the format: [[distance1, speed1], [distance2, speed2], ...]")
            return
            
        # Convert any lists to tuples
        sections = [tuple(section) for section in sections]
        
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please use the format: [[distance1, speed1], [distance2, speed2], ...]")
        return
        
    # Button to start the simulation
    if st.button("Run Simulation"):
        with st.spinner("Calculating optimal path..."):
            # Use the interactive plot instead of the static plot
            fig, optimal_theta, optimal_time, optimal_distance, comparison_data = draw_interactive_path(sections, target_y, compare_straight)
            
            # Display the interactive plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display comparison data if available
            if comparison_data:
                # Create comparison metrics table for the overall path
                metrics_df = pd.DataFrame({
                    "Metric": ["Distance (units)", "Travel time (days)", "Average speed (units/day)"],
                    "Optimal Path": [
                        f"{comparison_data['optimal_distance']:.2f}",
                        f"{comparison_data['optimal_time']:.2f}",
                        f"{comparison_data['optimal_avg_speed']:.2f}"
                    ],
                    "Straight Path": [
                        f"{comparison_data['straight_distance']:.2f}",
                        f"{comparison_data['straight_time']:.2f}",
                        f"{comparison_data['straight_avg_speed']:.2f}"
                    ]
                })
                

                st.info("Hover over each section in the plot to see detailed comparisons for that specific section.")

                # Display the table right after the plot
                st.subheader("Overall Path Comparison")
                st.table(metrics_df)
                
                # Extract comparison metrics
                time_saved = comparison_data['time_saved']
                time_pct = comparison_data['time_saved_pct']
                
                dist_diff = comparison_data['distance_diff']
                dist_pct = comparison_data['distance_pct']
                
                speed_diff = comparison_data['speed_diff']
                speed_pct = comparison_data['speed_diff_pct']
                
                # Create simple sentences with blanks filled by bold values
                st.markdown(f"""
                The optimal path is:

                - **{abs(dist_diff):.2f} units** {'shorter' if dist_diff > 0 else 'longer'} than the straight path (**{abs(dist_pct):.2f}%** {'less' if dist_diff > 0 else 'more'} distance).

                - **{abs(speed_diff):.2f} units/day** {'faster' if speed_diff > 0 else 'slower'} than the straight path (**{abs(speed_pct):.2f}%** {'speed increase' if speed_diff > 0 else 'speed decrease'}).

                - **{abs(time_saved):.2f} days** {'less' if time_saved > 0 else 'more'} than the straight path (**{abs(time_pct):.2f}%** {'less time' if time_saved > 0 else 'more time'}).
                """)
                
            
            st.write("Simulation complete! You can adjust parameters and run again.")

if __name__ == "__main__":
    main() 