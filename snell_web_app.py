import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from math import sin, cos, tan, sqrt, pi, asin, atan2
import random
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the solver code from the new module
from snell_solver import MarshOptimizerSnell, calculate_straight_path, timeout_handler    

# Remove the MarshOptimizerSnell class from this file
# (The class has been moved to snell_solver.py)

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
            comparison_word = "shorter" if section_data["distance_diff"] < 0 else "longer"
            
            hover_text = (
                f"<b>Section {i+1}</b><br>"
                f"Speed: {speed} units/day<br><br>"
                f"<b>Optimal path:</b> {section_data['optimal_distance']:.2f} units, {section_data['optimal_time']:.2f} days<br>"
                f"<b>Straight path:</b> {section_data['distance']:.2f} units, {section_data['time']:.2f} days<br><br>"
                f"Optimal path is <b>{abs(section_data['distance_diff_pct']):.1f}%</b> {comparison_word}"
            )
        else:
            # still share length and days for the optimal path
            hover_text = (
                f"<b>Section {i+1}</b><br>"
                f"Speed: {speed} units/day<br><br>"
                f"<b>Optimal path:</b> {optimal_distance:.2f} units, {optimal_time:.2f} days<br>"
            )
            
        # Add the section rectangle as the background
        fig.add_shape(
            type="rect",
            x0=x_start,
            y0=0,
            x1=x_end,
            y1=max_y,
            fillcolor=fill_color,
            opacity=0.3,
            line_width=0,
            layer="below"
        )
        
        # Add hover area with consistent hover text positioning
        # Use a column of invisible dots but specify consistent hover position
        mid_x = x_start + (x_end - x_start)/2
        
        fig.add_trace(go.Scatter(
            x=[mid_x] * 10,  # Vertical column of points
            y=[max_y * .1 * i for i in range(1, 11)],  # Evenly spaced points
            mode="markers",
            marker=dict(
                opacity=0.01,
                size=min(max_y, x_end - x_start) * 0.2,  # Smaller markers
                symbol="square",
            ),
            hoverinfo="text",
            hovertext=hover_text,
            hoverlabel=dict(
                bgcolor="rgba(40, 40, 40, 0.65)",
                font=dict(size=12, color="white", family="Arial"),
                bordercolor="darkgray"
            ),
            showlegend=False,
            name=f"Section {i+1}",
            hoverlabel_align='auto'  # This helps keep the position consistent
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
        height=400,
        hovermode='closest',
        hoverdistance=100,
        title_x=0.5,
        title_xanchor='center',
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Set axis limits with some padding
    fig.update_xaxes(range=[0, max_x])
    fig.update_yaxes(range=[0, max_y])
    
    return fig, optimal_theta, optimal_time, optimal_distance, comparison_data


# Initialize sidebar state
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'collapsed'

# Custom favicon URL
favicon_url = "https://gallery.yopriceville.com/var/albums/Free-Clipart-Pictures/Decorative-Elements-PNG/Light_Effect_PNG_Clip_Art_Image.png?m=1629792209" 

# Set page config with initial sidebar state and custom favicon
st.set_page_config(
    page_title="Snell's Law Path Finder",
    page_icon=favicon_url,  # Use custom favicon URL instead of "🔍"
    initial_sidebar_state=st.session_state.sidebar_state,
    layout="wide"
)

# Default sections
default_sections = [
    (round((50.0 * sqrt(2) - 50)/2, 3), 10),
    (10, 9),
    (10, 8),
    (10, 7),
    (10, 6),
    (10, 5),
    (round((50.0 * sqrt(2) - 50)/2, 3), 10)
]
default_target_y = 50.0 * sqrt(2)

# Initialize session state variables
if 'sections_str' not in st.session_state:
    st.session_state.sections_str = json.dumps(default_sections)
    
if 'target_y' not in st.session_state:
    st.session_state.target_y = default_target_y
    
if 'num_sections' not in st.session_state:
    st.session_state.num_sections = 5
    
if 'compare_straight' not in st.session_state:
    st.session_state.compare_straight = True
    
# Add a flag to track if initial visualization has been shown
if 'initial_viz_shown' not in st.session_state:
    st.session_state.initial_viz_shown = False

# Add a callback function to update the text area
def update_random_sections():
    # Generate random data
    random_sections = []
    for i in range(st.session_state.num_sections):
        distance = round(random.uniform(5.0, 15.0), 2)
        speed = round(random.uniform(3.0, 10.0), 2)
        random_sections.append((distance, speed))
    
    # Generate random target
    random_target_y = round(random.uniform(1.0, 100.0), 2)
    
    # Update session state variables
    st.session_state.random_sections = random_sections
    st.session_state.random_target_y = random_target_y
    st.session_state.sections_str = json.dumps(random_sections)
    st.session_state.target_y = random_target_y

# Initialize random sections if not already set
if 'random_sections' not in st.session_state:
    st.session_state.random_sections = None
    
if 'random_target_y' not in st.session_state:
    st.session_state.random_target_y = None

# Title with reduced margins and flashy home button (cool colors only)
st.markdown("""
<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes cool-border {
    0% { border-color: #4b6cb7; box-shadow: 0 0 10px #4b6cb7; }
    25% { border-color: #3498db; box-shadow: 0 0 10px #3498db; }
    50% { border-color: #2ecc71; box-shadow: 0 0 10px #2ecc71; }
    75% { border-color: #9b59b6; box-shadow: 0 0 10px #9b59b6; }
    100% { border-color: #4b6cb7; box-shadow: 0 0 10px #4b6cb7; }
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.03); }
    100% { transform: scale(1); }
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
}

.flashy-home-button {
    text-decoration: none;
    background: linear-gradient(45deg, #4b6cb7, #182848);
    color: white !important;
    padding: 0.6rem 1.2rem;
    border-radius: 0.5rem;
    border: 3px solid #3498db;
    font-weight: bold;
    display: flex;
    align-items: center;
    position: relative;
    overflow: hidden;
    animation: cool-border 2s linear infinite, pulse 1.5s ease-in-out infinite;
    transition: all 0.3s ease;
    z-index: 1;
}

.flashy-home-button:hover {
    animation: cool-border 0.5s linear infinite, shake 0.5s infinite, pulse 0.8s ease-in-out infinite;
    transform: scale(1.05);
}

.flashy-home-button::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, rgba(75,108,183,0.1), rgba(46,204,113,0.5), rgba(75,108,183,0.1));
    transform: rotate(45deg);
    animation: spin 3s linear infinite;
    z-index: -1;
}

.home-icon {
    margin-right: 0.5rem;
    animation: spin 3s linear infinite;
    display: inline-block;
    font-size: 1.2rem;
}
</style>

<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
    <h1 style="margin: 0;">Snell's Law Path Finder</h1>
    <a href="#visualization" class="button-link" style="margin-right: 1rem;"><button style="width:100%; padding: 0.5rem; cursor: pointer; background-color: rgba(46, 204, 113, 0.5); color: white;">Jump to Visualization 📈</button></a>
    <a href="https://antounmichael.github.io" target="_blank" class="flashy-home-button">
        <span class="home-icon">🏠</span> <span style="text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">Visit My Site!</span>
    </a>
</div>
""", unsafe_allow_html=True)

# Create a session state variable to track if intro is expanded
if 'intro_expanded' not in st.session_state:
    st.session_state.intro_expanded = True

# Create collapsible section for introduction - use session state to control expanded state
with st.expander("📚 Problem Statement", expanded=st.session_state.intro_expanded) as intro_expander:
    st.write("""
    ### Introduction
    I got the idea to create this solver/visualizer from a [project euler problem](https://projecteuler.net/problem=607). 
             
    We are presented with a map, comprised of a series of parallel strips of terrain, each with a different travel speed. 
    """)

    st.markdown("""
    <div style="display: flex; align-items: flex-start;">
        <img src="https://projecteuler.net/project/images/p607_marsh.png" width="400" style="margin-right: 20px;">
        <div>
            <p>Speeds are as follows:</p>
            <ul>
                <li>white: 10 leagues/day</li>
                <li>green: 9 leagues/day</li>
                <li>yellow: 8 leagues/day</li>
                <li>tan(?): 7 leagues/day</li>
                <li>orange: 6 leagues/day</li>
                <li>red: 5 leagues/day</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <br>The goal is to find the *shortest time* path from A to B. Of course, the *shortest distance* path is the straight line, 
    but the variation in speed across the map means that we can do better! For example, the straight line path in 
    this map takes ~13.5 days, but the optimal path takes only ~13.1 days.
    """, unsafe_allow_html=True)
    
    # Add CSS for the buttons
    st.markdown("""
    <style>
    .button-link {
        text-decoration: none;
    }
    .button-link button {
        background-color: white;
        color: black;
        border: 2px solid #000;
        border-radius: 4px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    .button-link button:hover {
        background-color: black;
        color: white;
        border-color: #333;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add buttons at the end of the first expander
    button_col1, button_col2, _ = st.columns([1, 1, 2])
    with button_col1:
        st.markdown('<a href="#solution-details" class="button-link"><button style="width:100%; padding: 0.5rem; cursor: pointer; background-color: rgba(46, 204, 113, 0.5); color: white;">Continue to Solution Details ⬇️</button></a>', unsafe_allow_html=True)
    
    with button_col2:
        st.markdown('<a href="#visualization" class="button-link"><button style="width:100%; padding: 0.5rem; cursor: pointer; background-color: rgba(46, 204, 113, 0.5); color: white;">Jump to Visualization 📈</button></a>', unsafe_allow_html=True)

# Add an anchor for the solution details section
st.markdown('<div id="solution-details"></div>', unsafe_allow_html=True)

# Add JavaScript to check URL and expand solution details if anchor is present
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Check if URL contains the solution-details anchor
    if (window.location.hash === '#solution-details') {
        // Find the expander button and click it if it's not already expanded
        setTimeout(function() {
            const expanderButtons = document.querySelectorAll('button[aria-expanded="false"]');
            for (let button of expanderButtons) {
                // Look for the solution details expander by checking the text content
                if (button.textContent.includes('Solution Details')) {
                    button.click();
                    break;
                }
            }
        }, 500); // Small delay to ensure DOM is fully loaded
    }
});
</script>
""", unsafe_allow_html=True)

# Create a separate collapsible section for the solution details - closed by default
with st.expander("🔬 Solution Details: Please don't read this if you want to solve it yourself!", expanded=False):
    st.markdown("""
    #### Gradient Descent
    In my first attempt, I used [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) to find the optimal path.  I had confidence this would work because the optimal path within any section must be a straight line,
    and the cost function is differentiable with no local minima.
    To do this, I
    - placed a waypoint at the boundary of each section
    - wrote a cost function to compute the total travel time for a path 
    - called [`minimize()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) 
                
    This worked, but it was lowkey unsatisfying. So why not solve the problem by simulating light rays?
                
    ### Light Simulation
    """, unsafe_allow_html=True)

    light_col1, light_col2 = st.columns(2)
    with light_col1:
        st.markdown("""
        A few things to note about light:
        - [Fermat's principle](https://en.wikipedia.org/wiki/Fermat%27s_principle) states that light takes the path of least time
        - [Snell's law](https://en.wikipedia.org/wiki/Snell%27s_law) gives the exact relationship between the angle of incidence and the angle of refraction between dissimilar mediums
        - We can simulate the path of light through a series of mediums by applying Snell's law at the boundary between each medium (ensuring not to confuse velocity with index of refraction :D)
        
        We're almost there! If we knew the starting angle of the light ray, we could simulate the path of the light ray through the map. So, what is it?
        
        Hmm... Maybe this protractor will tell us, it should know about angles and stuff.
        """, unsafe_allow_html=True)
                    
        st.image("https://i.imgur.com/V1VVKs3.png", width=300)
        st.markdown("""
        ...okay, fine. Let's do a [binary search](https://en.wikipedia.org/wiki/Binary_search_algorithm). 
                    
        ### Binary Search
        """, unsafe_allow_html=True)
    
    with light_col2:
        # Add a soft background behind the image with fading borders
        st.markdown("""
        <div style="
                    background: radial-gradient(circle, rgba(240, 240, 255, 0.8) 50%, rgba(240, 240, 255, 0) 100%); 
                    border-radius: 10px; 
                    padding: 20px; 
                    text-align: center;">
            <img src="https://i.imgur.com/sRdKi0u.png" width="450" style="max-width: 100%;">
        </div>
        """, unsafe_allow_html=True)


    binary_search_col1, binary_search_col2 = st.columns(2)
    with binary_search_col1:
        st.markdown("""
                The idea is to:
    
    - guess a starting angle (within the range of possible angles)
    - simulate the path of the light ray through the map
    - adjust the starting angle range based on whether the Y coordinate is above or below the target Y coordinate
        - if the Y coordinate is above the target Y coordinate, we know the starting angle is too high, so we can move the upper bound down
        - if the Y coordinate is below the target Y coordinate, we know the starting angle is too low, so we can move the lower bound up
    - repeat until the Y coordinate is within tolerance of the target Y coordinate
    """, unsafe_allow_html=True)
    
    with binary_search_col2:
        st.markdown("""
<div style="display: flex; justify-content: left; align-items: left; height: 200px;">
    <figure style="text-align: center; margin: 0;">
        <img src="https://i.imgur.com/ufFrsNG.gif" width="300" alt="Binary search simulation">
        <figcaption style="margin-top: 10px; font-style: italic; color: gray;">
            Simulation of binary search, green star is the current guess
        </figcaption>
    </figure>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("""
     There is one additional wrinkle: there is a possibility for [total internal reflection](https://en.wikipedia.org/wiki/Total_internal_reflection),
                in which case the simulation will not terminate. 
                However, if we know whether the ray is reflecting upwards or downwards, we know which way to adjust our aim.

    This is so much better than gradient descent. We're solving for only one variable, and we don't even have to compute derivatives!
                

    Thank you for reading! I've gone ahead and implemented a generalized version of this problem with visualizations 
                (at the bottom of the page). You can enter your own map, or generate a random one.
    If you have any questions, please reach out to me on [LinkedIn](https://www.linkedin.com/in/michael-antoun/)!
    """, unsafe_allow_html=True)
    

                
                
                

st.info(" Open the sidebar to configure the map!", icon="↖️")

random_button_col1, random_button_col2, random_button_col3 = st.columns([1, 3, 1])
# Add a random button in the main content area - now positioned after the background info
with random_button_col2:
    main_random_button = st.button("🎲 Generate Random Map 🎲", 
                             type="primary",
                             on_click=update_random_sections,
                             use_container_width=True)

# Add an anchor for the visualization section
st.markdown('<div id="visualization"></div>', unsafe_allow_html=True)

# Create placeholders for the plot and metrics in the main area
plot_placeholder = st.empty()
metrics_placeholder = st.empty()


# Helper function to show plot and metrics
def show_visualization(sections_to_use, target_y_to_use, compare_straight_to_use):
    with plot_placeholder.container():
        with st.spinner("Calculating optimal path..."):
            # Display the simulation
            fig, optimal_theta, optimal_time, optimal_distance, comparison_data = draw_interactive_path(
                sections_to_use, target_y_to_use, compare_straight_to_use)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Style for the button
            m = st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: white;
                    color: black;
                    border: 2px solid #000;
                    transition: all 0.3s ease;
                    padding: 0.25em 1em;
                    margin: 0.25em 0;
                }

                div.stButton > button:first-child:hover {
                    background-color: black;
                    color: white;
                    border-color: #333;
                    transform: scale(1.02);
                }
                </style>""", unsafe_allow_html=True)
            
            path_comp_title_col1, _, path_comp_title_col2 = st.columns([1.5, .001, 0.001])
            with path_comp_title_col2:
                pass  # Keep the column structure but remove the banner
            
            # Display comparison data if available
            if comparison_data:
                with metrics_placeholder.container():
                    # Create comparison metrics table for the overall path
                    metrics_df = pd.DataFrame({
                        "Metric": ["Travel time (days)", "Distance (units)", "Average speed (units/day)"],
                        "Optimal Path": [
                            f"{comparison_data['optimal_time']:.2f}",
                            f"{comparison_data['optimal_distance']:.2f}",
                            f"{comparison_data['optimal_avg_speed']:.2f}"
                        ],
                        "Straight Path": [
                            f"{comparison_data['straight_time']:.2f}",
                            f"{comparison_data['straight_distance']:.2f}",
                            f"{comparison_data['straight_avg_speed']:.2f}"
                        ],
                        "% Difference": [
                            f"{abs(comparison_data['time_saved_pct']):.2f}%",
                            f"{abs(comparison_data['distance_pct']):.2f}%",
                            f"{abs(comparison_data['speed_diff_pct']):.2f}%"
                        ]
                    })
                    
                    with path_comp_title_col1:
                        st.markdown("#### Path Comparison", )
                    st.table(metrics_df)

# no anchor links! this isnt documentation for crying out loud
st.html("<style>[data-testid='stHeaderActionElements'] {display: none;}</style>")

# Sidebar controls
with st.sidebar:
    st.header("Simulation Controls")
    # Text area for sections
    sections_str = st.text_area(
        "Enter sections as a list of [distance, speed] pairs:",
        value=st.session_state.sections_str,
        height=200,
        key="sections_str"
    )
    
    # Target Y input
    st.subheader("Target", anchor=False)
    target_y = st.number_input(
        "Target Y coordinate:", 
        0.0, 10000.0, 
        st.session_state.target_y,
        key="target_input"
    )
    
    # Random section generator settings
    st.subheader("Random Map Settings", anchor=False)
    num_sections = st.number_input(
        "Number of sections:", 
        min_value=1, 
        max_value=200, 
        value=st.session_state.num_sections,
        step=1,
        key="num_sections"
    )
    
    # Comparison toggle
    st.subheader("Options", anchor=False)
    compare_straight = st.checkbox(
        "Compare with straight-line path", 
        value=st.session_state.compare_straight,
        key="compare_straight"
    )
    
    # Put buttons in the sidebar with two columns
    st.subheader("Actions", anchor=False)
    
    col1, col2 = st.columns(2)
    with col1:
        # Use on_click to update the session state before rerun
        random_button = st.button("🎲 Random", 
                                 help="Generate a random map", 
                                 type="primary", 
                                 use_container_width=True,
                                 on_click=update_random_sections)
        
    with col2:
        compute_button = st.button("▶️ Compute", 
                                 help="Compute the path", 
                                 type="primary", 
                                 use_container_width=True)

# Get compare_straight from session state
compare_straight = st.session_state.compare_straight if 'compare_straight' in st.session_state else True

# Handle random generation WITHOUT rerun - fix the condition to be more reliable
if main_random_button or (random_button if 'random_button' in locals() else False):
    # Use the sections and target that were generated in the callback
    random_sections = st.session_state.random_sections
    random_target_y = st.session_state.random_target_y

    if random_target_y is None or random_sections is None:
        update_random_sections()
    
    # Create a unique ID for this visualization
    viz_id = f"viz-{int(time.time())}"
    
    # Create a marker for the visualization
    st.markdown(f'<div id="{viz_id}"></div>', unsafe_allow_html=True)
    
    # Add JavaScript to scroll to the visualization - one time only
    st.markdown(f"""
    <script>
        // One-time scroll to the visualization
        setTimeout(function() {{
            const vizElement = document.getElementById('{viz_id}');
            if (vizElement) {{
                vizElement.scrollIntoView({{behavior: 'auto', block: 'start'}});
                console.log('Scrolled to visualization');
            }}
        }}, 200);
    </script>
    """, unsafe_allow_html=True)
    
    # Show the visualization directly - no rerun!
    show_visualization(random_sections, random_target_y, compare_straight)

# Process sidebar inputs and compute path if requested
elif compute_button:
    try:
        sections = json.loads(sections_str.strip())
        
        # Check if there are too many sections
        if len(sections) > 100:
            st.error(f"Too many sections ({len(sections)}). Please limit to 100 sections for performance reasons.")
        else:
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
            else:
                # Convert any lists to tuples
                sections = [tuple(section) for section in sections]
                show_visualization(sections, target_y, compare_straight)
                
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please use the format: [[distance1, speed1], [distance2, speed2], ...]")

# Automatically show initial visualization if it hasn't been shown yet
elif not st.session_state.initial_viz_shown:
    # Parse the default sections from the text area
    try:
        sections = json.loads(sections_str.strip())
        # Convert any lists to tuples
        sections = [tuple(section) for section in sections]
        # Show the visualization
        show_visualization(sections, target_y, compare_straight)
        # Mark that we've shown the initial visualization
        st.session_state.initial_viz_shown = True
    except json.JSONDecodeError:
        st.error("Invalid JSON format in default sections. Please check the format.")

# Add custom CSS to style the sidebar toggle button
st.markdown("""
<style>
/* Style the sidebar toggle button */
[data-testid="collapsedControl"] {
    background-color: rgba(46, 204, 113, 0.2) !important;  /* Light green background */
    border-radius: 0 10px 10px 0 !important;
    transition: all 0.3s ease !important;
}

[data-testid="collapsedControl"]:hover {
    background-color: rgba(46, 204, 113, 0.4) !important;  /* Darker green on hover */
}

/* Style the arrow icon inside the toggle button */
[data-testid="collapsedControl"] svg {
    color: rgb(46, 204, 113) !important;  /* Green arrow */
    width: 25px !important;
    height: 25px !important;
}

/* Style the expanded control button */
[data-testid="expandedControl"] {
    background-color: rgba(46, 204, 113, 0.2) !important;  /* Light green background */
    transition: all 0.3s ease !important;
}

[data-testid="expandedControl"]:hover {
    background-color: rgba(46, 204, 113, 0.4) !important;  /* Darker green on hover */
}

/* Style the arrow icon inside the expanded button */
[data-testid="expandedControl"] svg {
    color: rgb(46, 204, 113) !important;  /* Green arrow */
    width: 25px !important;
    height: 25px !important;
}
</style>
""", unsafe_allow_html=True)
