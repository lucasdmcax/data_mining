"""
Centralized styles for the AIAI Customer Analytics Dashboard
"""

# ==========================================
# COLOR PALETTE - CUSTOMIZE HERE
# ==========================================

# Primary Colors
PRIMARY_BLUE = "#1f4788"
PRIMARY_BLUE_DARK = "#163561"
PRIMARY_BLUE_LIGHT = "#2d5ba8"

# Secondary Colors
SECONDARY_GRAY = "#5a6c7d"
SECONDARY_GRAY_LIGHT = "#8b99a8"
SECONDARY_GRAY_DARK = "#3d4a56"

# Background Colors
BG_WHITE = "#ffffff"
BG_LIGHT_BLUE = "#f0f4f8"
BG_OFF_WHITE = "#fafbfc"

# Gradient Colors
GRADIENT_START = "#667eea"
GRADIENT_END = "#764ba2"

# Text Colors
TEXT_PRIMARY = "#2c3e50"
TEXT_SECONDARY = "#5a6c7d"
TEXT_LIGHT = "#8b99a8"
TEXT_WHITE = "#ffffff"

# Shadows
SHADOW_LIGHT = "rgba(0, 0, 0, 0.08)"
SHADOW_MEDIUM = "rgba(0, 0, 0, 0.1)"
SHADOW_DARK = "rgba(0, 0, 0, 0.15)"

# Border & Radius
BORDER_COLOR = "#e1e8ed"
BORDER_RADIUS_SM = "10px"
BORDER_RADIUS_MD = "15px"
BORDER_RADIUS_LG = "20px"


# ==========================================
# HELPER FUNCTIONS FOR COMMON COMPONENTS
# ==========================================

def get_metric_html(title, description):
    """
    Generate HTML for a metric container with centralized colors.
    
    Args:
        title (str): The main metric value or title
        description (str): Description text below the title
    
    Returns:
        str: HTML for metric container
    """
    return f"""
        <div class="metric-container">
            <h2 style="color: {PRIMARY_BLUE};">{title}</h2>
            <p style="color: {SECONDARY_GRAY};">{description}</p>
        </div>
    """


def get_info_box_html(title, content):
    """
    Generate HTML for an info box with centralized colors.
    
    Args:
        title (str): The info box title
        content (str): The content text
    
    Returns:
        str: HTML for info box
    """
    return f"""
        <div class="info-box">
            <h3 style="color: {PRIMARY_BLUE};">{title}</h3>
            <p>{content}</p>
        </div>
    """


def get_section_title_html(icon, title):
    """
    Generate HTML for a section title with icon.
    
    Args:
        icon (str): Emoji or icon
        title (str): Section title text
    
    Returns:
        str: HTML for section title
    """
    return f'<h4 style="color: {PRIMARY_BLUE};">{icon} {title}</h4>'


def get_custom_css():
    """
    Returns the custom CSS styling for the entire dashboard.
    Import this function and apply it to any page for consistent styling.
    All colors are defined as Python variables at the top of this file.
    """
    return """
    <style>
    /* ==========================================
       MAIN HEADERS
       ========================================== */
    .main-header {{
        font-size: 3rem;
        font-weight: bold;
        color: {primary_blue};
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    
    .sub-header {{
        font-size: 1.5rem;
        color: {secondary_gray};
        text-align: center;
        margin-bottom: 2rem;
    }}
    
    .section-header {{
        font-size: 1.8rem;
        font-weight: bold;
        color: {primary_blue};
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid {primary_blue};
        padding-bottom: 0.5rem;
    }}
    
    /* ==========================================
       CONTAINERS AND BOXES
       ========================================== */
    .info-box {{
        background-color: {bg_light_blue};
        padding: 1.5rem;
        border-radius: {border_radius_sm};
        border-left: 5px solid {primary_blue};
        margin: 1rem 0;
    }}
    
    .metric-container {{
        background-color: {bg_white};
        padding: 1.5rem;
        border-radius: {border_radius_sm};
        box-shadow: 0 2px 4px {shadow_medium};
        text-align: center;
    }}
    
    /* ==========================================
       TEAM CARDS (Full Size)
       ========================================== */
    .team-card {{
        background-color: {bg_white};
        padding: 2rem;
        border-radius: {border_radius_md};
        box-shadow: 0 4px 6px {shadow_medium};
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .team-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 6px 12px {shadow_dark};
    }}
    
    .student-name {{
        font-size: 1.5rem;
        font-weight: bold;
        color: {primary_blue};
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }}
    
    .placeholder-img {{
        width: 150px;
        height: 150px;
        background: linear-gradient(135deg, {gradient_start} 0%, {gradient_end} 100%);
        border-radius: 50%;
        margin: 0 auto 1rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        color: {text_white};
    }}
    
    /* ==========================================
       TEAM CARDS (Compact for Home Page)
       ========================================== */
    .team-card-compact {{
        background-color: {bg_white};
        padding: 1.5rem;
        border-radius: {border_radius_sm};
        box-shadow: 0 2px 4px {shadow_light};
        text-align: center;
        height: 100%;
        transition: transform 0.2s ease;
    }}
    
    .team-card-compact:hover {{
        transform: translateY(-3px);
        box-shadow: 0 4px 8px {shadow_medium};
    }}
    
    .placeholder-img-compact {{
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, {gradient_start} 0%, {gradient_end} 100%);
        border-radius: 50%;
        margin: 0 auto 0.5rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        color: {text_white};
    }}
    
    .student-name-compact {{
        font-size: 1.1rem;
        font-weight: bold;
        color: {primary_blue};
        margin-top: 0.5rem;
    }}
    
    .student-number-compact {{
        font-size: 0.9rem;
        color: {secondary_gray};
        margin-top: 0.3rem;
        font-style: italic;
    }}
    </style>
    """.format(
        primary_blue=PRIMARY_BLUE,
        primary_blue_dark=PRIMARY_BLUE_DARK,
        primary_blue_light=PRIMARY_BLUE_LIGHT,
        secondary_gray=SECONDARY_GRAY,
        secondary_gray_light=SECONDARY_GRAY_LIGHT,
        secondary_gray_dark=SECONDARY_GRAY_DARK,
        bg_white=BG_WHITE,
        bg_light_blue=BG_LIGHT_BLUE,
        bg_off_white=BG_OFF_WHITE,
        gradient_start=GRADIENT_START,
        gradient_end=GRADIENT_END,
        text_primary=TEXT_PRIMARY,
        text_secondary=TEXT_SECONDARY,
        text_light=TEXT_LIGHT,
        text_white=TEXT_WHITE,
        shadow_light=SHADOW_LIGHT,
        shadow_medium=SHADOW_MEDIUM,
        shadow_dark=SHADOW_DARK,
        border_color=BORDER_COLOR,
        border_radius_sm=BORDER_RADIUS_SM,
        border_radius_md=BORDER_RADIUS_MD,
        border_radius_lg=BORDER_RADIUS_LG
    )
