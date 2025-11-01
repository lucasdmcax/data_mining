"""
Centralized styles for the AIAI Customer Analytics Dashboard
"""

def get_custom_css():
    """
    Returns the custom CSS styling for the entire dashboard.
    Import this function and apply it to any page for consistent styling.
    """
    return """
    <style>
    /* Main Headers */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #5a6c7d;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f4788;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f4788;
        padding-bottom: 0.5rem;
    }
    
    /* Containers and Boxes */
    .info-box {
        background-color: #f0f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f4788;
        margin: 1rem 0;
    }
    
    .metric-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Team Cards */
    .team-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease;
    }
    
    .team-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .student-name {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f4788;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .placeholder-img {
        width: 150px;
        height: 150px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        margin: 0 auto 1rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        color: white;
    }
    
    /* Compact Team Cards for Home Page */
    .team-card-compact {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        text-align: center;
        height: 100%;
    }
    
    .placeholder-img-compact {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        margin: 0 auto 0.5rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        color: white;
    }
    
    .student-name-compact {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1f4788;
        margin-top: 0.5rem;
    }
    </style>
    """
