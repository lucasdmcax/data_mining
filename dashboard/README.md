# AIAI Customer Analytics Dashboard

## Structure

```
dashboard/
├── home.py                 # Main landing page (run with: streamlit run home.py)
├── styles.py              # Centralized CSS styles for all pages
├── pages/
│   ├── about_us.py        # Detailed team information page
│   └── template_page.py   # Template for creating new pages
└── README.md              # This file
```

## Quick Start

1. **Run the dashboard:**
   ```bash
   cd dashboard
   streamlit run home.py
   ```

2. **Create a new page:**
   - Copy `pages/template_page.py`
   - Rename it to your desired page name (e.g., `eda_analysis.py`)
   - Modify the content as needed
   - The page will automatically appear in the sidebar

## Styling

All pages should import and use the centralized styles from `styles.py`:

```python
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import styles
sys.path.append(str(Path(__file__).parent.parent))
from styles import get_custom_css

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)
```

### Available CSS Classes

- `.main-header` - Large centered headers
- `.sub-header` - Subtitle text
- `.section-header` - Section dividers with underline
- `.info-box` - Information callout boxes
- `.metric-container` - Metric display boxes
- `.team-card` - Full team member cards (for about_us page)
- `.team-card-compact` - Compact team cards (for home page)
- `.placeholder-img` - Large profile picture placeholder
- `.placeholder-img-compact` - Small profile picture placeholder
- `.student-name` - Team member name styling

## Pages Overview

### Home Page (`home.py`)
The main landing page featuring:
- Project overview and objectives
- Data scope metrics
- Methodology overview
- Compact team member display
- Navigation guide

## Color Palette

- Primary Blue: `#1f4788`
- Secondary Gray: `#5a6c7d`
- Background: `#f0f4f8`
- Gradient: `#667eea` to `#764ba2`

## Customization

### Update Team Members
Edit student names in:
- `home.py` (lines with "Student Name 1-4")
- `pages/about_us.py` (lines with "Student Name 1-4")

### Add Real Profile Pictures
Replace the placeholder emoji (👤) with:
```python
st.image("path/to/image.jpg", width=150)
```

### Modify Styles
Edit `styles.py` to change colors, fonts, or layouts globally.
