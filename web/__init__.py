"""
PHISE Web Application - Main Entry Point
================================

This directory contains the complete Streamlit web application for PHISE analysis.

QUICK START
===========

1. Install dependencies:
   pip install -r requirements-web.txt

2. Ensure PHISE is installed:
   pip install -e .

3. Run the application:
   python run.py

   Or directly:
   streamlit run main.py

The app will open at: http://localhost:8501

DOCUMENTATION ROADMAP
====================

START HERE (Pick one):
  
  - QUICKSTART.md
    → 30-second setup guide
    → Recommended for first-time users
    
  - README.md
    → Comprehensive technical documentation
    → For developers and detailed setup
    
  - USER_GUIDE.md
    → How to use each module
    → Recommended after installation
    
  - MIGRATION_GUIDE.md
    → For users coming from Jupyter notebooks
    → See mapping of old notebooks to new modules
    
  - INDEX.md
    → Module inventory and architecture overview
    → For project understanding
    
  - SUMMARY.md
    → This implementation summary
    → Project statistics and completion status

APPLICATION STRUCTURE
====================

main.py
  → Hub interface with navigation menu
  → 4 categories: Foundational, Calibration, Geometry, Education
  → Central entry point

pages/
  → 13 individual analysis modules
  → Named with numbers (01-13) for easy ordering
  → Each is a complete Streamlit app

utils/
  → Shared utility functions
  → Caching, parameter control, plotting
  → Imported by all modules

.streamlit/
  → Streamlit configuration
  → Theme, logging, server settings

requirements-web.txt
  → Python package dependencies
  → Install with: pip install -r requirements-web.txt

MODULES AT A GLANCE
==================

📊 Foundational Analysis (4 modules)
   01_data_representations.py    - Output distributions
   02_test_statistics.py         - Detection metrics & ROC
   03_transmission_maps.py       - Null depth maps
   04_sky_contribution.py        - Thermal background

🔧 Calibration & Control (3 modules)
   05_calibration.py             - Classical piston correction
   06_neural_calibration.py      - Neural network calibration
   07_manual_control.py          - Interactive commissioning

🗺️ Geometry & Observation (4 modules)
   08_projected_telescopes.py    - Baseline geometry
   09_temporal_response.py       - Time evolution
   10_wavelength_scan.py         - Spectral response
   11_noise_sensitivity.py       - Noise analysis

🎓 Education & Demo (2 modules)
   12_demonstration.py           - 5-step walkthrough
   13_distribution_model.py      - Statistical models

GETTING HELP
============

Problem: "I'm new to PHISE"
Solution: Read QUICKSTART.md, then start with:
          🎓 Education & Demo → Demonstration

Problem: "I used the Jupyter notebooks before"
Solution: Read MIGRATION_GUIDE.md for module mapping

Problem: "How do I use a specific module?"
Solution: Open that module and read the description at the top
          Also check USER_GUIDE.md for detailed explanations

Problem: "The app is slow"
Solution: See README.md troubleshooting section
          Key: Reduce "Number of Samples" parameter

SYSTEM REQUIREMENTS
===================

✓ Python 3.8+
✓ 4GB RAM minimum
✓ Modern web browser (Chrome, Firefox, Safari, Edge)
✓ ~500MB for dependencies

VERSIONS
========

Application Version: 1.0.0
Release Date: 2025-11-26
Status: Production Ready

Compatible with:
  - Streamlit 1.28.0+
  - Python 3.8+
  - PHISE 0.1.0+

ORIGINAL NOTEBOOKS
==================

The original Jupyter notebooks (pre-web conversion) are archived in:
  THESIS/analysis/

They remain fully functional and can still be used with:
  jupyter notebook analysis/demonstration.ipynb

KEY FEATURES
============

✓ Zero coding required
✓ Real-time parameter adjustment
✓ Cached computations for instant switching
✓ Export figures (PNG, PDF, SVG)
✓ Export data (CSV)
✓ Responsive design (desktop/tablet)
✓ Guided learning paths
✓ Comprehensive help text

DEPLOYMENT OPTIONS
==================

Local (default):
  python web/run.py
  → Runs on http://localhost:8501

Remote (Streamlit Cloud):
  → See README.md for deployment instructions
  → Free public hosting available

Docker:
  → Build with: docker build -t phise-web .
  → Run with: docker run -p 8501:8501 phise-web

AWS/Cloud:
  → See README.md for instructions

NEXT STEPS
==========

1. Run the quick start:
   python run.py

2. Explore a module:
   → Try 🎓 Education & Demo → Demonstration

3. Read the appropriate guide:
   → Beginner: QUICKSTART.md
   → Advanced: README.md
   → Migration: MIGRATION_GUIDE.md

4. Start analyzing:
   → Pick a module that matches your research

SUPPORT & FEEDBACK
==================

For detailed help, see:
  - README.md: Technical setup and configuration
  - USER_GUIDE.md: How to use each module
  - MIGRATION_GUIDE.md: Transition from notebooks
  - INDEX.md: Module inventory and architecture

For issues:
  - Check README.md troubleshooting section
  - Verify dependencies: pip list
  - Check Python version: python --version

---

📝 Last Updated: 2025-11-26
🚀 Ready to start? Run: python run.py
"""

# This file is for reference. The actual app starts with:
# streamlit run main.py
# or
# python run.py

if __name__ == "__main__":
    print(__doc__)
