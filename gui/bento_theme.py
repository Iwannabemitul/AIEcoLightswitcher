"""
gui/bento_theme.py - Dark Telemetry Bento UI Design System Tokens
AIEcoLightswitcher Showcase Edition (Milestone 4 Implementation)

Unified color palette, spacing, and typography tokens shared across
Python OpenCV HUD rendering and modern web frontend.
"""

BENTO_PALETTE = {
    # Surfaces
    "bg_root": "#0B0F17",              # Deep charcoal root canvas
    "card_surface": "#131822",         # Elevated dark slate card surface
    "card_surface_alt": "#161B22",     # Secondary card / chip tile surface
    "card_surface_subtle": "#0D1117",  # Recessed canvas / terminal background
    
    # Borders
    "border_subtle": "#21262D",        # Subtle 1px card perimeter
    "border_accent": "#30363D",        # Focused / active border
    "border_cyan": "#00F0FF",          # Electric Cyan accent outline
    
    # Accents & Status Indicators
    "accent_cyan": "#00F0FF",          # Electric Cyan (Active vision metrics)
    "accent_indigo": "#6366F1",        # Indigo (Branding & category tags)
    "status_active_mint": "#10B981",   # Vivid Mint (Occupancy active, relay ON)
    "status_sleep_amber": "#F59E0B",   # Amber (Duty-cycle power-saving sleep)
    "status_tamper_crimson": "#EF4444",# Crimson (Disconnect, alarm, relay OFF)
    "status_offline_gray": "#6E7681",  # Neutral Gray (Offline, unassigned)
    
    # Typography
    "text_primary": "#F0F6FC",         # Crisp foreground text
    "text_secondary": "#8B949E",       # Muted slate secondary labels
    "text_muted": "#6E7681",           # Inactive subtext
}

# OpenCV BGR Color Tuples for HUD Drawing
BGR_COLORS = {
    "mint": (129, 185, 16),      # #10B981
    "cyan": (255, 240, 0),       # #00F0FF
    "amber": (11, 158, 245),     # #F59E0B
    "crimson": (68, 68, 239),    # #EF4444
    "slate": (139, 116, 100),    # #64748B
    "card_bg": (34, 24, 19),     # #131822
    "border": (45, 38, 33),      # #21262D
}
