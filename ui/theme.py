def inject_accessibility_styles(streamlit) -> None:
    streamlit.markdown(
        """
        <style>
        :focus-visible {
            outline: 3px solid #49a3ff !important;
            outline-offset: 3px !important;
        }

        .stButton > button,
        [data-testid="stDownloadButton"] > button {
            min-height: 2.75rem;
        }

        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
                transition-duration: 0.01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
