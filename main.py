"""
Main entry point for the Article Generation Application.
This is a simplified interface that uses the refactored modules.
"""
import streamlit as st
from src.app.streamlit_app import create_app

if __name__ == "__main__":
    create_app()