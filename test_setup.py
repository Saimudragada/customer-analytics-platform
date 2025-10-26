"""
Test script to verify all dependencies are installed correctly
"""

def test_imports():
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import numpy as np
        print("✅ numpy imported successfully")
        
        import sklearn
        print("✅ scikit-learn imported successfully")
        
        import xgboost as xgb
        print("✅ xgboost imported successfully")
        
        import plotly
        print("✅ plotly imported successfully")
        
        import streamlit as st
        print("✅ streamlit imported successfully")
        
        import openai
        print("✅ openai imported successfully")
        
        import langchain
        print("✅ langchain imported successfully")
        
        print("\n🎉 All dependencies installed successfully!")
        print(f"Python version: {__import__('sys').version}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")

if __name__ == "__main__":
    test_imports()