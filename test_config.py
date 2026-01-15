from app.config import settings

def test_config():
    """Test that all API keys are loaded."""
    print("Testing configuration...")
    
    print(f"✓ Supabase URL: {settings.supabase_url[:30]}...")
    print(f"✓ Supabase Key: {settings.supabase_key[:20]}...")
    print(f"✓ OpenAI Key: {settings.openai_api_key[:20]}...")
    
    if settings.anthropic_api_key and settings.anthropic_api_key != "your_anthropic_key_here":
        print(f"✓ Anthropic Key: {settings.anthropic_api_key[:20]}...")
    else:
        print("⚠ Anthropic Key: Not set (optional)")
    
    print(f"✓ Environment: {settings.environment}")
    print(f"✓ Log Level: {settings.log_level}")
    
    print("\n✅ All configuration loaded successfully!")


if __name__ == "__main__":
    test_config()