from supabase import create_client, Client
from app.config import settings
from functools import lru_cache


@lru_cache()
def get_supabase_client() -> Client:
    """Get cached Supabase client."""
    return create_client(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_key
    )


# For easy imports
supabase = get_supabase_client()