from .http_agent import HTTPAgent

# FastChatAgent requires torch + fastchat which are heavy optional deps.
# Import it lazily so the rest of the package works without them.
try:
    from .fastchat_client import FastChatAgent
except ImportError:
    pass
