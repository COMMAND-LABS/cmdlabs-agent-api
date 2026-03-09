from .session_logger import SessionLogger


async def route_next_speaker(*args, **kwargs):
    from .router import route_next_speaker as _route_next_speaker

    return await _route_next_speaker(*args, **kwargs)


async def stream_agent(*args, **kwargs):
    from .streamer import stream_agent as _stream_agent

    async for token in _stream_agent(*args, **kwargs):
        yield token


__all__ = ["route_next_speaker", "stream_agent", "SessionLogger"]
