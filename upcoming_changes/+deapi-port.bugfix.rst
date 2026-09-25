The deapi face caches its port at start, so a face stopped before its UDP thread binds no longer calls ``getsockname`` on a closed socket.
