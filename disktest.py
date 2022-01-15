from diskcache import Cache

cache = Cache('caaac')

cache['lol'] = 'hehehaha'
cache.close()


